from __future__ import annotations

import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.agents.autonomous import (
    AgentPrivateMemory,
    AgentRuntimeServices,
    ContextAgent,
    CoordinatorAgent,
    ResponseAgent,
    SafetyAgent,
    UnderstandingAgent,
    fallback_response_text,
)
from app.agents.coordinator import EventDrivenCoordinator
from app.agents.events import AgentEvent, AgentEventType, CollaborationBlackboard
from app.agents.registry import AgentRegistry
from app.agents.result import AgentRunResult, AgentStep
from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.core.timezone import now_cn
from app.models.entities import ChatSession, UserAccount
from app.schemas.dtos import AiMessage
from app.services.agent_models import AgentModelRegistry
from app.services.ai import AiClient, PromptTemplates
from app.services.knowledge import KnowledgeService, SearchResult
from app.services.memory import RedisShortTermMemoryStore


# 这是真正的 事件驱动多 Agent 运行时。一次调用会走完一整轮 Agent 协作流程，
class EventDrivenAgentRuntimeService:
    """Actor-style multi-agent runtime.

    Agents observe open tasks, claim work independently, and return the shared
    AgentRunResult contract consumed by the rest of the app.
    """

    framework_name = "event_driven_multi_agent"
    max_steps = 8

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.ai = AiClient(settings)
        self.knowledge = KnowledgeService(db, settings)
        self.memory = RedisShortTermMemoryStore(settings)
        self.model_registry = AgentModelRegistry(settings)
        self.private_memory = AgentPrivateMemory(settings)

    def run(self, user: UserAccount, session: ChatSession, original_input: str, model_input: str) -> AgentRunResult:
        started_at = now_cn()
        # AgentRuntimeServices时共享服务包
        # 把所有外部依赖（数据库、AI 客户端、记忆、知识库、模型注册表）打包，后续各 Agent 共用。
        services = AgentRuntimeServices(
            db=self.db,
            settings=self.settings,
            user=user,
            session=session,
            ai=self.ai,
            model_registry=self.model_registry,
            memory=self.memory,
            private_memory=self.private_memory,
            knowledge=self.knowledge,
            llm_call_records=[],
        )
        # 创建agent
        # CoordinatorAgent：总协调者，负责任务分解、调度、最终采纳。
        # UnderstandingAgent：理解用户意图。
        # SafetyAgent：安全/风险评估。
        # ContextAgent：检索记忆和知识库。
        # ResponseAgent：生成候选回复。
        coordinator_agent = CoordinatorAgent(services)
        agents = [
            UnderstandingAgent(services),
            SafetyAgent(services),
            ContextAgent(services),
            ResponseAgent(services),
        ]
        # 创建协作黑板
        # 黑板是本轮协作的共享状态中心，所有 Agent 都在上面发布事件、任务、artifact（中间产物）。
        board = CollaborationBlackboard(
            turn_id=uuid.uuid4().hex,
            user_id=user.id,
            session_id=session.public_id,
            user_input=original_input,
            model_input=model_input,
        )
        # 发布开始事件
        board = board.append_event(
            AgentEvent(
                type=AgentEventType.TURN_STARTED,
                actor=coordinator_agent.name,
                message="user turn published to shared task board",
            )
        )
        # 启动协调器运行多 Agent 循环
        # AgentRegistry：管理所有业务 Agent。
        # EventDrivenCoordinator：实际的调度循环引擎。它会反复让 Agent 认领任务、执行、发布事件，直到产生被采纳的最终回复或达到最大步数（max_steps = 8）。
        registry = AgentRegistry(agents)
        final_board = EventDrivenCoordinator(registry, coordinator_agent, self.settings).run(board)
        # 运行时异常不在此处吞掉，由 harness 落 FAILED trace 后继续抛出
        return self._to_result(final_board, user, services, started_at)

    def _to_result(
        self,
        board: CollaborationBlackboard,
        user: UserAccount,
        services: AgentRuntimeServices,
        started_at,
    ) -> AgentRunResult:
        intent = self._select_intent(board)
        risk = self._select_risk(board)
        context = board.latest_artifact("context")
        risk_artifact = board.latest_artifact("risk")
        # 只采纳经 Coordinator 接受（即已通过对应版本安全审核）的候选；
        # 未采纳的候选文本绝不能作为最终回复发送（未审核内容不得放行）。
        accepted = board.accepted_artifact()
        memory_brief = "无相关历史记忆。"
        retrieved: list[SearchResult] = []
        response_messages: list[AiMessage] = []
        if context:
            memory_brief = context.payload.get("memoryBrief") or memory_brief
            retrieved = context.payload.get("retrievedKnowledge") or []
        if accepted:
            response_messages = accepted.payload.get("messages") or []
        if not response_messages:
            response_messages = self._fallback_messages(intent, risk, user.display_name, board.model_input)
        # 最终文本必须来自被采纳（已审核）的候选 artifact；未被采纳时
        # （如审核持续不通过、预算耗尽）降级为确定性安全兜底文本，绝不发送未审核内容。
        final_response = (accepted.payload.get("text") or "").strip() if accepted else ""
        if not final_response:
            final_response = fallback_response_text(risk)
        final_review = None
        if accepted:
            final_review = next(
                (
                    artifact
                    for artifact in reversed(board.artifacts)
                    if artifact.kind == "safety_review" and artifact.metadata.get("responseArtifactId") == accepted.id
                ),
                None,
            )
        assessment = risk_artifact.payload.get("assessment") if risk_artifact else None
        collaboration_events = list(board.events)
        collaboration_events.extend(_llm_record_events(services.llm_call_records))
        return AgentRunResult(
            intent=intent,
            risk_level=risk,
            assessment=assessment,
            retrieved_knowledge=retrieved,
            response_messages=response_messages,
            steps=self._events_to_steps(board),
            memory_brief=memory_brief,
            collaboration_events=collaboration_events,
            collaboration_tasks=list(board.tasks.values()),
            collaboration_artifacts=list(board.artifacts),
            final_response=final_response,
            final_response_artifact_id=accepted.id if accepted else "",
            final_review_artifact_id=final_review.id if final_review else "",
            trace_id=board.turn_id,
            status="COMPLETED",
            started_at=started_at,
            completed_at=now_cn(),
            llm_call_records=list(services.llm_call_records),
        )

    def _select_intent(self, board: CollaborationBlackboard) -> IntentType:
        if any(event.type == AgentEventType.SAFETY_OVERRIDE for event in board.events):
            return IntentType.RISK
        artifact = board.latest_artifact("intent")
        if not artifact:
            return IntentType.CHAT
        try:
            return IntentType(str(artifact.payload.get("intent", IntentType.CHAT.value)).upper())
        except ValueError:
            return IntentType.CHAT

    def _select_risk(self, board: CollaborationBlackboard) -> RiskLevel:
        order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
        highest = RiskLevel.LOW
        for artifact in board.artifacts_by_kind("risk"):
            try:
                risk = RiskLevel(str(artifact.payload.get("risk", RiskLevel.LOW.value)).upper())
            except ValueError:
                risk = RiskLevel.LOW
            if order[risk] > order[highest]:
                highest = risk
        if any(event.type == AgentEventType.SAFETY_OVERRIDE for event in board.events):
            return RiskLevel.HIGH
        return highest

    def _fallback_messages(self, intent: IntentType, risk: RiskLevel, display_name: str, model_input: str) -> list[AiMessage]:
        return [
            PromptTemplates.answer_system_prompt(intent, risk, "", display_name),
            AiMessage(role="user", content=model_input),
        ]

    def _events_to_steps(self, board: CollaborationBlackboard) -> list[AgentStep]:
        steps = []
        for index, event in enumerate(board.events, start=1):
            # 决策评估事件不代表该 agent 真的执行了，不进入 legacy steps 视图
            if event.type == AgentEventType.DECISION_EVALUATED:
                continue
            detail = event.message or _compact_json(event.metadata)
            if event.artifact_id:
                detail = f"{detail}; artifact={event.artifact_id}" if detail else f"artifact={event.artifact_id}"
            steps.append(AgentStep(index, event.actor, event.type.value, detail))
        return steps


def _llm_record_events(records: list) -> list[AgentEvent]:
    """把本轮 LLM 调用记录转成 LLM_CALL_COMPLETED/LLM_CALL_FAILED 事件。"""
    events: list[AgentEvent] = []
    for record in records:
        actor = record.get("actor", "") if isinstance(record, dict) else getattr(record, "actor", "")
        metrics = record.get("metrics") if isinstance(record, dict) else record
        status = getattr(metrics, "status", "OK")
        events.append(
            AgentEvent(
                type=AgentEventType.LLM_CALL_COMPLETED if status == "OK" else AgentEventType.LLM_CALL_FAILED,
                actor=actor,
                message=f"llm call model={getattr(metrics, 'model_name', '')} status={status}",
                metadata={
                    "modelName": getattr(metrics, "model_name", ""),
                    "inputTokens": getattr(metrics, "input_tokens", None),
                    "outputTokens": getattr(metrics, "output_tokens", None),
                    "errorType": getattr(metrics, "error_type", None),
                    "retryCount": getattr(metrics, "retry_count", 0),
                    "truncated": getattr(metrics, "truncated", False),
                },
                duration_ms=getattr(metrics, "duration_ms", None),
            )
        )
    return events


def _compact_json(value: Any) -> str:
    jsonable = _to_jsonable(value)
    if not jsonable:
        return ""
    return str(jsonable)[:240]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value
