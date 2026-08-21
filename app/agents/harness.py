from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.agents.events import AgentEvent, AgentEventType
from app.agents.factory import create_agent_runtime
from app.agents.result import AgentRunResult, AgentStep
from app.core.config import Settings
from app.core.enums import IntentType, MessageRole, RiskLevel
from app.core.timezone import now_cn
from app.models.entities import ChatMessage, ChatSession, PsychologicalReport, UserAccount
from app.schemas.dtos import AiMessage, ChatRequest
from app.services.assessment import PsychologyAssessment
from app.services.knowledge import SearchResult
from app.services.mcp_client import MindBridgeMcpToolClient
from app.services.memory import RedisShortTermMemoryStore
from app.services.privacy import PrivacySanitizer
from app.services.tool_queue import ToolQueueService
from app.services.trace import AgentTraceService


@dataclass
class AgentToolPlan:
    report_id: int | None
    risk_level: str | None

    @property
    def requires_tools(self) -> bool:
        return self.report_id is not None


@dataclass
class AgentHarnessOutcome:
    user: UserAccount
    session: ChatSession
    original_input: str
    model_input: str
    intent: IntentType
    risk_level: str | None
    assessment: PsychologyAssessment | None
    response_messages: list[AiMessage]
    agent_steps: list[AgentStep]
    retrieved_knowledge: list[SearchResult]
    report_id: int | None
    tool_plan: AgentToolPlan
    final_response: str
    trace_id: str
    agent_run: AgentRunResult | None
    started_at: datetime | None


class MindBridgeAgentHarness:
    """Runtime harness for one MindBridge agent turn.

    The harness owns business orchestration around the agent runtime. HTTP/SSE
    code can stay thin while this class manages input preparation, persistence,
    risk report creation, tool planning, and trace data.
    """

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self.memory = RedisShortTermMemoryStore(settings)

    def run(self, user: UserAccount, request: ChatRequest) -> AgentHarnessOutcome:
        started_at = now_cn()
        original_input = request.message.strip()
        model_input = self.privacy.sanitize(original_input)
        session = self._resolve_session(user, request.sessionId, original_input)
        try:
            # Agent Runtime 返回AgentRunResult后，Harness 创建报告
            agent_run = create_agent_runtime(self.db, self.settings).run(user, session, original_input, model_input)
        except Exception as exc:
            # 运行时崩溃也要落一条 FAILED trace，方便事后排查，然后继续抛出
            failed_run = AgentRunResult(
                intent=IntentType.CHAT,
                risk_level=RiskLevel.LOW,
                assessment=None,
                retrieved_knowledge=[],
                response_messages=[],
                steps=[],
                memory_brief="",
                trace_id=uuid.uuid4().hex,
                status="FAILED",
                started_at=started_at,
                completed_at=now_cn(),
                error={"type": type(exc).__name__, "message": str(exc)[:500]},
            )
            AgentTraceService(self.db).save_run(
                user=user,
                session=session,
                original_input=original_input,
                sanitized_input=model_input,
                memory_brief="",
                agent_run=failed_run,
                report_id=None,
            )
            raise

        self.save_message(user, session, MessageRole.USER, original_input)

        report = self._create_report(user, session, original_input, agent_run)
        risk_level = report.risk_level if report is not None else None
        # 创建AgentToolPlan，它不是 LLM 生成的，也不是某个 Agent 在 act() 中生成的，而是 Harness 根据报告结果确定性创建的。
        # 只有report_id和risk_level
        tool_plan = AgentToolPlan(report_id=report.id if report is not None else None, risk_level=risk_level)
        return AgentHarnessOutcome(
            user=user,
            session=session,
            original_input=original_input,
            model_input=model_input,
            intent=agent_run.intent,
            risk_level=risk_level,
            assessment=agent_run.assessment,
            response_messages=agent_run.response_messages,
            agent_steps=agent_run.steps,
            retrieved_knowledge=agent_run.retrieved_knowledge,
            report_id=report.id if report is not None else None,
            tool_plan=tool_plan,
            final_response=agent_run.final_response,
            trace_id=agent_run.trace_id,
            agent_run=agent_run,
            started_at=started_at,
        )

    def finalize_trace(self, outcome: AgentHarnessOutcome, tool_records: list[dict], error: Exception | None = None) -> None:
        """回复链路收尾：工具执行事件 + TRACE_COMPLETED/FAILED 落库。"""
        agent_run = outcome.agent_run
        if agent_run is None:
            return
        for record in tool_records:
            agent_run.collaboration_events.append(
                AgentEvent(
                    type=AgentEventType.TOOL_EXECUTION_STARTED,
                    actor="ToolDispatcher",
                    message=str(record.get("kind", "")),
                    metadata={"kind": record.get("kind", ""), "idempotencyKey": record.get("idempotencyKey", "")},
                )
            )
            failed = record.get("status") == "FAILED"
            agent_run.collaboration_events.append(
                AgentEvent(
                    type=AgentEventType.TOOL_EXECUTION_FAILED if failed else AgentEventType.TOOL_EXECUTION_COMPLETED,
                    actor="ToolDispatcher",
                    message=str(record.get("kind", "")),
                    metadata={
                        "kind": record.get("kind", ""),
                        "status": record.get("status", ""),
                        "retryCount": record.get("retryCount", 0),
                        "idempotencyKey": record.get("idempotencyKey", ""),
                        "errorType": record.get("errorType"),
                    },
                    duration_ms=record.get("durationMs"),
                )
            )
        if error is not None:
            agent_run.status = "FAILED"
            agent_run.error = {"type": type(error).__name__, "message": str(error)[:500]}
            agent_run.collaboration_events.append(
                AgentEvent(
                    type=AgentEventType.TRACE_FAILED,
                    actor="MindBridgeAgentHarness",
                    message=str(error)[:200],
                    metadata={"errorType": type(error).__name__},
                )
            )
        else:
            agent_run.status = "COMPLETED"
            agent_run.collaboration_events.append(
                AgentEvent(
                    type=AgentEventType.TRACE_COMPLETED,
                    actor="MindBridgeAgentHarness",
                    message="turn trace finalized",
                )
            )
        agent_run.completed_at = now_cn()
        AgentTraceService(self.db).save_run(
            user=outcome.user,
            session=outcome.session,
            original_input=outcome.original_input,
            sanitized_input=outcome.model_input,
            memory_brief=agent_run.memory_brief,
            agent_run=agent_run,
            report_id=outcome.report_id,
        )

    def save_assistant_message(self, user: UserAccount, session: ChatSession, content: str) -> None:
        self.save_message(user, session, MessageRole.ASSISTANT, content)

# 聊天处理结束后，会根据风险报告调用
    async def dispatch_tools(self, tool_plan: AgentToolPlan) -> list[dict]:
        """执行回复后工具；返回结构化执行记录，任何失败都不中断主流程。"""
        if tool_plan.report_id is None:
            return []
        if self.settings.tool_queue_enabled:
            started = time.perf_counter()
            try:
                # enqueue_report() 根据风险等级创建不同任务：
                # LOW	写入 Excel
                # MEDIUM 写入 Excel、创建风险个案
                # HIGH	写入 Excel、创建风险个案、发送预警
                jobs = ToolQueueService(self.db, self.settings).enqueue_report(tool_plan.report_id, tool_plan.risk_level)
            except Exception as exc:
                return [
                    {
                        "kind": "tool_queue",
                        "status": "FAILED",
                        "durationMs": (time.perf_counter() - started) * 1000.0,
                        "retryCount": 0,
                        "idempotencyKey": f"tool_queue:{tool_plan.report_id}",
                        "errorType": type(exc).__name__,
                        "detail": str(exc)[:200],
                    }
                ]
            duration_ms = (time.perf_counter() - started) * 1000.0
            return [
                {
                    "kind": job.kind,
                    "status": "QUEUED",
                    "durationMs": duration_ms,
                    "retryCount": job.attempts,
                    "idempotencyKey": f"{job.kind}:{tool_plan.report_id}",
                    "errorType": None,
                    "detail": f"job_id={job.id}",
                }
                for job in jobs
            ]
        started = time.perf_counter()
        try:
            results = await MindBridgeMcpToolClient(self.settings).handle_report(tool_plan.report_id, tool_plan.risk_level)
        except Exception as exc:
            return [
                {
                    "kind": "mcp_tool",
                    "status": "FAILED",
                    "durationMs": (time.perf_counter() - started) * 1000.0,
                    "retryCount": 0,
                    "idempotencyKey": f"mcp_tool:{tool_plan.report_id}",
                    "errorType": type(exc).__name__,
                    "detail": str(exc)[:200],
                }
            ]
        duration_ms = (time.perf_counter() - started) * 1000.0
        return [
            {
                "kind": "mcp_tool",
                "status": "COMPLETED",
                "durationMs": duration_ms,
                "retryCount": 0,
                "idempotencyKey": f"mcp_tool:{tool_plan.report_id}",
                "errorType": None,
                "detail": str(result)[:200],
            }
            for result in results
        ]

    def save_message(self, user: UserAccount, session: ChatSession, role: MessageRole, content: str) -> None:
        self.db.add(ChatMessage(user_id=user.id, session_id=session.id, role=role.value, content=content))
        session.touch()
        self.db.add(session)
        self.db.commit()
        self.memory.append(session.public_id, role.value, content)

    def _resolve_session(self, user: UserAccount, public_id: str | None, text: str) -> ChatSession:
        if public_id:
            session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id, ChatSession.user_id == user.id).first()
            if session is None:
                raise ValueError("Session not found")
            return session
        session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=text[:36])
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def _create_report(self, user: UserAccount, session: ChatSession, text: str, agent_run) -> PsychologicalReport | None:
        if not agent_run.requires_report or agent_run.assessment is None:
            return None
        report = PsychologicalReport(
            user_id=user.id,
            session_id=session.id,
            content=text,
            intent=agent_run.intent.value,
            emotion=agent_run.assessment.emotion.value,
            emotion_score=agent_run.assessment.emotion_score,
            risk_level=agent_run.assessment.risk.value,
            confidence=agent_run.assessment.confidence,
            summary=agent_run.assessment.summary,
        )
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        return report
