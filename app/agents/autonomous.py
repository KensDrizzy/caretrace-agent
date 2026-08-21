from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from app.agents.events import (
    AgentArtifact,
    AgentEvent,
    AgentEventType,
    AgentMessage,
    AgentTask,
    AgentTurnResult,
    CollaborationBlackboard,
    TaskPriority,
)
from app.agents.registry import AgentCapability, AgentDecision, AgentProfile
from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.agent_models import AgentModelRegistry
from app.services.ai import AiClient, PromptTemplates, has_consult_signal, has_high_risk_signal
from app.services.assessment import PsychologicalAssessmentService

if TYPE_CHECKING:
    from app.models.entities import ChatSession, UserAccount
    from app.services.knowledge import KnowledgeService, SearchResult
    from app.services.memory import RedisShortTermMemoryStore


GENERAL_TASK_WORDS = [
    "java", "python", "javascript", "代码", "编程", "程序", "算法", "数据库", "spring", "maven",
    "前端", "后端", "项目", "接口", "bug", "报错", "作业", "论文", "翻译", "总结", "解释",
    "怎么写", "如何", "是什么", "为什么", "给我", "帮我", "推荐", "查询", "天气", "路线",
]

CONTEXTUAL_FOLLOW_UP_WORDS = [
    "这个", "那个", "这样", "那样", "刚才", "上面", "有用吗", "有必要吗",
    "为什么", "怎么做", "然后呢", "具体呢", "可以吗", "真的吗", "呢？", "吗？",
]

EXPLICIT_TOPIC_SWITCH_WORDS = [
    "java", "python", "javascript", "代码", "编程", "程序", "算法", "数据库", "spring", "maven",
    "前端", "后端", "接口", "bug", "报错", "作业", "论文", "翻译", "天气", "路线",
]

# SafetyAgent 结构化审核规则词表（只记录 reasonCode，不记录思维链）
BOUNDARY_VIOLATION_WORDS = ["绝对保密", "我确诊你", "你患有"]
INTERNAL_PROCESS_WORDS = [
    "ResponseAgent",
    "CoordinatorAgent",
    "SafetyAgent",
    "ContextAgent",
    "UnderstandingAgent",
    "黑板上的",
    "共享黑板",
    "response_candidate",
    "artifact",
]
INTERNAL_PROCESS_PHRASE_GROUPS = [
    ("候选", "最终是否采纳"),
    ("候选", "协调者"),
    ("候选", "内部审核"),
]
SAFETY_GUIDANCE_WORDS = ["紧急", "可信任", "安全", "求助", "心理中心", "辅导员", "110", "120"]
REVIEW_REASON_CODES = {
    "OK",
    "DISMISSIVE_TONE",
    "DIAGNOSTIC_CLAIM",
    "BOUNDARY_VIOLATION",
    "INTERNAL_PROCESS_DISCLOSURE",
    "UNSAFE_CONTENT",
    "MISSING_SAFETY_GUIDANCE",
}


# 所有 Agent 共享的服务包
@dataclass
class AgentRuntimeServices:
    db: Session
    settings: Settings
    user: UserAccount
    session: ChatSession
    ai: AiClient
    model_registry: AgentModelRegistry
    memory: RedisShortTermMemoryStore
    private_memory: "AgentPrivateMemory"
    knowledge: KnowledgeService
    llm_call_records: list = field(default_factory=list)


# 每个agent独立的记忆隔离层，每个 Agent 只能读写自己的私有记忆
class AgentPrivateMemory:
    """Per-agent memory facade backed by isolated Redis keys."""

    def __init__(self, settings: Settings):
        from app.services.memory import RedisShortTermMemoryStore

        self.store = RedisShortTermMemoryStore(settings)

    def load(self, agent_name: str, session_public_id: str) -> list[AiMessage]:
        return self.store.load_recent(self._key(agent_name, session_public_id))

    def append(self, agent_name: str, session_public_id: str, content: str) -> None:
        self.store.append(self._key(agent_name, session_public_id), "system", content)

    def _key(self, agent_name: str, session_public_id: str) -> str:
        return f"agent:{agent_name}:{session_public_id}"

# 所有业务 Agent 的基类，提供：
# name：从 AgentProfile 读取
# client()：获取当前 Agent 专属的 AI 模型客户端
# private_memory() / remember()：读写私有记忆
# _artifact()：生成 artifact 对象，ID 格式为 {agent}:{kind}:{uuid}

class BaseAutonomousAgent:
    profile: AgentProfile

    def __init__(self, services: AgentRuntimeServices):
        self.services = services

    @property
    def name(self) -> str:
        return self.profile.name

    def client(self) -> AiClient:
        return self.services.model_registry.client_for(self.name, metrics_hook=self._llm_metrics_hook)

    def _llm_metrics_hook(self, metrics) -> None:
        self.services.llm_call_records.append({"actor": self.name, "metrics": metrics})

    def private_memory(self) -> list[AiMessage]:
        return self.services.private_memory.load(self.name, self.services.session.public_id)

    def remember(self, content: str) -> None:
        self.services.private_memory.append(self.name, self.services.session.public_id, content)

    def _artifact(
        self,
        kind: str,
        payload: dict[str, Any],
        task: AgentTask,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentArtifact:
        return AgentArtifact(
            id=f"{self.name}:{kind}:{uuid.uuid4().hex[:10]}",
            owner=self.name,
            kind=kind,
            payload=payload,
            confidence=confidence,
            task_id=task.id,
            metadata=metadata or {},
        )


class UnderstandingAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="UnderstandingAgent",
        capabilities=frozenset({AgentCapability.UNDERSTANDING}), # frozenset 就是不可变的集合，意思是“我的能力范围写死了，不能临时加戏”。
        system_prompt=(
            "你是 UnderstandingAgent。你只负责理解用户当前请求，输出意图、主题、置信度和理由，"
            "不生成最终回复，不做风险处置。"
        ),
        memory_policy="private_intent_history",
        model_profile="understanding",
        tool_permissions=frozenset({"llm.intent"}), # 这个 Agent 被允许调用的工具白名单。这里只给了 llm.intent，意思是“只能调用意图分类相关的能力”，不能去查数据库或发邮件。
    )

    # 如果黑板上已有 intent artifact，不再重复生成
    # 如果当前任务是需要理解的 root/understanding 任务，或任务明确要求 UNDERSTANDING 能力，则认领
    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("intent"):
            return AgentDecision(False, reason="intent artifact already exists", reason_code="INTENT_EXISTS")
        if self._is_directed(task, board):
            return AgentDecision(True, 0.82, "open user-turn task needs understanding", reason_code="UNDERSTANDING_REQUIRED")
        return AgentDecision(False, reason="task does not need understanding", reason_code="UNDERSTANDING_NOT_REQUIRED")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        intent = self._classify(board.model_input or board.user_input, board)
        current_intent = self._current_intent(board.model_input or board.user_input)
        continuation = self._is_contextual_follow_up(board.model_input or board.user_input, board)
        confidence = 0.92 if intent == IntentType.RISK else 0.78
        payload = {
            "intent": intent.value,
            "currentIntent": current_intent.value,
            "continuation": continuation,
            "topic": self._topic(board.model_input or board.user_input),
            "reason": "high risk hard signal" if intent == IntentType.RISK else "autonomous intent proposal",
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        self.remember(f"intent={intent.value}; topic={payload['topic']}")
        return AgentTurnResult(
            artifacts=(self._artifact("intent", payload, task, confidence),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="*",
                    task_id=task.id,
                    kind="PROPOSAL",
                    content=f"我判断本轮意图是 {intent.value}",
                ),
            ),
        )

    def _is_directed(self, task: AgentTask, board: CollaborationBlackboard) -> bool:
        if AgentCapability.UNDERSTANDING.value in task.required_capabilities:
            return True
        return bool(board.user_input and task.metadata.get("kind") in {"root", "understanding"})

    def _classify(self, text: str, board: CollaborationBlackboard) -> IntentType:
        lowered = text.lower()
        # 高风险关键词
        if has_high_risk_signal(lowered):
            return IntentType.RISK
        if has_consult_signal(lowered):
            return IntentType.CONSULT
        if self._is_contextual_follow_up(text, board) and self._history_has_support_topic(board):
            return IntentType.CONSULT
        # 明确的新普通任务不继承上一轮支持主题。
        if any(word in lowered for word in GENERAL_TASK_WORDS):
            return IntentType.CHAT
        try:
            # 如果前面的规则都无法确定，就调用模型：
            memory_context = "\n".join(item.content for item in self.private_memory()[-6:])
            messages = [
                *PromptTemplates.intent_prompt(list(board.conversation_history), text),
                AiMessage(role="system", content=f"{self.profile.system_prompt}\n私有记忆：\n{memory_context or '无'}"),
            ]
            label = self.client().complete(messages).upper()
            if "RISK" in label:
                return IntentType.RISK
            if "CONSULT" in label:
                return IntentType.CONSULT
            if "CHAT" in label:
                # 当本地模型把明显的心理求助信号误判为闲聊时，用规则兜底
                if has_consult_signal(lowered):
                    return IntentType.CONSULT
                return IntentType.CHAT
        except Exception:
            pass
        return IntentType.CONSULT if has_consult_signal(lowered) else IntentType.CHAT

    def _current_intent(self, text: str) -> IntentType:
        lowered = text.lower()
        if has_high_risk_signal(lowered):
            return IntentType.RISK
        if has_consult_signal(lowered):
            return IntentType.CONSULT
        return IntentType.CHAT

    def _is_contextual_follow_up(self, text: str, board: CollaborationBlackboard) -> bool:
        normalized = text.strip().lower()
        if not board.conversation_history or len(normalized) > 36:
            return False
        if any(word in normalized for word in EXPLICIT_TOPIC_SWITCH_WORDS):
            return False
        return any(word in normalized for word in CONTEXTUAL_FOLLOW_UP_WORDS)

    def _history_has_support_topic(self, board: CollaborationBlackboard) -> bool:
        recent_user_text = "\n".join(
            item.content for item in board.conversation_history[-8:] if item.role == "user"
        )
        return has_consult_signal(recent_user_text) or has_high_risk_signal(recent_user_text)

    def _topic(self, text: str) -> str:
        lowered = text.lower()
        if has_high_risk_signal(lowered):
            return "safety"
        if has_consult_signal(lowered):
            return "mental_health_support"
        if any(word in lowered for word in GENERAL_TASK_WORDS):
            return "general_task"
        return "conversation"


class SafetyAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="SafetyAgent",
        capabilities=frozenset({AgentCapability.SAFETY}),
        system_prompt=(
            "你是 SafetyAgent。你独立评估风险，并审查候选回复是否安全。"
            "你可以发布 SAFETY_OVERRIDE；你不生成最终回复。"
        ),
        memory_policy="private_safety_ledger",
        model_profile="safety",
        tool_permissions=frozenset({"llm.risk", "rules.high_risk", "response.review"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        latest_response = board.latest_artifact("response_candidate")
        if latest_response and _latest_review_for(board, latest_response.id) is None:
            return AgentDecision(True, 0.95, "candidate response needs safety critique", reason_code="REVIEW_REQUIRED")
        if not board.latest_artifact("risk") and board.user_input:
            confidence = 0.98 if has_high_risk_signal(board.user_input) else 0.84
            return AgentDecision(True, confidence, "user input needs independent risk assessment", reason_code="RISK_ASSESSMENT_REQUIRED")
        if AgentCapability.SAFETY.value in task.required_capabilities:
            return AgentDecision(True, 0.8, "task explicitly asks for safety", reason_code="TASK_REQUIRES_SAFETY")
        return AgentDecision(False, reason="no safety work needed", reason_code="NO_SAFETY_WORK")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        response = board.latest_artifact("response_candidate")
        if response and _latest_review_for(board, response.id) is None:
            return self._review_response(task, board, response)
        return self._assess_risk(task, board)

    def _assess_risk(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        assessment = PsychologicalAssessmentService(self.client()).assess(board.model_input or board.user_input, _context_history(board))
        payload = {
            "risk": assessment.risk.value,
            "emotion": assessment.emotion.value,
            "emotionScore": assessment.emotion_score,
            "confidence": assessment.confidence,
            "summary": assessment.summary,
            "assessment": assessment,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        events: tuple[AgentEvent, ...] = ()
        # 如果评估出来的风险是high，则触发SAFETY_OVERRIDE，也就是说，即使其他 artifact 中有较低风险判断，只要出现 SAFETY_OVERRIDE：最终意图 = RISK，最终风险 = HIGH
        # 这个事件非常重要，因为后续的风险选择逻辑会把它视为最高优先级：
        if assessment.risk == RiskLevel.HIGH:
            events = (
                AgentEvent(
                    type=AgentEventType.SAFETY_OVERRIDE,
                    actor=self.name,
                    task_id=task.id,
                    message="RiskGuardian hard/LLM assessment raised this turn to HIGH",
                    metadata={"risk": RiskLevel.HIGH.value},
                    round=board.current_round,
                ),
            )
        self.remember(f"risk={assessment.risk.value}; summary={assessment.summary}")
        return AgentTurnResult(
            artifacts=(self._artifact("risk", payload, task, assessment.confidence),),
            events=events,
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="CoordinatorAgent",
                    task_id=task.id,
                    kind="SAFETY_ASSESSMENT",
                    content=f"risk={assessment.risk.value}",
                ),
            ),
        )

    def _review_response(self, task: AgentTask, board: CollaborationBlackboard, response: AgentArtifact) -> AgentTurnResult:
        risk = _risk_level(board)
        text = str(response.payload.get("text") or "")
        approved = True
        reason_code = "OK"
        reason = "response candidate satisfies current safety constraints"
        review_mode = "rules"
        if not text.strip():
            approved = False
            reason_code = "EMPTY_RESPONSE"
            reason = "response candidate text is empty"
        elif any(word in text for word in BOUNDARY_VIOLATION_WORDS):
            approved = False
            reason_code = "BOUNDARY_VIOLATION"
            reason = "response candidate contains boundary or diagnosis wording"
        elif _contains_internal_process_disclosure(text):
            approved = False
            reason_code = "INTERNAL_PROCESS_DISCLOSURE"
            reason = "response candidate exposes internal agent or review workflow"
        elif risk == RiskLevel.HIGH and not any(word in text for word in SAFETY_GUIDANCE_WORDS):
            approved = False
            reason_code = "MISSING_SAFETY_GUIDANCE"
            reason = "high-risk response candidate lacks immediate safety guidance"
        if approved:
            # 规则只能兜住确定性问题；语气贬低、隐性诊断、危险认同等需要语义审核。
            # 模型审核不可用时回退为纯规则结论，绝不因审核服务故障阻断安全回复。
            llm_verdict = self._llm_semantic_review(text, risk)
            if llm_verdict is not None:
                review_mode = "rules+llm"
                llm_approved, llm_reason_code, llm_reason = llm_verdict
                if not llm_approved:
                    approved = False
                    reason_code = llm_reason_code
                    reason = llm_reason
        payload = {
            "approved": approved,
            "reasonCode": reason_code,
            "reason": reason,
            "reviewMode": review_mode,
            "responseArtifactId": response.id,
            "responseArtifactVersion": response.version,
            "risk": risk.value,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        kind = "safety_review" if approved else "critique"
        events: tuple[AgentEvent, ...] = (
            AgentEvent(
                type=AgentEventType.FINAL_RESPONSE_REVIEWED,
                actor=self.name,
                task_id=task.id,
                artifact_id=response.id,
                message=reason,
                metadata={
                    "approved": approved,
                    "reasonCode": reason_code,
                    "responseArtifactId": response.id,
                    "responseArtifactVersion": response.version,
                },
                round=board.current_round,
            ),
        )
        follow_up_tasks = ()
        if not approved:
            events = (
                *events,
                AgentEvent(
                    type=AgentEventType.REVISION_REQUESTED,
                    actor=self.name,
                    task_id=task.id,
                    artifact_id=response.id,
                    message=reason,
                    metadata={"reasonCode": reason_code},
                    round=board.current_round,
                ),
            )
            # 修订任务统一由 Coordinator 根据 critique 派生（确定性 id，天然去重）。
            # SafetyAgent 不再自行创建，避免同一否决产生两个修订任务、生成两个
            # 同版本候选导致审核/采纳 id 错位并烧掉认领预算。
        self.remember(f"review approved={approved}; reasonCode={reason_code}")
        return AgentTurnResult(
            artifacts=(
                self._artifact(
                    kind,
                    payload,
                    task,
                    0.95,
                    {
                        "responseArtifactId": response.id,
                        "responseArtifactVersion": response.version,
                        "reasonCode": reason_code,
                        "approved": approved,
                    },
                ),
            ),
            tasks=follow_up_tasks,
            events=events,
        )

    def _llm_semantic_review(self, text: str, risk: RiskLevel) -> tuple[bool, str, str] | None:
        """LLM 语义审核：捕捉规则覆盖不到的贬低语气、隐性诊断、危险认同等。

        只输出结构化 reasonCode 与证据片段，不记录任何思维链。
        返回 None 表示审核模型不可用，调用方回退为纯规则结论。
        """
        try:
            raw = self.client().complete([
                AiMessage(role="system", content=(
                    f"{self.profile.system_prompt}\n"
                    "你正在审核一段即将发送给学生的回复。只输出一个 JSON 对象，格式："
                    '{"approved": true|false, "reasonCode": "OK|DISMISSIVE_TONE|DIAGNOSTIC_CLAIM|'
                    'BOUNDARY_VIOLATION|INTERNAL_PROCESS_DISCLOSURE|UNSAFE_CONTENT|MISSING_SAFETY_GUIDANCE", '
                    '"evidence": "简短证据片段"}\n'
                    "拒绝标准：贬低/说教/否定用户感受；直接下诊断或建议用药；承诺绝对保密或越界关系；"
                    "认同、鼓励或合理化用户的自伤念头；提供任何自伤/伤人方法细节；"
                    "声称自己是某个内部 Agent，或解释黑板、候选生成、内部审核、采纳、artifact 等幕后协作流程；"
                    "仅因用户提到“候选回复”而道歉并说明是不应出现的格式错误，不属于泄露；"
                    "高风险情境下没有优先关注即时安全与求助渠道。只给结论，不输出分析过程。"
                )),
                AiMessage(role="user", content=f"当前风险等级：{risk.value}\n\n待审核回复：\n{text}"),
            ])
            data = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
            approved = bool(data.get("approved", True))
            reason_code = str(data.get("reasonCode") or "OK").upper()
            if reason_code not in REVIEW_REASON_CODES:
                reason_code = "OK" if approved else "UNSAFE_CONTENT"
            evidence = str(data.get("evidence") or "")[:120]
            if approved:
                return True, "OK", "llm semantic review passed"
            return False, reason_code, f"llm semantic review rejected: {reason_code}; evidence={evidence}"
        except Exception:
            return None


class ContextAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="ContextAgent",
        capabilities=frozenset({AgentCapability.CONTEXT}),
        system_prompt=(
            "你是 ContextAgent。你只负责为本轮协作提供上下文，包括私有记忆、会话摘要、RAG 证据和 skill 约束。"
            "你不判断最终答案是否可采纳。"
        ),
        memory_policy="private_context_memory",
        model_profile="context",
        tool_permissions=frozenset({"redis.memory", "mysql.messages", "rag.retrieve", "skills.read"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("context"):
            return AgentDecision(False, reason="context artifact already exists", reason_code="CONTEXT_EXISTS")
        if AgentCapability.CONTEXT.value in task.required_capabilities:
            return AgentDecision(True, 0.86, "task explicitly asks for context", reason_code="TASK_REQUIRES_CONTEXT")
        # Don't claim the root task based only on keyword signals before UnderstandingAgent
        # has published an intent artifact; wait for the coordinator to create a context task.
        if task.metadata.get("kind") == "root" and board.latest_artifact("intent") is None:
            return AgentDecision(False, reason="waiting for intent artifact before claiming root task", reason_code="WAITING_FOR_INTENT")
        risk = _risk_level(board)
        intent = _intent(board)
        if risk in {RiskLevel.MEDIUM, RiskLevel.HIGH} or intent in {IntentType.CONSULT, IntentType.RISK}:
            return AgentDecision(True, 0.82, "support path needs memory, RAG, and skill context", reason_code="SUPPORT_PATH_NEEDS_CONTEXT")
        return AgentDecision(False, reason="context not necessary for current artifacts", reason_code="CONTEXT_NOT_REQUIRED")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        from app.services.skills import MindBridgeSkillLibrary

        # 读取会话历史和摘要，从共享黑板取出本轮开始时加载好的历史消息。
        history = list(board.conversation_history)
        deterministic_brief = board.memory_brief
        # 让 ContextAgent 调用 LLM 【重新生成】更适合上下文检索的摘要：
        memory_brief = self._summarize_memory(history, board.model_input, deterministic_brief)
        model_history = list(board.model_history) or [AiMessage(role="user", content=board.model_input)]
        intent = _intent(board)
        risk = _risk_level(board)

        # 初始化检索相关变量
        retrieved: list["SearchResult"] = [] # 检索结果
        query = "" # 检索查询词
        skill_context = "" # skill 上下文
        iterations = 0 # RAG 执行次数
        events: tuple[AgentEvent, ...] = ()
        # 判断是否需要 RAG
        if intent != IntentType.CHAT or risk != RiskLevel.LOW:
            # 记录检索开始时间
            retrieval_started = time.perf_counter()
            try:
                query, retrieved, iterations = self._iterative_retrieve(memory_brief, board.model_input)
            except Exception as exc:
                # RAG 失败不阻断链路：降级为空检索结果继续
                # 如果知识库、向量检索或查询改写失败：
                # 不让整个 Agent 流程失败；
                # 查询词退化为当前输入前 60 个字符；
                # 检索结果置空；
                # 迭代次数置为 0。  board.user_input  = 原始用户输入；board.model_input = 脱敏后的用户输入

                # 让query退化为当前输入前 60 个字符的原因：
                # 保留用户真正的问题；
                # 避免因为改写失败而把 query 设为空；
                # 限制查询长度，避免过长文本进入检索或日志；
                # 方便 trace 记录本次原始检索意图。
                query, retrieved, iterations = board.model_input[:60], [], 0
                events = (
                    AgentEvent(
                        type=AgentEventType.RAG_RETRIEVAL_FAILED,
                        actor=self.name,
                        task_id=task.id,
                        message="knowledge retrieval failed; continuing without RAG evidence",
                        metadata={"errorType": type(exc).__name__, "durationMs": (time.perf_counter() - retrieval_started) * 1000.0},
                        duration_ms=(time.perf_counter() - retrieval_started) * 1000.0,
                        round=board.current_round,
                    ),
                )
            else:
                events = (
                    AgentEvent(
                        type=AgentEventType.RAG_RETRIEVAL_COMPLETED,
                        actor=self.name,
                        task_id=task.id,
                        message=f"retrieved {len(retrieved)} knowledge chunks",
                        metadata={
                            "resultCount": len(retrieved),
                            "durationMs": (time.perf_counter() - retrieval_started) * 1000.0,
                            "iterations": iterations,
                        },
                        duration_ms=(time.perf_counter() - retrieval_started) * 1000.0,
                        round=board.current_round,
                    ),
                )
            # 加载回复技能上下文
            # 根据当前用户的意图、风险等级和原始输入，选择对应的几个 Skill 文件，
            # 并把这些文件的内容拼成一段字符串，保存到 skill_context 变量中。
            skill_context = MindBridgeSkillLibrary.response_skill_context(intent, risk, board.user_input)
        payload = {
            "memoryBrief": memory_brief,
            "modelHistory": model_history,
            "knowledgeQuery": query,
            "retrievedKnowledge": retrieved,
            "skillContext": skill_context,
            "agenticRagIterations": iterations,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        # 写入context-agent的私有记忆，记录了本次context-agent的工作摘要
        self.remember(f"context intent={intent.value}; risk={risk.value}; retrieved={len(retrieved)}; ragIterations={iterations}")
        return AgentTurnResult(
            artifacts=(self._artifact("context", payload, task, 0.88),),
            events=events,
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="ResponseAgent",
                    task_id=task.id,
                    kind="CONTEXT_READY",
                    content=f"context ready; retrieved={len(retrieved)}",
                ),
            ),
        )

    def _rewrite_query(self, memory_brief: str, model_input: str) -> str:
        try:
            query = self.client().complete([
                AiMessage(role="system", content=f"{self.profile.system_prompt}\n把学生输入改写成适合检索校园心理知识库的中文查询词，只输出查询词。"),
                AiMessage(role="user", content=f"记忆摘要：\n{memory_brief}\n\n当前输入：\n{model_input}"),
            ]).strip()
            return (query or model_input)[:60]
        except Exception:
            return model_input[:60]

    def _summarize_memory(self, history: list[AiMessage], current_input: str, fallback: str) -> str:
        # 从配置读取摘要最大长度，至少保留 120 个字符。
        max_chars = max(120, self.services.settings.memory_summary_max_chars)
        # 没有历史消息
        if not history:
            return "无相关历史记忆。"
        # 提示模型只输出 1～3 条中文记忆要点，不输出风险等级或诊断。
        # 将当前输入和最近 12 条历史消息发送给模型
        # 摘要超过限制时截断
        try:
            summary = self.client().complete([
                AiMessage(role="system", content=f"{self.profile.system_prompt}\n只输出 1-3 条中文记忆要点，不输出风险等级或诊断。"),
                AiMessage(role="user", content=f"当前输入：\n{current_input}\n\n最近历史：\n{history[-12:]}"),
            ]).strip()
            return summary[:max_chars] or fallback
        except Exception:
            return fallback or "无相关历史记忆。"

    def _iterative_retrieve(self, memory_brief: str, model_input: str) -> tuple[str, list["SearchResult"], int]:
        """Agentic RAG: rewrite query, retrieve, evaluate sufficiency, and iterate if needed."""
        settings = self.services.settings
        top_k = settings.knowledge_top_k
        max_iterations = max(1, settings.agentic_rag_max_iterations) # 表示最多检索多少轮

        # 如果关闭了智能迭代 RAG，流程很简单：
        # 当前输入 + 记忆摘要
        #         ↓
        # LLM 改写查询词
        #         ↓
        # 知识库检索一次
        #         ↓
        # 返回结果
        if not settings.agentic_rag_enabled:
            # 调用 LLM，把用户输入改成更适合知识库检索的查询词
            query = self._rewrite_query(memory_brief, model_input)
            # 使用改写后的 query 查询知识库，最多返回 top_k 条结果。
            return query, self.services.knowledge.retrieve(query, top_k), 1

        # 开启agentic rag
        current_query = self._rewrite_query(memory_brief, model_input)
        all_results: list["SearchResult"] = [] # 保存所有轮次累计得到的结果。为什么不是每一轮覆盖？因为第二轮可能找到第一轮没有找到的新资料，需要把多轮结果合并起来。
        seen_keys: set[tuple[str, str]] = set() # 记录已经出现过
        query = current_query

        # 开始循环检索
        for iteration in range(1, max_iterations + 1):
            batch = self.services.knowledge.retrieve(current_query, top_k) # 执行一次普通的知识检索，Agentic RAG 没有把你之前的 Hybrid RAG 替换掉。
            for item in batch:
                key = (item.source, item.content)  # 只要来源和正文都相同，就认为是重复结果。
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(item)

            # 判断检索结果是否足够
            # 使用启发式词语覆盖率判断：
            # 1.从用户输入提取关键词；
            # 2.从检索结果中提取关键词；
            # 3.计算用户关键词被结果覆盖的比例；
            # 4.如果覆盖率达到阈值，就认为足够。
            if self._is_sufficient(model_input, all_results):
                return current_query, self._rank_dedup_results(all_results, top_k), iteration

            # 结果不足时重新改写查询
            # 这次不是普通改写，而是针对“当前资料缺少什么”来改写查询。
            if iteration < max_iterations:
                current_query = self._rewrite_for_gap(memory_brief, model_input, all_results)
                if not current_query or current_query == query:
                    break
                query = current_query

        return query, self._rank_dedup_results(all_results, top_k), max_iterations

    def _is_sufficient(self, model_input: str, results: list["SearchResult"]) -> bool:
        """Heuristic sufficiency check before paying for an LLM judgment."""
        if not results:
            return False
        threshold = max(0.0, min(1.0, self.services.settings.agentic_rag_sufficiency_threshold)) # 0.7
        # 提取用户问题的关键词，_extract_terms() 最终调用项目的 tokenize()，转换成set，方便计算交集
        query_terms = set(self._extract_terms(model_input))
        # 用户问题没有关键词时，只要有检索结果，就认为足够
        if not query_terms:
            return len(results) > 0

        best_coverage = 0.0
        for item in results:
            # 将先前检索得到的知识片段正文也进行分词
            item_terms = set(self._extract_terms(item.content))
            # 计算关键词覆盖率
            if item_terms:
                # 计算用户输入关键词在每个检索结果中的覆盖率【计算集合的交集】
                coverage = len(query_terms & item_terms) / len(query_terms)
                best_coverage = max(best_coverage, coverage)
            # 如果某一条结果的覆盖率达到阈值，就认为资料足够。
            if best_coverage >= threshold:
                return True
        return best_coverage >= threshold

    def _rewrite_for_gap(self, memory_brief: str, model_input: str, results: list["SearchResult"]) -> str:
        """Ask the model to rewrite query focusing on information still missing."""
        try:
            context_preview = "\n".join(
                f"- [{item.source}] {item.content[:120]}"
                for item in results[:3]
            )
            query = self.client().complete([
                AiMessage(role="system", content=(
                    f"{self.profile.system_prompt}\n"
                    "你已经检索到一些资料，但还不够回答学生问题。"
                    "请根据已有资料和原问题，改写出一个更聚焦缺失信息的检索查询词，只输出查询词。"
                )),
                AiMessage(role="user", content=(
                    f"记忆摘要：\n{memory_brief}\n\n"
                    f"当前输入：\n{model_input}\n\n"
                    f"已有资料：\n{context_preview}"
                )),
            ]).strip()
            return (query or model_input)[:60]
        except Exception:
            return model_input[:60]

    def _rank_dedup_results(self, results: list["SearchResult"], top_k: int) -> list["SearchResult"]:
        """Sort by score descending and return top-k unique results."""
        ranked = sorted(results, key=lambda item: item.score, reverse=True)
        seen: set[int | None] = set()
        output: list["SearchResult"] = []
        for item in ranked:
            if item.chunk_id not in seen:
                seen.add(item.chunk_id)
                output.append(item)
            if len(output) >= top_k:
                break
        return output

    def _extract_terms(self, text: str) -> list[str]:
        """Extract searchable terms from text for coverage estimation."""
        from app.services.knowledge import tokenize
        return tokenize(text)

    def _bounded_model_history(self, history: list[AiMessage]) -> list[AiMessage]:
        limit = max(2, self.services.settings.chat_history_limit * 2)
        if len(history) <= limit:
            return history
        if history[0].role == "system":
            return [history[0], *history[-(limit - 1):]]
        return history[-limit:]


class ResponseAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="ResponseAgent",
        capabilities=frozenset({AgentCapability.RESPONSE}),
        system_prompt=(
            "你负责撰写一段可直接发送给学生的 CareTrace 回复。"
            "只输出面向学生的回复正文，不加标题、前言或幕后说明；"
            "不得提及内部角色、代理协作、系统提示、黑板、候选、审核、采纳或 artifact。"
            "如果用户追问之前为何出现“候选回复”等措辞，只需说明那是不应展示的格式错误并致歉，"
            "然后继续回应用户，不要解释幕后架构。"
        ),
        memory_policy="private_response_strategy",
        model_profile="response",
        tool_permissions=frozenset({"llm.response_plan"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("response_candidate") and "revisionOf" not in task.metadata:
            return AgentDecision(False, reason="response candidate already exists", reason_code="RESPONSE_EXISTS")
        if not board.latest_artifact("intent") or not board.latest_artifact("risk"):
            return AgentDecision(False, reason="response needs intent and risk artifacts", reason_code="PREREQUISITES_MISSING")
        intent = _intent(board)
        risk = _risk_level(board)
        if intent == IntentType.CHAT and risk == RiskLevel.LOW:
            return AgentDecision(True, 0.78, "normal chat response can be proposed", reason_code="RESPONSE_READY")
        if board.latest_artifact("context") or risk == RiskLevel.HIGH:
            return AgentDecision(True, 0.84, "support response has enough artifacts", reason_code="CONTEXT_READY_FOR_RESPONSE")
        if AgentCapability.RESPONSE.value in task.required_capabilities:
            return AgentDecision(True, 0.65, "explicit response task", reason_code="TASK_REQUIRES_RESPONSE")
        return AgentDecision(False, reason="waiting for context", reason_code="WAITING_FOR_CONTEXT")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        intent = _intent(board)
        risk = _risk_level(board)
        context = board.latest_artifact("context")
        context_payload = context.payload if context else {}
        model_history = context_payload.get("modelHistory") or list(board.model_history) or [AiMessage(role="user", content=board.model_input)]
        memory_brief = context_payload.get("memoryBrief") or board.memory_brief or "无相关历史记忆。"
        knowledge = context_payload.get("retrievedKnowledge") or []
        skill_context = context_payload.get("skillContext") or ""
        knowledge_context = "\n\n".join(f"- [{item.source}] {item.content}" for item in knowledge)
        if intent == IntentType.CHAT and risk == RiskLevel.LOW:
            messages = [
                PromptTemplates.answer_system_prompt(IntentType.CHAT, RiskLevel.LOW, "", self.services.user.display_name),
                AiMessage(
                    role="system",
                    content=(
                        f"{self.profile.system_prompt}\n"
                        "请自然、直接地回答用户。你的整段输出会原样展示给学生。\n"
                        f"私有记忆：\n{_format_private_memory(self.private_memory())}\n"
                        f"记忆摘要：\n{memory_brief}"
                    ),
                ),
                *model_history,
            ]
            mode = "normal_chat"
        else:
            messages = [
                PromptTemplates.answer_system_prompt(
                    intent if intent != IntentType.CHAT else IntentType.CONSULT,
                    risk,
                    knowledge_context,
                    self.services.user.display_name,
                    skill_context,
                ),
                AiMessage(
                    role="system",
                    content=(
                        f"{self.profile.system_prompt}\n"
                        "请共情、具体地回应用户。你的整段输出会原样展示给学生。\n"
                        f"私有记忆：\n{_format_private_memory(self.private_memory())}\n"
                        f"记忆摘要：\n{memory_brief}"
                    ),
                ),
                *model_history,
            ]
            mode = "support"
        # ResponseAgent 直接生成完整候选文本；后续 SafetyAgent 审核、ChatService 发送的都是这份文本。
        llm_started = time.perf_counter()
        degraded = False
        error_type = None
        truncated = False
        try:
            text = self.client().complete(messages).strip()
            records = self.services.llm_call_records
            if records:
                latest_metrics = records[-1].get("metrics") if isinstance(records[-1], dict) else records[-1]
                truncated = bool(getattr(latest_metrics, "truncated", False))
        except Exception as exc:
            degraded = True
            error_type = type(exc).__name__
            text = fallback_response_text(risk)
        if not text:
            degraded = True
            text = fallback_response_text(risk)
        if truncated:
            # 输出被 max_tokens 截断：文本不完整但已生成，标记进 trace 供观测与评测，
            # 不降级（截断文本仍是模型真实输出，由安全审核决定是否可用）。
            self.remember("response truncated by max_tokens")
        llm_duration_ms = (time.perf_counter() - llm_started) * 1000.0
        revision_of = str(task.metadata.get("revisionOf") or "")
        previous = next((artifact for artifact in board.artifacts if artifact.id == revision_of), None) if revision_of else None
        version = previous.version + 1 if previous else 1
        payload = {
            "text": text,
            "messages": messages,
            "mode": mode,
            "intent": intent.value,
            "risk": risk.value,
            "responseAgent": self.name,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        if degraded:
            payload["degraded"] = True
            payload["errorType"] = error_type
        if truncated:
            payload["truncated"] = True
        metadata = {"mode": mode, "version": version}
        if revision_of:
            metadata["revisionOf"] = revision_of
        if degraded:
            metadata["degraded"] = True
            metadata["errorType"] = error_type
        if truncated:
            metadata["truncated"] = True
        artifact = replace(self._artifact("response_candidate", payload, task, 0.86, metadata), version=version)
        self.remember(f"response mode={mode}; intent={intent.value}; risk={risk.value}; degraded={degraded}")
        return AgentTurnResult(
            artifacts=(artifact,),
            events=(
                AgentEvent(
                    type=AgentEventType.FINAL_RESPONSE_GENERATED,
                    actor=self.name,
                    task_id=task.id,
                    artifact_id=artifact.id,
                    message=f"response candidate generated mode={mode} degraded={degraded} truncated={truncated}",
                    metadata={"mode": mode, "degraded": degraded, "version": version, "truncated": truncated},
                    duration_ms=llm_duration_ms,
                    round=board.current_round,
                ),
            ),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="SafetyAgent",
                    task_id=task.id,
                    kind="REVIEW_REQUEST",
                    content="请审查候选回复方案。",
                ),
            ),
        )


class CoordinatorAgent(BaseAutonomousAgent):
    # AgentProfile：Agent 的“静态身份证 + 配置描述”。
    profile = AgentProfile(
        name="CoordinatorAgent",
        capabilities=frozenset({AgentCapability.COORDINATION}),
        system_prompt=(
            "你是 CoordinatorAgent。你不规定固定 Agent 顺序；你只维护任务板、预算、安全门槛、冲突仲裁和最终采纳。"
        ),
        memory_policy="private_coordination_trace",
        model_profile="coordinator",
        tool_permissions=frozenset({"taskboard.write", "blackboard.accept"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        return AgentDecision(False, reason="CoordinatorAgent is driven by the event loop, not by fixed workflow slots", reason_code="EVENT_LOOP_DRIVEN")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        return AgentTurnResult(close_task=False)

    def root_task(self, board: CollaborationBlackboard) -> AgentTask:
        return AgentTask(
            id="task:root",
            title="Resolve user turn",
            description=board.user_input,
            priority=TaskPriority.CRITICAL if has_high_risk_signal(board.user_input) else TaskPriority.NORMAL,
            created_by=self.name,
            metadata={"kind": "root"},
        )

    def remember_acceptance(self, artifact_id: str, reason: str) -> None:
        self.remember(f"accepted={artifact_id}; reason={reason}")


def fallback_response_text(risk: RiskLevel) -> str:
    """确定性安全兜底文本：LLM 调用失败或无候选文本时使用，不再调模型。"""
    if risk == RiskLevel.HIGH:
        return (
            "我听见你现在非常难受，这种痛苦不该一个人扛。请马上联系身边可信任的人，"
            "或立刻联系辅导员、学校心理中心；如果情况紧急，请拨打当地紧急求助电话（110/120）。"
            "先把自己移到有人的地方，把可能伤害自己的东西放远一点。我会一直在这里陪你。"
        )
    if risk == RiskLevel.MEDIUM:
        return (
            "我听见你最近承受了不少压力。先做几次缓慢的呼吸，把最担心的一件事写下来，"
            "只挑一个最小的步骤先处理。如果这种状态持续，建议联系学校心理中心或辅导员一起看一看。"
        )
    return (
        "我在。你可以先说说现在最具体的困扰，我们一步一步拆开来看。"
        "如果感觉撑不住，请马上联系身边可信任的人或学校心理中心。"
    )


def _contains_internal_process_disclosure(text: str) -> bool:
    lowered = text.lower()
    if any(word.lower() in lowered for word in INTERNAL_PROCESS_WORDS):
        return True
    return any(all(fragment.lower() in lowered for fragment in group) for group in INTERNAL_PROCESS_PHRASE_GROUPS)


def _latest_review_for(board: CollaborationBlackboard, response_id: str):
    """查找引用指定候选的最新审核结论（safety_review 或 critique）。

    critique（审核未通过）同样是"已审核"：被否决的候选不应被重复审核，
    否则每轮都会对同一旧候选重复发 critique 并烧掉认领预算。
    """
    for artifact in reversed(board.artifacts):
        if artifact.kind in {"safety_review", "critique"} and artifact.metadata.get("responseArtifactId") == response_id:
            return artifact
    return None


def _intent(board: CollaborationBlackboard) -> IntentType:
    artifact = board.latest_artifact("intent")
    if artifact:
        try:
            return IntentType(str(artifact.payload.get("intent", IntentType.CHAT.value)).upper())
        except ValueError:
            return IntentType.CHAT
    if has_high_risk_signal(board.user_input):
        return IntentType.RISK
    if has_consult_signal(board.user_input):
        return IntentType.CONSULT
    return IntentType.CHAT


def _risk_level(board: CollaborationBlackboard) -> RiskLevel:
    highest = RiskLevel.LOW
    order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
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


def _context_history(board: CollaborationBlackboard) -> list[AiMessage]:
    # PsychologyAssessmentService receives the current input separately, so
    # provide prior turns only here to avoid duplicating the current message.
    return list(board.conversation_history)


def _format_private_memory(items: list[AiMessage]) -> str:
    if not items:
        return "无"
    return "\n".join(f"- {item.content}" for item in items[-5:])
