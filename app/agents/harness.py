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
from app.models.entities import ChatMessage, ChatSession, PsychologicalReport, ToolJob, UserAccount
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
        # Harness 根据 Agent 结果创建心理报告，
        # 报告只在 intent != CHAT 且有风险评估时创建
        try:
            report = self._create_report(user, session, original_input, agent_run)
        except Exception as exc:
            # Queue submission is part of the report transaction. Preserve the
            # actual run's diagnostics after rollback rather than claiming success.
            agent_run.status = "FAILED"
            agent_run.error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            agent_run.completed_at = now_cn()
            agent_run.collaboration_events.append(AgentEvent(
                type=AgentEventType.TRACE_FAILED,
                actor="MindBridgeAgentHarness",
                message="report and tool submission rolled back",
                metadata={"errorType": type(exc).__name__, "stage": "report_enqueue"},
            ))
            AgentTraceService(self.db).save_run(
                user=user, session=session, original_input=original_input,
                sanitized_input=model_input, memory_brief=agent_run.memory_brief,
                agent_run=agent_run, report_id=None,
            )
            raise
        risk_level = report.risk_level if report is not None else None
        # 创建工具计划，这就是 Agent Runtime 和 MCP/工具系统之间真正的连接点：
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
            status = record.get("status", "")
            event_type = (AgentEventType.TOOL_EXECUTION_QUEUED if status == "QUEUED" else
                          AgentEventType.TOOL_EXECUTION_FAILED if status == "FAILED" else
                          AgentEventType.TOOL_EXECUTION_SIMULATED if status == "SIMULATED" else
                          AgentEventType.TOOL_EXECUTION_COMPLETED)
            agent_run.collaboration_events.append(AgentEvent(
                type=event_type, actor="ToolDispatcher", message=str(record.get("kind", "")),
                metadata={"kind": record.get("kind", ""), "status": status,
                          "jobId": record.get("jobId"), "retryCount": record.get("retryCount", 0),
                          "idempotencyKey": record.get("idempotencyKey", ""), "errorType": record.get("errorType")},
                duration_ms=record.get("durationMs")))
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

    async def dispatch_tools(self, tool_plan: AgentToolPlan) -> list[dict]:
        """Describe committed queue work, or execute explicit synchronous MCP mode."""
        if tool_plan.report_id is None:
            return []
        if self.settings.tool_queue_enabled:
            started = time.perf_counter()
            # Already committed atomically with the report; this only describes submission.
            jobs = self.db.query(ToolJob).filter_by(report_id=tool_plan.report_id).all()
            duration_ms = (time.perf_counter() - started) * 1000.0
            return [
                {
                    "kind": job.kind,
                    "status": "QUEUED",
                    "durationMs": duration_ms,
                    "retryCount": job.attempts,
                    "idempotencyKey": job.idempotency_key,
                    "jobId": job.id,
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
                "status": "SIMULATED" if result.get("status") == "SIMULATED" else "COMPLETED",
                "durationMs": duration_ms,
                "retryCount": 0,
                "idempotencyKey": result.get("idempotencyKey", ""),
                "errorType": None,
                "detail": str(result)[:200],
            }
            for result in results
        ]

    def save_message(self, user: UserAccount, session: ChatSession, role: MessageRole, content: str) -> None:
        message = ChatMessage(user_id=user.id, session_id=session.id, role=role.value, content=content)
        self.db.add(message)
        self.db.flush()
        if role == MessageRole.USER and self.settings.student_memory_enabled:
            from app.services.student_memory import StudentMemoryService, extract_facts
            service = StudentMemoryService(self.db, self.settings)
            for kind, fact in extract_facts(content):
                service.store(user.id, session.id, message.id, kind, fact)
        session.touch()
        self.db.add(session)
        self.db.commit()
        self.memory.append(session.public_id, role.value, content)

    def _resolve_session(self, user: UserAccount, public_id: str | None, text: str) -> ChatSession:
        if public_id:
            session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id, ChatSession.user_id == user.id, ChatSession.deleted_at.is_(None)).first()
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
        try:
            self.db.add(report)
            self.db.flush()
            if self.settings.tool_queue_enabled:
                ToolQueueService(self.db, self.settings).enqueue_report(
                    report.id, report.risk_level, commit=False, trace_id=agent_run.trace_id)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self.db.refresh(report)
        return report
