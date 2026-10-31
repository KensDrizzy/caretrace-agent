from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.entities import AlertRecord, AgentRunTrace, AgentTraceEvent, CaseNote, ChatMessage, ChatSession, DeadLetterRecord, ExcelRecord, PsychologicalReport, RiskCase, ToolAuditRecord, ToolJob, UserAccount, now
from app.schemas.dtos import AgentRunTraceResponse, CaseNoteResponse, ChatSessionResponse, ConversationMessageResponse, ConversationResponse, DeadLetterResponse, ReportResponse, RiskCaseResponse, ToolAuditResponse, ToolJobResponse, ToolRecordResponse


class ReportService:
    def __init__(self, db: Session):
        self.db = db

    def latest_reports(self, user_id: int | None = None) -> list[ReportResponse]:
        query = self.db.query(PsychologicalReport).order_by(PsychologicalReport.created_at.desc())
        if user_id is not None:
            query = query.filter(PsychologicalReport.user_id == user_id)
        return [self._report_response(item) for item in query.limit(100).all()]

    def excel_records(self) -> list[ToolRecordResponse]:
        rows = self.db.query(ExcelRecord).order_by(ExcelRecord.created_at.desc()).limit(100).all()
        return [
            ToolRecordResponse(id=row.id, reportId=row.report_id, status=row.status, message=row.message, createdAt=row.created_at, filePath=row.file_path)
            for row in rows
        ]

    def alert_records(self) -> list[ToolRecordResponse]:
        rows = self.db.query(AlertRecord).order_by(AlertRecord.created_at.desc()).limit(100).all()
        return [
            ToolRecordResponse(
                id=row.id,
                reportId=row.report_id,
                status=row.status,
                message=row.message,
                createdAt=row.created_at,
                channel=row.channel,
                recipient=row.recipient,
            )
            for row in rows
        ]

    def risk_cases(self) -> list[RiskCaseResponse]:
        rows = self.db.query(RiskCase).order_by(RiskCase.updated_at.desc()).limit(100).all()
        return [
            RiskCaseResponse(
                id=row.id,
                reportId=row.report_id,
                riskLevel=row.risk_level,
                status=row.status,
                owner=row.owner,
                summary=row.summary,
                handoffSummary=row.handoff_summary,
                acknowledgedBy=row.acknowledged_by,
                acknowledgedAt=row.acknowledged_at,
                createdAt=row.created_at,
                updatedAt=row.updated_at,
            )
            for row in rows
        ]

    def case_notes(self, case_id: int) -> list[CaseNoteResponse]:
        rows = self.db.query(CaseNote).filter(CaseNote.case_id == case_id).order_by(CaseNote.created_at.asc()).all()
        return [
            CaseNoteResponse(id=row.id, caseId=row.case_id, actor=row.actor, note=row.note, createdAt=row.created_at)
            for row in rows
        ]

    def tool_jobs(self, job_id: int | None = None) -> list[ToolJobResponse]:
        query = self.db.query(ToolJob)
        if job_id is not None:
            query = query.filter(ToolJob.id == job_id)
        rows = query.order_by(ToolJob.created_at.desc()).limit(100).all()
        return [
            ToolJobResponse(
                id=row.id,
                reportId=row.report_id,
                kind=row.kind,
                status=row.status,
                idempotencyKey=row.idempotency_key,
                traceId=row.trace_id or "",
                workerId=row.worker_id,
                leaseUntil=row.lease_until,
                result=_loads(row.result_json, {}),
                attempts=row.attempts,
                maxAttempts=row.max_attempts,
                dependsOnJobId=row.depends_on_job_id,
                runAfter=row.run_after,
                lastError=row.last_error,
                createdAt=row.created_at,
                updatedAt=row.updated_at,
            )
            for row in rows
        ]

    def tool_job_detail(self, job_id: int) -> dict:
        jobs = self.tool_jobs(job_id)
        if not jobs:
            raise ValueError("任务不存在")
        events = self.db.query(AgentTraceEvent).filter(
            AgentTraceEvent.actor == "ToolQueueWorker", AgentTraceEvent.task_id == str(job_id)
        ).order_by(AgentTraceEvent.id).all()
        audits = self.db.query(ToolAuditRecord).filter_by(job_id=job_id).order_by(ToolAuditRecord.id).all()
        return {
            "job": jobs[0],
            "events": [{"eventType": row.event_type, "timestamp": row.timestamp,
                        "metadata": _loads(row.metadata_json, {})} for row in events],
            "audits": [{"status": row.status, "reason": row.reason,
                        "payload": _loads(row.payload, {}), "createdAt": row.created_at} for row in audits],
        }

    def dead_letters(self) -> list[DeadLetterResponse]:
        rows = self.db.query(DeadLetterRecord).order_by(DeadLetterRecord.created_at.desc()).limit(100).all()
        return [
            DeadLetterResponse(
                id=row.id,
                jobId=row.job_id,
                reportId=row.report_id,
                kind=row.kind,
                reason=row.reason,
                payload=row.payload,
                createdAt=row.created_at,
            )
            for row in rows
        ]


    def agent_run_traces(self) -> list[AgentRunTraceResponse]:
        rows = self.db.query(AgentRunTrace).order_by(AgentRunTrace.created_at.desc()).limit(100).all()
        responses = []
        for row in rows:
            user = self.db.get(UserAccount, row.user_id)
            session = self.db.get(ChatSession, row.session_id)
            responses.append(
                AgentRunTraceResponse(
                    id=row.id,
                    sessionId=session.public_id if session else "",
                    reportId=row.report_id,
                    username=user.username if user else "",
                    intent=row.intent,
                    riskLevel=row.risk_level,
                    originalInput=row.original_input,
                    sanitizedInput=row.sanitized_input,
                    memoryBrief=row.memory_brief,
                    agentSteps=_loads(row.agent_steps_json, []),
                    retrievedKnowledge=_loads(row.retrieved_knowledge_json, []),
                    responseMessages=_loads(row.response_messages_json, []),
                    assessment=_loads(row.assessment_json, {}),
                    createdAt=row.created_at,
                    traceId=getattr(row, "trace_id", None) or None,
                    traceVersion=getattr(row, "trace_version", None) or None,
                    status=getattr(row, "status", None) or None,
                    finalResponse=getattr(row, "final_response", None) or None,
                    startedAt=getattr(row, "started_at", None),
                    completedAt=getattr(row, "completed_at", None),
                )
            )
        return responses

    def tool_audits(self) -> list[ToolAuditResponse]:
        rows = self.db.query(ToolAuditRecord).order_by(ToolAuditRecord.created_at.desc()).limit(100).all()
        return [
            ToolAuditResponse(
                id=row.id,
                jobId=row.job_id,
                reportId=row.report_id,
                toolName=row.tool_name,
                policy=row.policy,
                allowed=row.allowed,
                status=row.status,
                reason=row.reason,
                payload=_loads(row.payload, {}),
                createdAt=row.created_at,
                updatedAt=row.updated_at,
            )
            for row in rows
        ]

    def conversation(self, public_id: str) -> ConversationResponse:
        session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id).first()
        if session is None:
            raise ValueError("Session not found")
        rows = self.db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.asc()).all()
        return ConversationResponse(
            sessionId=session.public_id,
            title=session.title,
            messages=[ConversationMessageResponse(role=row.role, content=row.content, createdAt=row.created_at) for row in rows],
        )

    def user_sessions(self, user_id: int) -> list[ChatSessionResponse]:
        rows = self.db.query(ChatSession).filter(ChatSession.user_id == user_id, ChatSession.deleted_at.is_(None)).order_by(ChatSession.updated_at.desc()).limit(50).all()
        return [ChatSessionResponse(sessionId=row.public_id, title=row.title, updatedAt=row.updated_at) for row in rows]

    def rename_user_session(self, user_id: int, public_id: str, title: str) -> ChatSessionResponse:
        session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id, ChatSession.user_id == user_id, ChatSession.deleted_at.is_(None)).first()
        if session is None:
            raise ValueError("Session not found")
        session.title = title
        self.db.commit()
        self.db.refresh(session)
        return ChatSessionResponse(sessionId=session.public_id, title=session.title, updatedAt=session.updated_at)

    def delete_user_session(self, user_id: int, public_id: str) -> None:
        session = self.db.query(ChatSession).filter(
            ChatSession.public_id == public_id, ChatSession.user_id == user_id
        ).first()
        if session is None:
            raise ValueError("Session not found")
        # Idempotent soft deletion keeps reports, cases and trace associations intact.
        if session.deleted_at is None:
            session.deleted_at = now()
            self.db.commit()

    def user_conversation(self, user_id: int, public_id: str) -> ConversationResponse:
        session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id, ChatSession.user_id == user_id, ChatSession.deleted_at.is_(None)).first()
        if session is None:
            raise ValueError("Session not found")
        rows = self.db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.asc()).all()
        return ConversationResponse(
            sessionId=session.public_id,
            title=session.title,
            messages=[ConversationMessageResponse(role=row.role, content=row.content, createdAt=row.created_at) for row in rows],
        )

    def _report_response(self, report: PsychologicalReport) -> ReportResponse:
        user = self.db.get(UserAccount, report.user_id)
        session = self.db.get(ChatSession, report.session_id)
        return ReportResponse(
            id=report.id,
            sessionId=session.public_id if session else "",
            username=user.username if user else "",
            displayName=user.display_name if user else "",
            content=report.content,
            intent=report.intent,
            emotion=report.emotion,
            emotionScore=report.emotion_score,
            riskLevel=report.risk_level,
            confidence=report.confidence,
            summary=report.summary,
            createdAt=report.created_at,
        )


def _loads(raw: str, default):
    import json

    try:
        return json.loads(raw or "")
    except Exception:
        return default
