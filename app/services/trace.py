
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.agents.events import AgentEventType
from app.agents.result import AgentRunResult
from app.models.entities import AgentRunTrace, AgentTraceEvent, ChatSession, UserAccount

TRACE_VERSION = "2.0"


class AgentTraceService:
    def __init__(self, db: Session):
        self.db = db

    def save_run(
        self,
        user: UserAccount,
        session: ChatSession,
        original_input: str,
        sanitized_input: str,
        memory_brief: str,
        agent_run: AgentRunResult,
        report_id: int | None,
        user_message_id: int | None = None,
    ) -> AgentRunTrace:
        record = build_trace_record(
            agent_run,
            original_input=original_input,
            sanitized_input=sanitized_input,
            memory_brief=memory_brief,
            report_id=report_id,
            user_message_id=user_message_id,
        )
        trace = AgentRunTrace(
            user_id=user.id,
            session_id=session.id,
            report_id=report_id,
            intent=agent_run.intent.value,
            risk_level=agent_run.risk_level.value,
            original_input=original_input,
            sanitized_input=sanitized_input,
            memory_brief=memory_brief,
            agent_steps_json=_json(_agent_steps_with_collaboration(agent_run)),
            retrieved_knowledge_json=_json(agent_run.retrieved_knowledge),
            response_messages_json=_json(agent_run.response_messages),
            assessment_json=_json(agent_run.assessment or {}),
            trace_id=record["traceId"],
            trace_version=record["traceVersion"],
            user_message_id=user_message_id,
            status=record["status"],
            started_at=record["startedAt"],
            completed_at=record["completedAt"],
            final_response=record["finalResponse"],
            final_response_artifact_id=record["finalResponseArtifactId"],
            final_review_artifact_id=record["finalReviewArtifactId"],
            error_json=_json(record["error"]),
            metrics_json=_json(record["metrics"]),
        )
        self.db.add(trace)
        self.db.flush()
        for event in record["events"]:
            self.db.add(
                AgentTraceEvent(
                    trace_id=record["traceId"],
                    event_id=event["eventId"],
                    event_type=event["eventType"],
                    actor=event["actor"],
                    task_id=event["taskId"],
                    round=event["round"],
                    timestamp=event["timestamp"],
                    duration_ms=event["durationMs"],
                    input_artifact_ids_json=_json(event["inputArtifactIds"]),
                    output_artifact_ids_json=_json(event["outputArtifactIds"]),
                    metadata_json=_json(event["metadata"]),
                )
            )
        self.db.commit()
        self.db.refresh(trace)
        return trace


def build_trace_record(
    agent_run: AgentRunResult,
    original_input: str = "",
    sanitized_input: str = "",
    memory_brief: str = "",
    report_id: int | None = None,
    user_message_id: int | None = None,
) -> dict[str, Any]:
    """生成与 DB 行等价的 v2 trace dict，供离线评测 runner 在无 DB 情况下复用。"""
    trace_id = agent_run.trace_id or ""
    events = [_event_to_record(event, trace_id) for event in agent_run.collaboration_events]
    status = agent_run.status if agent_run.status in {"RUNNING", "COMPLETED", "FAILED"} else "COMPLETED"
    if status == "RUNNING":
        status = "COMPLETED"
    metrics = _trace_metrics(agent_run, events)
    return {
        "traceId": trace_id,
        "traceVersion": TRACE_VERSION,
        "status": status,
        "intent": agent_run.intent.value,
        "riskLevel": agent_run.risk_level.value,
        "finalResponse": agent_run.final_response,
        "finalResponseArtifactId": agent_run.final_response_artifact_id,
        "finalReviewArtifactId": agent_run.final_review_artifact_id,
        "startedAt": agent_run.started_at,
        "completedAt": agent_run.completed_at,
        "durationMs": metrics.get("totalDurationMs"),
        "error": agent_run.error or {},
        "originalInput": original_input,
        "sanitizedInput": sanitized_input,
        "memoryBrief": memory_brief,
        "reportId": report_id,
        "userMessageId": user_message_id,
        "events": events,
        "artifacts": [_artifact_to_record(artifact) for artifact in agent_run.collaboration_artifacts],
        "tasks": [_task_to_record(task) for task in agent_run.collaboration_tasks],
        "metrics": metrics,
    }


def _artifact_to_record(artifact: Any) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "owner": artifact.owner,
        "kind": artifact.kind,
        "version": getattr(artifact, "version", 1),
        "confidence": getattr(artifact, "confidence", 1.0),
        "taskId": getattr(artifact, "task_id", ""),
        "metadata": _to_jsonable(getattr(artifact, "metadata", {})),
        "payload": _to_jsonable(getattr(artifact, "payload", {})),
    }


def _task_to_record(task: Any) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": getattr(task, "title", ""),
        "status": getattr(task.status, "value", task.status),
        "priority": getattr(task.priority, "value", task.priority),
        "requiredCapabilities": sorted(getattr(task, "required_capabilities", ()) or ()),
        "claimedBy": list(getattr(task, "claimed_by", ()) or ()),
        "createdBy": getattr(task, "created_by", ""),
        "dependsOn": list(getattr(task, "depends_on", ()) or ()),
        "metadata": _to_jsonable(getattr(task, "metadata", {})),
    }


def _event_to_record(event: Any, trace_id: str) -> dict[str, Any]:
    return {
        "eventId": getattr(event, "event_id", ""),
        "traceId": trace_id,
        "eventType": getattr(event.type, "value", event.type),
        "actor": event.actor,
        "taskId": getattr(event, "task_id", ""),
        "round": getattr(event, "round", 0),
        "timestamp": getattr(event, "timestamp", None),
        "durationMs": getattr(event, "duration_ms", None),
        "inputArtifactIds": list(getattr(event, "input_artifact_ids", ())),
        "outputArtifactIds": list(getattr(event, "output_artifact_ids", ())),
        "metadata": getattr(event, "metadata", {}),
    }


def _trace_metrics(agent_run: AgentRunResult, events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event["eventType"]] = counts.get(event["eventType"], 0) + 1
    total_duration_ms = None
    if agent_run.started_at is not None and agent_run.completed_at is not None:
        total_duration_ms = (agent_run.completed_at - agent_run.started_at).total_seconds() * 1000.0
    return {
        "rounds": counts.get(AgentEventType.ROUND_STARTED.value, 0),
        "revisions": counts.get(AgentEventType.REVISION_REQUESTED.value, 0),
        "agentExecutions": counts.get(AgentEventType.AGENT_EXECUTION_COMPLETED.value, 0)
        + counts.get(AgentEventType.AGENT_EXECUTION_FAILED.value, 0),
        "llmCalls": counts.get(AgentEventType.LLM_CALL_COMPLETED.value, 0)
        + counts.get(AgentEventType.LLM_CALL_FAILED.value, 0),
        "totalDurationMs": total_duration_ms,
        "toolExecutions": counts.get(AgentEventType.TOOL_EXECUTION_COMPLETED.value, 0)
        + counts.get(AgentEventType.TOOL_EXECUTION_FAILED.value, 0),
    }


def _json(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False, default=str)


def _agent_steps_with_collaboration(agent_run: AgentRunResult) -> list[Any]:
    entries: list[Any] = [*agent_run.steps]
    entries.extend(
        {
            "kind": "agent_event",
            "type": getattr(event.type, "value", event.type),
            "actor": event.actor,
            "taskId": event.task_id,
            "artifactId": event.artifact_id,
            "message": event.message,
            "metadata": event.metadata,
        }
        for event in agent_run.collaboration_events
    )
    entries.extend(
        {
            "kind": "agent_task",
            "id": task.id,
            "title": task.title,
            "status": getattr(task.status, "value", task.status),
            "priority": getattr(task.priority, "value", task.priority),
            "requiredCapabilities": sorted(task.required_capabilities),
            "claimedBy": list(task.claimed_by),
            "createdBy": task.created_by,
            "dependsOn": list(task.depends_on),
            "metadata": task.metadata,
        }
        for task in agent_run.collaboration_tasks
    )
    entries.extend(
        {
            "kind": "agent_artifact",
            "id": artifact.id,
            "owner": artifact.owner,
            "artifactKind": artifact.kind,
            "confidence": artifact.confidence,
            "version": artifact.version,
            "taskId": artifact.task_id,
            "metadata": artifact.metadata,
            "payload": artifact.payload,
        }
        for artifact in agent_run.collaboration_artifacts
    )
    return entries


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
