"""Trace dict -> TraceRecord 解析，含版本门槛与 DB 行还原辅助。"""

from __future__ import annotations

import json
from typing import Any

from app.agent_eval.schemas import TraceRecord


class TraceVersionError(Exception):
    """旧版本 Trace 无法解析，请先迁移（scripts/migrate_trace_v2.py）或重新生成。"""


def parse_trace(raw: dict[str, Any]) -> TraceRecord:
    version = raw.get("traceVersion")
    if not isinstance(version, str) or not version.startswith("2."):
        raise TraceVersionError(
            f"旧版本 Trace 无法解析（traceVersion={version!r}），请先迁移（scripts/migrate_trace_v2.py）或重新生成"
        )
    return TraceRecord.model_validate(raw)


def parse_db_row(row: Any, events: list[Any] | None = None) -> dict[str, Any]:
    """把 AgentRunTrace ORM 行（+ AgentTraceEvent 行）还原成 v2 trace dict。

    events 为 None 时假定传入的是已加载 ``trace_events`` 关系的 ORM 对象不可用，
    调用方需自行查询 AgentTraceEvent 并传入。artifact/task 从 agent_steps_json
    里的 kind=agent_artifact/agent_task 条目还原（save_run 已持久化这些内容）。
    """
    step_entries = json.loads(row.agent_steps_json or "[]")
    artifacts = [
        {
            "id": entry.get("id", ""),
            "owner": entry.get("owner", ""),
            "kind": entry.get("artifactKind", ""),
            "version": entry.get("version", 1),
            "confidence": entry.get("confidence", 1.0),
            "taskId": entry.get("taskId", ""),
            "metadata": entry.get("metadata") or {},
            "payload": entry.get("payload") or {},
        }
        for entry in step_entries
        if isinstance(entry, dict) and entry.get("kind") == "agent_artifact"
    ]
    tasks = [
        {
            "id": entry.get("id", ""),
            "title": entry.get("title", ""),
            "status": entry.get("status", "OPEN"),
            "priority": entry.get("priority", "NORMAL"),
            "requiredCapabilities": entry.get("requiredCapabilities") or [],
            "claimedBy": entry.get("claimedBy") or [],
            "createdBy": entry.get("createdBy", ""),
            "dependsOn": entry.get("dependsOn") or [],
            "metadata": entry.get("metadata") or {},
        }
        for entry in step_entries
        if isinstance(entry, dict) and entry.get("kind") == "agent_task"
    ]
    metrics = json.loads(row.metrics_json or "{}")
    event_dicts = [
        {
            "eventId": event.event_id,
            "traceId": event.trace_id,
            "eventType": event.event_type,
            "actor": event.actor,
            "taskId": event.task_id,
            "round": event.round,
            "timestamp": event.timestamp,
            "durationMs": event.duration_ms,
            "inputArtifactIds": json.loads(event.input_artifact_ids_json or "[]"),
            "outputArtifactIds": json.loads(event.output_artifact_ids_json or "[]"),
            "metadata": json.loads(event.metadata_json or "{}"),
        }
        for event in (events or [])
    ]
    return {
        "traceId": row.trace_id,
        "traceVersion": row.trace_version,
        "status": row.status,
        "intent": row.intent,
        "riskLevel": row.risk_level,
        "finalResponse": row.final_response,
        "finalResponseArtifactId": row.final_response_artifact_id,
        "finalReviewArtifactId": row.final_review_artifact_id,
        "startedAt": row.started_at,
        "completedAt": row.completed_at,
        "durationMs": metrics.get("totalDurationMs"),
        "error": json.loads(row.error_json or "{}"),
        "originalInput": row.original_input,
        "sanitizedInput": row.sanitized_input,
        "memoryBrief": row.memory_brief,
        "reportId": row.report_id,
        "userMessageId": row.user_message_id,
        "events": event_dicts,
        "artifacts": artifacts,
        "tasks": tasks,
        "metrics": metrics,
    }
