"""Trusted local stdio MCP service. No HTTP transport or client-supplied identity."""
import json

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.enums import ToolJobKind
from app.models.entities import RiskCase, ToolAuditRecord
from app.services.tool_errors import ToolExecutionError
from app.services.tool_execution import execute_tool
from app.services.tools import ToolOrchestrationService

mcp = FastMCP("mindbridge-python-tools")


def _result(callback) -> CallToolResult:
    try:
        payload = callback()
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))], structuredContent=payload)
    except ToolExecutionError as exc:
        error = {"code": exc.code, "message": str(exc), "retryable": exc.retryable, "retryAfter": exc.retry_after}
    except Exception as exc:
        # Never include credentials or raw SQL/SMTP arguments in protocol errors.
        error = {"code": "INTERNAL_ERROR", "message": type(exc).__name__, "retryable": True}
    payload = {"status": "FAILED", "error": error}
    return CallToolResult(isError=True, content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))], structuredContent=payload)


def _execute(kind, report_id, job_id, lease_token, idempotency_key):
    with SessionLocal() as db:
        return execute_tool(db, get_settings(), kind, report_id, job_id=job_id, lease_token=lease_token, idempotency_key=idempotency_key)


@mcp.tool()
def mindbridge_excel_report(report_id: int, job_id: int | None = None, lease_token: str | None = None, idempotency_key: str | None = None) -> CallToolResult:
    """Export a persisted report to the Excel ledger; returns a structured execution result."""
    return _result(lambda: _execute(ToolJobKind.EXCEL_REPORT.value, report_id, job_id, lease_token, idempotency_key))


@mcp.tool()
def mindbridge_case_create(report_id: int, job_id: int | None = None, lease_token: str | None = None, idempotency_key: str | None = None) -> CallToolResult:
    """Create or reuse a medium/high-risk report's counselor case."""
    return _result(lambda: _execute(ToolJobKind.CASE_CREATE.value, report_id, job_id, lease_token, idempotency_key))


@mcp.tool()
def mindbridge_alert_send(case_id: int, job_id: int | None = None, lease_token: str | None = None, idempotency_key: str | None = None) -> CallToolResult:
    """Submit a high-risk case alert; SIMULATED means no email was sent."""
    def call():
        with SessionLocal() as db:
            case = db.get(RiskCase, case_id)
            if case is None:
                raise ToolExecutionError("个案不存在", code="NOT_FOUND", retryable=False)
            return execute_tool(db, get_settings(), ToolJobKind.ALERT_SEND.value, case.report_id,
                                job_id=job_id, lease_token=lease_token, idempotency_key=idempotency_key)
    return _result(call)


@mcp.tool()
def mindbridge_alert_notify(report_id: int, job_id: int | None = None, lease_token: str | None = None, idempotency_key: str | None = None) -> CallToolResult:
    """Submit a high-risk report alert without a case (legacy compatibility)."""
    return _result(lambda: _execute(ToolJobKind.RISK_ALERT.value, report_id, job_id, lease_token, idempotency_key))


def _manage_case(case_id, note, acknowledge):
    settings = get_settings()
    with SessionLocal() as db:
        allowed = settings.mcp_allow_case_management
        audit = ToolAuditRecord(tool_name="case_ack" if acknowledge else "case_note", policy="local_case_management",
                                allowed=allowed, status="AUTHORIZED" if allowed else "BLOCKED", reason="trusted local service identity")
        db.add(audit)
        db.commit()
        if not allowed:
            raise ToolExecutionError("MCP 个案管理未启用；请使用已认证的管理员接口", code="POLICY_DENIED", retryable=False)
        try:
            tools = ToolOrchestrationService(db, settings)
            if acknowledge:
                case = tools.acknowledge_case(case_id, settings.mcp_service_actor, note)
                result = {"status": "SUCCESS", "caseId": case.id, "actor": settings.mcp_service_actor}
            else:
                record = tools.add_case_note(case_id, settings.mcp_service_actor, note)
                result = {"status": "SUCCESS", "noteId": record.id, "actor": settings.mcp_service_actor}
            audit.status = "SUCCESS"
            return result
        except Exception:
            db.rollback()
            audit.status = "FAILED"
            raise
        finally:
            db.add(audit)
            db.commit()


@mcp.tool()
def mindbridge_alert_ack(case_id: int, note: str = "") -> CallToolResult:
    """Acknowledge a case as the configured service identity; disabled by default."""
    return _result(lambda: _manage_case(case_id, note, True))


@mcp.tool()
def mindbridge_case_note_add(case_id: int, note: str) -> CallToolResult:
    """Add a case note as the configured service identity; disabled by default."""
    return _result(lambda: _manage_case(case_id, note, False))


if __name__ == "__main__":
    # Initialize/migrate the database before launching; tools never run schema DDL.
    mcp.run()
