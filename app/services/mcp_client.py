from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from app.core.config import Settings
from app.core.enums import RiskLevel, ToolJobKind
from app.services.tool_errors import ToolExecutionError


class McpToolError(ToolExecutionError):
    pass


class MindBridgeMcpToolClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def execute_job(self, job, case_id: int | None = None) -> dict:
        names = {ToolJobKind.EXCEL_REPORT.value: "mindbridge_excel_report", ToolJobKind.CASE_CREATE.value: "mindbridge_case_create",
                 ToolJobKind.ALERT_SEND.value: "mindbridge_alert_send", ToolJobKind.RISK_ALERT.value: "mindbridge_alert_notify"}
        if job.kind not in names:
            raise McpToolError("未知任务类型", code="UNKNOWN_TOOL", retryable=False)
        arguments = {"job_id": job.id, "lease_token": job.lease_token, "idempotency_key": job.idempotency_key}
        arguments.update({"case_id": case_id} if job.kind == ToolJobKind.ALERT_SEND.value else {"report_id": job.report_id})
        try:
            async with asyncio.timeout(self.settings.tool_mcp_timeout_seconds):
                async with self._session() as session:
                    return await self._call_tool(session, names[job.kind], arguments)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise McpToolError(f"MCP transport: {type(exc).__name__}", code="TRANSPORT_ERROR") from exc

    async def handle_report(self, report_id: int, risk_level: str | None) -> list[dict]:
        # Explicit synchronous compatibility mode, not the durable queue path.
        async with asyncio.timeout(self.settings.tool_mcp_timeout_seconds):
            async with self._session() as session:
                results = []
                if risk_level in {RiskLevel.MEDIUM.value, RiskLevel.HIGH.value}:
                    case = await self._call_tool(session, "mindbridge_case_create", {"report_id": report_id})
                    results.append(case)
                    if risk_level == RiskLevel.HIGH.value:
                        results.append(await self._call_tool(session, "mindbridge_alert_send", {"case_id": case["caseId"]}))
                results.append(await self._call_tool(session, "mindbridge_excel_report", {"report_id": report_id}))
                return results

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = os.environ.copy()
        # Pass the resolved settings, not just the parent's environment. This is
        # essential for isolated tests and programmatically selected DB/config.
        for key, value in self.settings.model_dump(mode="json").items():
            if value is not None:
                env[key.upper()] = (json.dumps(value) if isinstance(value, (dict, list))
                                    else str(value).lower() if isinstance(value, bool) else str(value))
        root = self.settings.project_root
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        server = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_tools.server"], env=env, cwd=str(root))
        tool_error = None
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                try:
                    yield session
                except ToolExecutionError as exc:
                    # Preserve retryable/code instead of letting AnyIO wrap a
                    # business error into a transport ExceptionGroup.
                    tool_error = exc
        if tool_error is not None:
            raise tool_error

    async def _call_tool(self, session: Any, name: str, arguments: dict[str, Any]) -> dict:
        result = await session.call_tool(name, arguments=arguments)
        data = getattr(result, "structuredContent", None)
        if not isinstance(data, dict):
            try:
                data = json.loads("\n".join(item.text for item in result.content if getattr(item, "type", None) == "text"))
            except (ValueError, TypeError):
                data = {}
        if getattr(result, "isError", False) or data.get("status") == "FAILED":
            error = data.get("error") or {}
            raise McpToolError(error.get("message", f"{name} failed"), code=error.get("code", "TOOL_ERROR"),
                               retryable=error.get("retryable", True), retry_after=error.get("retryAfter"))
        if data.get("status") not in {"SUCCESS", "SIMULATED"}:
            raise McpToolError("工具返回了未知结果结构", code="INVALID_RESULT", retryable=False)
        return data
