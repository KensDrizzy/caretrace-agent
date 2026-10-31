"""The shared MCP execution boundary: policy, fencing, receipts and audit."""
import hashlib
import json

from app.core.enums import ToolJobKind, ToolStatus
from app.models.entities import PsychologicalReport, RiskCase, ToolJob, ToolExecutionRecord, now
from app.services.tool_errors import ToolExecutionError
from app.services.tool_governance import ToolGovernanceService
from app.services.tool_locks import operation_lock
from app.services.tools import ToolOrchestrationService


def operation_key(kind: str, report_id: int, settings) -> str:
    key = f"{kind}:{report_id}:v1"
    if kind in {ToolJobKind.ALERT_SEND.value, ToolJobKind.RISK_ALERT.value}:
        recipients = ",".join(sorted(set(x.strip() for x in settings.alert_email_to.replace(";", ",").split(",") if x.strip()))) or "unconfigured"
        key += ":" + hashlib.sha256(f"{settings.alert_email_delivery_mode.strip().lower()}:{recipients}".encode()).hexdigest()[:32]
    return key


def execute_tool(db, settings, kind: str, report_id: int, *, job_id=None, lease_token=None, idempotency_key=None) -> dict:
    key = operation_key(kind, report_id, settings)
    fingerprint = hashlib.sha256(json.dumps([kind, report_id, key]).encode()).hexdigest()
    job = db.get(ToolJob, job_id) if job_id is not None else None
    if job_id is not None:
        if (job is None or job.kind != kind or job.report_id != report_id or job.status != "RUNNING"
                or not lease_token or job.lease_token != lease_token or job.lease_until is None
                or job.lease_until.replace(tzinfo=None) <= now().replace(tzinfo=None)):
            raise ToolExecutionError("任务租约无效或已过期", code="STALE_LEASE", retryable=False)
    report = db.get(PsychologicalReport, report_id)
    governance = ToolGovernanceService(db)
    audit_job = job or ToolJob(report_id=report_id, kind=kind, attempts=1)
    audit = governance.start_job(audit_job, report)
    try:
        if not audit.allowed:
            raise ToolExecutionError(audit.reason, code="POLICY_DENIED", retryable=False)
        if idempotency_key is not None and idempotency_key != key:
            raise ToolExecutionError("幂等键与当前操作或投递配置不匹配", code="IDEMPOTENCY_CONFLICT", retryable=False)
        resource_kind = "alert" if kind in {ToolJobKind.ALERT_SEND.value, ToolJobKind.RISK_ALERT.value} else kind
        with operation_lock(db.get_bind(), f"report:{report_id}:{resource_kind}", settings.tool_lock_timeout_seconds):
            # Close the old read snapshot before inspecting another process's receipt.
            db.commit()
            if job is not None:
                db.refresh(job)
                if (job.status != "RUNNING" or job.lease_token != lease_token or job.lease_until is None
                        or job.lease_until.replace(tzinfo=None) <= now().replace(tzinfo=None)):
                    raise ToolExecutionError("等待执行锁期间租约失效", code="STALE_LEASE", retryable=False)
            db.refresh(report)
            governance.require_allowed(audit_job, report)
            receipt = db.get(ToolExecutionRecord, key)
            if receipt is not None:
                if receipt.request_hash != fingerprint:
                    raise ToolExecutionError("幂等参数冲突", code="IDEMPOTENCY_CONFLICT", retryable=False)
                result = json.loads(receipt.result_json)
                governance.finish(audit, "REUSED", payload={"idempotencyKey": key, "result": result})
                return result
            tools = ToolOrchestrationService(db, settings)
            if kind == ToolJobKind.EXCEL_REPORT.value:
                record = tools.write_excel(report)
                result = {"status": record.status, "reportId": report.id, "filePath": record.file_path}
            elif kind == ToolJobKind.CASE_CREATE.value:
                case = tools.create_case(report)
                result = {"status": "SUCCESS", "reportId": report.id, "caseId": case.id}
            elif kind in {ToolJobKind.ALERT_SEND.value, ToolJobKind.RISK_ALERT.value}:
                if kind == ToolJobKind.ALERT_SEND.value:
                    case = db.query(RiskCase).filter_by(report_id=report.id).first()
                    if case is None:
                        raise ToolExecutionError("风险个案尚未创建", code="CASE_NOT_READY")
                    record = tools.send_case_alert(case)
                else:
                    record = tools.notify(report)
                if record.status not in {ToolStatus.SUCCESS.value, ToolStatus.SIMULATED.value}:
                    raise ToolExecutionError(record.message, code="DELIVERY_UNKNOWN" if record.status == "UNKNOWN" else "DELIVERY_FAILED", retryable=record.retryable)
                result = {"status": record.status, "reportId": report.id, "deliveryId": record.id,
                          "channel": record.channel, "message": record.message}
            else:
                raise ToolExecutionError("未知工具", code="UNKNOWN_TOOL", retryable=False)
            result["idempotencyKey"] = key
            db.add(ToolExecutionRecord(idempotency_key=key, request_hash=fingerprint, result_json=json.dumps(result, ensure_ascii=False)))
            db.commit()
            governance.finish(audit, result["status"], payload={"idempotencyKey": key, "result": result})
            return result
    except Exception as exc:
        db.rollback()
        governance.finish(audit, "BLOCKED" if not audit.allowed else "FAILED", str(exc),
                          {"idempotencyKey": key, "errorCode": getattr(exc, "code", type(exc).__name__)})
        raise
