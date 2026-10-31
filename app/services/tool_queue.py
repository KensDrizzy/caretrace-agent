from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from sqlalchemy import or_, update
from sqlalchemy.exc import IntegrityError

from app.core.database import SessionLocal
from app.core.enums import RiskLevel, ToolJobKind
from app.models.entities import DeadLetterRecord, PsychologicalReport, RiskCase, ToolJob, ToolAuditRecord, AgentTraceEvent, now
from app.services.mcp_client import MindBridgeMcpToolClient
from app.services.tool_errors import ToolExecutionError
from app.services.tool_execution import operation_key

logger = logging.getLogger(__name__)


class ToolQueueService:
    def __init__(self, db, settings):
        self.db = db
        self.settings = settings

    def enqueue_report(self, report_id: int, risk_level: str | None, *, commit: bool = True, trace_id: str = "") -> list[ToolJob]:
        # Re-read authoritative risk; caller-provided classifications are not authorization.
        # Serialize submissions for the same report on MySQL. This also avoids
        # insert-intention/gap-lock deadlocks while retaining the outer report transaction.
        report = self.db.query(PsychologicalReport).filter_by(id=report_id).with_for_update().one_or_none()
        if report is None:
            raise ValueError("report not found")
        risk_level = report.risk_level
        jobs = [self._find_or_create(ToolJobKind.EXCEL_REPORT.value, report_id, trace_id=trace_id)]
        if risk_level in {RiskLevel.MEDIUM.value, RiskLevel.HIGH.value}:
            case = self._find_or_create(ToolJobKind.CASE_CREATE.value, report_id, trace_id=trace_id)
            jobs.append(case)
            if risk_level == RiskLevel.HIGH.value:
                jobs.append(self._find_or_create(ToolJobKind.ALERT_SEND.value, report_id, case.id, trace_id))
        if commit:
            self.db.commit()
        return jobs

    def _find_or_create(self, kind, report_id, depends_on_job_id=None, trace_id=""):
        key = operation_key(kind, report_id, self.settings)
        existing = self.db.query(ToolJob).filter_by(idempotency_key=key).first()
        if existing:
            return existing
        job = ToolJob(report_id=report_id, kind=kind, idempotency_key=key, trace_id=trace_id,
                      status="PENDING", attempts=0, max_attempts=self.settings.tool_queue_max_attempts,
                      depends_on_job_id=depends_on_job_id, run_after=now(), last_error="")
        try:
            with self.db.begin_nested():
                self.db.add(job)
                self.db.flush()
        except IntegrityError:
            # A unique constraint is the final arbiter under concurrent submission.
            existing = self.db.query(ToolJob).filter_by(idempotency_key=key).with_for_update().one_or_none()
            if existing is None:
                raise
            return existing
        return job

    def replay(self, job_id: int, actor: str, reason: str):
        if not reason.strip():
            raise ValueError("重放必须填写原因")
        job = self.db.get(ToolJob, job_id)
        if job is None:
            raise ValueError("任务不存在")
        if "DELIVERY_UNKNOWN" in job.last_error:
            raise ValueError("投递结果未知，不能自动重发；先核查邮件服务商记录")
        changed = self.db.execute(update(ToolJob).where(ToolJob.id == job_id, ToolJob.status == "DEAD").values(
            status="PENDING", attempts=0, run_after=now(), worker_id=None, lease_token=None, lease_until=None, updated_at=now()))
        if changed.rowcount != 1:
            self.db.rollback()
            raise ValueError("仅允许重放 DEAD 任务")
        self.db.add(ToolAuditRecord(job_id=job.id, report_id=job.report_id, tool_name=job.kind, policy="admin_replay",
                                   allowed=True, status="REPLAYED", reason=reason,
                                   payload=json.dumps({"actor": actor, "previousError": job.last_error}, ensure_ascii=False)))
        self.db.commit()
        return job


class ToolQueueWorker:
    def __init__(self, settings, session_factory=None, client_factory=None):
        self.settings = settings
        self.session_factory = session_factory or SessionLocal
        self.client_factory = client_factory or MindBridgeMcpToolClient
        self.worker_id = uuid.uuid4().hex
        self.stop_event = threading.Event()
        self.dispatcher = None
        counts = {"excel": max(1, settings.tool_queue_excel_workers), "email": max(1, settings.tool_queue_email_workers)}
        self.executors = {key: ThreadPoolExecutor(max_workers=value, thread_name_prefix=f"caretrace-{key}") for key, value in counts.items()}
        self.slots = {key: threading.BoundedSemaphore(value) for key, value in counts.items()}

    def start(self):
        if not self.settings.tool_queue_enabled or self.dispatcher is not None:
            return
        self._recover_running_jobs()
        self.dispatcher = threading.Thread(target=self._loop, name="caretrace-tool-dispatcher", daemon=True)
        self.dispatcher.start()

    def stop(self):
        self.stop_event.set()
        if self.dispatcher is not None:
            self.dispatcher.join(timeout=5)
        for executor in self.executors.values():
            executor.shutdown(wait=False, cancel_futures=True)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self._recover_running_jobs()
                self._dispatch_once()
            except Exception:
                logger.exception("Tool queue dispatch failed")
            self.stop_event.wait(self.settings.tool_queue_poll_interval_seconds)

    def _dispatch_once(self):
        # Independent scans prevent a backlog of Excel exports from starving alerts.
        for pool in ("email", "excel"):
            with self.session_factory() as db:
                kind_filter = ToolJob.kind == ToolJobKind.EXCEL_REPORT.value if pool == "excel" else ToolJob.kind != ToolJobKind.EXCEL_REPORT.value
                candidates = db.query(ToolJob.id).filter(kind_filter,
                    ToolJob.status.in_(["PENDING", "RETRY_WAIT"]), ToolJob.run_after <= now()).order_by(
                    ToolJob.created_at).limit(self.settings.tool_queue_batch_size).all()
            for (job_id,) in candidates:
                if not self.slots[pool].acquire(blocking=False):
                    break
                token = self._claim_job(job_id)
                if token is None:
                    self.slots[pool].release()
                    continue
                try:
                    future = self.executors[pool].submit(self._run_job, job_id, token)
                    future.add_done_callback(lambda _, slot=self.slots[pool]: slot.release())
                except Exception:
                    self.slots[pool].release()
                    raise

    def _claim_job(self, job_id):
        token = uuid.uuid4().hex
        with self.session_factory() as db:
            result = db.execute(update(ToolJob).where(ToolJob.id == job_id, ToolJob.status.in_(["PENDING", "RETRY_WAIT"]),
                                                      ToolJob.run_after <= now()).values(
                status="RUNNING", worker_id=self.worker_id, lease_token=token,
                lease_until=now() + timedelta(seconds=self.settings.tool_queue_lease_seconds), updated_at=now()))
            db.commit()
            return token if result.rowcount == 1 else None

    def _heartbeat(self, job_id, token, done):
        while not done.wait(max(0.1, self.settings.tool_queue_lease_seconds / 3)):
            try:
                with self.session_factory() as db:
                    changed = db.execute(update(ToolJob).where(ToolJob.id == job_id, ToolJob.status == "RUNNING", ToolJob.lease_token == token).values(
                        lease_until=now() + timedelta(seconds=self.settings.tool_queue_lease_seconds)))
                    db.commit()
                    if changed.rowcount == 0:
                        return
            except Exception:
                logger.exception("Lease heartbeat failed for job=%s", job_id)

    def _run_job(self, job_id, token=None):
        done = threading.Event()
        heartbeat = None
        with self.session_factory() as db:
            try:
                job = db.get(ToolJob, job_id)
                if job is None or job.status != "RUNNING" or job.worker_id != self.worker_id or job.lease_token != token:
                    return
                heartbeat = threading.Thread(target=self._heartbeat, args=(job_id, token, done), daemon=True)
                heartbeat.start()
                if not self._dependency_ready(db, job):
                    self._transition(db, job, "PENDING", delay=2, error="等待风险个案创建成功")
                    return
                if job.attempts >= job.max_attempts:
                    raise ToolExecutionError("执行次数已耗尽", code="ATTEMPTS_EXHAUSTED", retryable=False)
                changed = db.execute(update(ToolJob).where(ToolJob.id == job_id, ToolJob.status == "RUNNING", ToolJob.lease_token == token).values(attempts=ToolJob.attempts + 1))
                if changed.rowcount != 1:
                    db.rollback()
                    return
                db.refresh(job)
                self._event(db, job, "TOOL_EXECUTION_STARTED")
                db.commit()
                result = self._execute(db, job)
                self._transition(db, job, "SIMULATED" if result["status"] == "SIMULATED" else "SUCCESS", result=result)
            except Exception as exc:
                db.rollback()
                self._fail_or_dead_letter(db, job_id, exc, token)
            finally:
                done.set()
                if heartbeat:
                    heartbeat.join(timeout=2)

    def _execute(self, db, job):
        case_id = None
        if job.kind == ToolJobKind.ALERT_SEND.value:
            case = db.query(RiskCase).filter_by(report_id=job.report_id).first()
            if case is None:
                raise ToolExecutionError("个案不存在", code="CASE_NOT_READY")
            case_id = case.id
        # Materialize attributes, then release read transaction while MCP runs.
        db.expunge(job)
        db.commit()
        return asyncio.run(self.client_factory(self.settings).execute_job(job, case_id))

    def _dependency_ready(self, db, job):
        if not job.depends_on_job_id:
            return True
        dependency = db.get(ToolJob, job.depends_on_job_id)
        if dependency is None or dependency.status == "DEAD":
            raise ToolExecutionError("依赖任务不存在或已永久失败", code="DEPENDENCY_FAILED", retryable=False)
        return dependency.status == "SUCCESS"

    def _transition(self, db, job, status, *, delay=0, error="", result=None):
        values = dict(status=status, last_error=error, worker_id=None, lease_token=None, lease_until=None,
                      updated_at=now(), run_after=now() + timedelta(seconds=delay))
        if result is not None:
            values["result_json"] = json.dumps(result, ensure_ascii=False)
        token = job.lease_token
        changed = db.execute(update(ToolJob).where(ToolJob.id == job.id, ToolJob.status == "RUNNING", ToolJob.lease_token == token).values(**values).execution_options(synchronize_session=False))
        if changed.rowcount != 1:
            db.rollback()
            return False
        event = {"SUCCESS": "TOOL_EXECUTION_COMPLETED", "SIMULATED": "TOOL_EXECUTION_SIMULATED", "DEAD": "TOOL_EXECUTION_FAILED", "RETRY_WAIT": "TOOL_RETRY_SCHEDULED"}.get(status)
        if event:
            self._event(db, job, event, {"status": status, "error": error, "result": result})
        if status == "DEAD":
            db.add(DeadLetterRecord(job_id=job.id, report_id=job.report_id, kind=job.kind, reason=error,
                                   payload=json.dumps({"idempotencyKey": job.idempotency_key, "attempts": job.attempts, "traceId": job.trace_id})))
        db.commit()
        return True

    def _fail_or_dead_letter(self, db, job_id, exc, token=None):
        job = db.get(ToolJob, job_id, populate_existing=True)
        if job is None or job.status != "RUNNING" or (token is not None and job.lease_token != token):
            return
        code = getattr(exc, "code", type(exc).__name__)
        message = f"{code}: {exc}"
        retryable = getattr(exc, "retryable", True)
        if code in {"RATE_LIMITED", "RATE_LIMIT_UNAVAILABLE", "RESOURCE_BUSY"}:
            # Scheduling deferrals are not failed delivery attempts.
            db.execute(update(ToolJob).where(ToolJob.id == job.id, ToolJob.lease_token == job.lease_token,
                       ToolJob.status == "RUNNING").values(attempts=max(0, job.attempts - 1)))
            db.refresh(job)

        if not retryable or job.attempts >= job.max_attempts:
            self._transition(db, job, "DEAD", error=message)
        else:
            delay = getattr(exc, "retry_after", None)
            if delay is None:
                delay = min(300, self.settings.tool_queue_retry_delay_seconds * 2 ** max(0, job.attempts - 1)) * random.uniform(0.8, 1.2)
            self._transition(db, job, "RETRY_WAIT", delay=max(1, delay), error=message)

    def _recover_running_jobs(self):
        with self.session_factory() as db:
            jobs = db.query(ToolJob).filter(ToolJob.status == "RUNNING", or_(ToolJob.lease_until.is_(None), ToolJob.lease_until < now())).all()
            for job in jobs:
                # CAS also checks expiry: a heartbeat may have renewed since SELECT.
                changed = db.execute(update(ToolJob).where(ToolJob.id == job.id, ToolJob.status == "RUNNING",
                    ToolJob.lease_token == job.lease_token, or_(ToolJob.lease_until.is_(None), ToolJob.lease_until < now())).values(
                    lease_token=uuid.uuid4().hex, worker_id=self.worker_id).execution_options(synchronize_session=False))
                if changed.rowcount != 1:
                    db.rollback()
                    continue
                db.refresh(job)
                self._transition(db, job, "DEAD" if job.attempts >= job.max_attempts else "RETRY_WAIT", error="LEASE_EXPIRED: 恢复超时任务")

    def _event(self, db, job, event_type, extra=None):
        if not job.trace_id:
            return
        metadata = {"jobId": job.id, "reportId": job.report_id, "kind": job.kind, "idempotencyKey": job.idempotency_key,
                    "retryCount": max(0, job.attempts - 1), **(extra or {})}
        db.add(AgentTraceEvent(trace_id=job.trace_id, event_id=uuid.uuid4().hex, event_type=event_type,
                              actor="ToolQueueWorker", task_id=str(job.id), round=0, timestamp=now(),
                              metadata_json=json.dumps(metadata, ensure_ascii=False)))


_worker = None


def get_tool_queue_worker(settings):
    global _worker
    if _worker is None:
        _worker = ToolQueueWorker(settings)
    return _worker
