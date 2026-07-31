from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import SessionLocal
from app.core.enums import RiskLevel, ToolJobKind, ToolJobStatus, ToolStatus
from app.models.entities import DeadLetterRecord, ExcelRecord, PsychologicalReport, ToolJob, now
from app.services.tool_governance import ToolGovernanceService
from app.services.tools import ToolOrchestrationService


logger = logging.getLogger(__name__)


class ToolQueueService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings

    def enqueue_report(self, report_id: int, risk_level: str | None) -> list[ToolJob]:
        excel_job = self._find_or_create(ToolJobKind.EXCEL_REPORT.value, report_id)
        jobs = [excel_job]
        case_job = None
        if risk_level in {RiskLevel.MEDIUM.value, RiskLevel.HIGH.value}:
            case_job = self._find_or_create(ToolJobKind.CASE_CREATE.value, report_id)
            jobs.append(case_job)
        if risk_level == RiskLevel.HIGH.value:
            alert_job = self._find_or_create(ToolJobKind.ALERT_SEND.value, report_id, case_job.id if case_job else None)
            jobs.append(alert_job)
        self.db.commit()
        return jobs

    def _find_or_create(self, kind: str, report_id: int, depends_on_job_id: int | None = None) -> ToolJob:
        existing = (
            self.db.query(ToolJob)
            .filter(ToolJob.report_id == report_id, ToolJob.kind == kind)
            .filter(ToolJob.status.in_([ToolJobStatus.PENDING.value, ToolJobStatus.RUNNING.value, ToolJobStatus.SUCCESS.value]))
            .first()
        )
        if existing is not None:
            return existing
        job = ToolJob(
            report_id=report_id,
            kind=kind,
            status=ToolJobStatus.PENDING.value,
            attempts=0,
            max_attempts=self.settings.tool_queue_max_attempts,
            depends_on_job_id=depends_on_job_id,
            run_after=now(),
            last_error="",
        )
        self.db.add(job)
        self.db.flush()
        return job


class RateLimiter:
    def __init__(self, limit_per_minute: int):
        self.limit = max(0, limit_per_minute)
        self.events: deque[float] = deque()
        self.lock = threading.Lock()

    def allow(self) -> tuple[bool, float]:
        if self.limit <= 0:
            return True, 0.0
        now_ts = time.monotonic()
        with self.lock:
            while self.events and now_ts - self.events[0] >= 60.0:
                self.events.popleft()
            if len(self.events) < self.limit:
                self.events.append(now_ts)
                return True, 0.0
            retry_after = max(1.0, 60.0 - (now_ts - self.events[0]))
            return False, retry_after


class ToolQueueWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.stop_event = threading.Event()
        self.dispatcher: threading.Thread | None = None
        self.excel_executor = ThreadPoolExecutor(
            max_workers=max(1, settings.tool_queue_excel_workers),
            thread_name_prefix="mindbridge-excel",
        )
        self.email_executor = ThreadPoolExecutor(
            max_workers=max(1, settings.tool_queue_email_workers),
            thread_name_prefix="mindbridge-email",
        )
        self.email_limiter = RateLimiter(settings.alert_email_rate_limit_per_minute)

    def start(self) -> None:
        if not self.settings.tool_queue_enabled or self.dispatcher is not None:
            return
        self._recover_running_jobs()
        self.dispatcher = threading.Thread(target=self._loop, name="mindbridge-tool-dispatcher", daemon=True)
        self.dispatcher.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.dispatcher is not None:
            self.dispatcher.join(timeout=5)
        self.excel_executor.shutdown(wait=False, cancel_futures=True)
        self.email_executor.shutdown(wait=False, cancel_futures=True)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._dispatch_once()
            except Exception:
                logger.exception("Tool queue dispatch failed")
            self.stop_event.wait(self.settings.tool_queue_poll_interval_seconds)

    def _dispatch_once(self) -> None:
        db = SessionLocal()
        try:
            jobs = (
                db.query(ToolJob)
                .filter(ToolJob.status == ToolJobStatus.PENDING.value, ToolJob.run_after <= now())
                .order_by(ToolJob.created_at.asc())
                .limit(self.settings.tool_queue_batch_size)
                .all()
            )
            for job in jobs:
                job.status = ToolJobStatus.RUNNING.value
                job.updated_at = now()
                db.add(job)
                db.commit()
                executor = self._executor_for(job)
                executor.submit(self._run_job, job.id)
        finally:
            db.close()

    def _executor_for(self, job: ToolJob) -> ThreadPoolExecutor:
        if job.kind in {ToolJobKind.EXCEL_REPORT.value, ToolJobKind.CASE_CREATE.value}:
            return self.excel_executor
        return self.email_executor

    def _run_job(self, job_id: int) -> None:
        db = SessionLocal()
        try:
            job = db.get(ToolJob, job_id)
            if job is None or job.status != ToolJobStatus.RUNNING.value:
                return
            if not self._dependency_ready(db, job):
                self._requeue(db, job, self._dependency_wait_reason(job), 2.0)
                return
            if job.kind in {ToolJobKind.RISK_ALERT.value, ToolJobKind.ALERT_SEND.value}:
                allowed, retry_after = self.email_limiter.allow()
                if not allowed:
                    self._requeue(db, job, "邮件预警限流中，稍后重试", retry_after)
                    return
            job.attempts += 1
            job.updated_at = now()
            db.add(job)
            db.commit()
            self._execute(db, job)
            job.status = ToolJobStatus.SUCCESS.value
            job.last_error = ""
            job.updated_at = now()
            db.add(job)
            db.commit()
        except Exception as exc:
            try:
                self._fail_or_dead_letter(db, job_id, exc)
            except Exception:
                logger.exception("Failed to record tool job failure")
        finally:
            db.close()

    def _execute(self, db: Session, job: ToolJob) -> None:
        report = db.get(PsychologicalReport, job.report_id)
        if report is None:
            raise RuntimeError(f"report {job.report_id} not found")
        tools = ToolOrchestrationService(db, self.settings)
        if job.kind == ToolJobKind.EXCEL_REPORT.value:
            record = tools.write_excel(report)
            if record.status != ToolStatus.SUCCESS.value:
                raise RuntimeError(record.message)
            return
        if job.kind == ToolJobKind.CASE_CREATE.value:
            tools.create_case(report)
            return
        if job.kind == ToolJobKind.ALERT_SEND.value:
            case = tools.create_case(report)
            record = tools.send_case_alert(case)
            if record.status != ToolStatus.SUCCESS.value:
                raise RuntimeError(record.message)
            return
        if job.kind == ToolJobKind.RISK_ALERT.value:
            record = tools.notify(report)
            if record.status != ToolStatus.SUCCESS.value:
                raise RuntimeError(record.message)
            return
        raise RuntimeError(f"unknown tool job kind: {job.kind}")

    def _dependency_ready(self, db: Session, job: ToolJob) -> bool:
        if job.kind not in {ToolJobKind.RISK_ALERT.value, ToolJobKind.ALERT_SEND.value}:
            return True
        if job.depends_on_job_id:
            dependency = db.get(ToolJob, job.depends_on_job_id)
            return dependency is not None and dependency.status == ToolJobStatus.SUCCESS.value
        if job.kind == ToolJobKind.ALERT_SEND.value:
            from app.models.entities import RiskCase

            return db.query(RiskCase).filter(RiskCase.report_id == job.report_id).first() is not None
        return (
            db.query(ExcelRecord)
            .filter(ExcelRecord.report_id == job.report_id, ExcelRecord.status == ToolStatus.SUCCESS.value)
            .first()
            is not None
        )

    def _dependency_wait_reason(self, job: ToolJob) -> str:
        if job.kind == ToolJobKind.ALERT_SEND.value:
            return "等待风险个案创建成功后再发送预警"
        return "等待 Excel 台账写入成功后再发送预警"

    def _requeue(self, db: Session, job: ToolJob, reason: str, delay_seconds: float) -> None:
        job.status = ToolJobStatus.PENDING.value
        job.last_error = reason
        job.run_after = now() + timedelta(seconds=max(1.0, delay_seconds))
        job.updated_at = now()
        db.add(job)
        db.commit()

    def _fail_or_dead_letter(self, db: Session, job_id: int, exc: Exception) -> None:
        job = db.get(ToolJob, job_id)
        if job is None:
            return
        message = f"{type(exc).__name__}: {exc}"
        job.last_error = message
        job.updated_at = now()
        if job.attempts >= job.max_attempts:
            job.status = ToolJobStatus.DEAD.value
            db.add(
                DeadLetterRecord(
                    job_id=job.id,
                    report_id=job.report_id,
                    kind=job.kind,
                    reason=message,
                    payload=json.dumps(
                        {"reportId": job.report_id, "kind": job.kind, "attempts": job.attempts},
                        ensure_ascii=False,
                    ),
                )
            )
        else:
            job.status = ToolJobStatus.PENDING.value
            job.run_after = now() + timedelta(seconds=self.settings.tool_queue_retry_delay_seconds * max(1, job.attempts))
        db.add(job)
        db.commit()

    def _recover_running_jobs(self) -> None:
        db = SessionLocal()
        try:
            rows = db.query(ToolJob).filter(ToolJob.status == ToolJobStatus.RUNNING.value).all()
            for job in rows:
                job.status = ToolJobStatus.PENDING.value
                job.last_error = "服务重启后恢复未完成任务"
                job.run_after = now()
                job.updated_at = now()
                db.add(job)
            db.commit()
        finally:
            db.close()


_worker: ToolQueueWorker | None = None


def get_tool_queue_worker(settings: Settings) -> ToolQueueWorker:
    global _worker
    if _worker is None:
        _worker = ToolQueueWorker(settings)
    return _worker
