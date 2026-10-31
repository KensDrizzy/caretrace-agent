from __future__ import annotations

import smtplib # 发送 SMTP 邮件
import ssl # 建立安全 TLS/SSL 连接
import threading # 同进程线程保护
import hashlib
import os
import tempfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from openpyxl import Workbook, load_workbook
from sqlalchemy.orm import Session

from app.services.tool_locks import file_lock
from app.services.tool_errors import DeliveryUnknown
from app.services.tool_rate_limit import require_email_capacity

from app.core.config import Settings
from app.core.enums import RiskCaseStatus, ToolStatus
from app.models.entities import AlertRecord, CaseNote, ExcelRecord, PsychologicalReport, RiskCase, UserAccount, now
from app.services.skills import MindBridgeSkillLibrary


EXCEL_WRITE_LOCK = threading.Lock()

# 负责处理高风险心理报告产生后的后台动作
# 这是工具编排服务，封装了 Excel、个案、邮件和备注等后台动作。
class ToolOrchestrationService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings

# 把心理报告写入 Excel 风险台账，并在数据库中记录写入结果。
    def write_excel(self, report: PsychologicalReport) -> ExcelRecord:
        path = Path(self.settings.excel_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # The file lock survives separate MCP subprocesses; check inside the lock.
        with EXCEL_WRITE_LOCK, file_lock(path.with_suffix(path.suffix + ".lock"), self.settings.tool_lock_timeout_seconds):
            existing = self.db.query(ExcelRecord).filter_by(report_id=report.id, status=ToolStatus.SUCCESS.value).first()
            if existing is not None and path.exists():
                return existing
            workbook = load_workbook(path) if path.exists() else Workbook()
            try:
                sheet = workbook.active
                if sheet.max_row == 1 and sheet.cell(1, 1).value is None:
                    sheet.title = "CareTrace Risk Ledger"

                    for column, label in enumerate(["reportId", "riskLevel", "emotion", "confidence", "summary", "createdAt"], 1):
                        sheet.cell(1, column, label)
                # A previous process may have saved the file before its DB commit.
                if not any(row[0] == report.id for row in sheet.iter_rows(values_only=True)):
                    sheet.append([report.id, report.risk_level, report.emotion, report.confidence, report.summary, report.created_at.isoformat()])
                handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".xlsx")
                os.close(handle)
                try:
                    workbook.save(temporary)
                    with open(temporary, "rb") as output:
                        os.fsync(output.fileno())
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            finally:
                workbook.close()
            record = existing or ExcelRecord(report_id=report.id, file_path=str(path), status=ToolStatus.SUCCESS.value, message="Excel 台账已写入")
            self.db.add(record)
            self.db.commit()
            return record

# 创建风险个案：根据心理报告创建一个需要人工跟进的风险个案。
    def create_case(self, report: PsychologicalReport) -> RiskCase:
        # 检查是否已经创建
        existing = self.db.query(RiskCase).filter(RiskCase.report_id == report.id).first()
        # 如果该报告已经有个案：直接返回原个案，避免重复创建
        if existing is not None:
            return existing
        # 查询学生信息
        user = self.db.get(UserAccount, report.user_id)
        # 创建个案对象
        case = RiskCase(
            report_id=report.id,
            risk_level=report.risk_level,
            status=RiskCaseStatus.OPEN.value,
            owner=self._primary_owner(),
            summary=report.summary,
            # MindBridgeSkillLibrary.counselor_handoff_summary会根据 Skill 模板生成后台交接内容
            handoff_summary=MindBridgeSkillLibrary.counselor_handoff_summary(report, user),
        )
        self.db.add(case)
        self.db.commit()
        return case

# 发送个案预警：根据风险个案找到对应报告，发送或记录预警，并更新个案状态。
    def send_case_alert(self, case: RiskCase) -> AlertRecord:
        report = self.db.get(PsychologicalReport, case.report_id)
        if report is None:
            raise RuntimeError(f"report {case.report_id} not found")
        record = self.notify(report, case)
        if record.status == ToolStatus.SUCCESS.value and case.status == RiskCaseStatus.OPEN.value:
            case.status = RiskCaseStatus.ALERT_SENT.value
        case.updated_at = now()
        self.db.add(case)
        self.db.commit()
        return record

# 确认接手个案：辅导员或管理员确认已经接手这个风险个案。
    def acknowledge_case(self, case_id: int, actor: str, note: str = "") -> RiskCase:
        # 查询个案
        case = self.db.get(RiskCase, case_id)
        if case is None:
            raise RuntimeError(f"case {case_id} not found")
        # 清洗接手人名称，如果传入空字符串，就记录为：unknown
        actor_name = actor.strip() or "unknown"
        # 状态变为 ACKNOWLEDGED
        case.status = RiskCaseStatus.ACKNOWLEDGED.value
        case.acknowledged_by = actor_name
        case.acknowledged_at = now()
        case.updated_at = now()
        self.db.add(case)
        # 生成接手备注
        self._add_case_note(case.id, actor_name, note.strip() or "已确认接手该个案")
        self.db.commit()
        return case

# 添加个案备注：add_case_note，给已有风险个案添加一条人工处理记录。
    def add_case_note(self, case_id: int, actor: str, note: str) -> CaseNote:
        # 查询个案
        case = self.db.get(RiskCase, case_id)
        if case is None:
            raise RuntimeError(f"case {case_id} not found")
        # 更新个案时间
        case.updated_at = now()
        self.db.add(case)
        # 创建备注
        record = self._add_case_note(case.id, actor.strip() or "unknown", note.strip())
        self.db.commit()
        return record

# 邮件/日志通知的核心方法
    def notify(self, report: PsychologicalReport, case: RiskCase | None = None) -> AlertRecord:
        recipient = ",".join(sorted(set(self._recipients()))) or "unconfigured"
        mode = self.settings.alert_email_delivery_mode.strip().lower()
        delivery_key = hashlib.sha256(f"{report.id}:v1:{mode}:{recipient}".encode()).hexdigest()
        record = self.db.query(AlertRecord).filter_by(delivery_key=delivery_key).first()
        if record is not None:
            if record.status in {ToolStatus.SUCCESS.value, ToolStatus.SIMULATED.value, ToolStatus.UNKNOWN.value}:
                return record
            if record.status == ToolStatus.SENDING.value:
                record.status = ToolStatus.UNKNOWN.value
                record.retryable = False
                record.message = "上次发送中断，投递结果未知；请核查邮件服务商记录后人工处理"
                self.db.commit()
                return record
        if record is None:
            record = AlertRecord(report_id=report.id, delivery_key=delivery_key, channel=mode,
                                 recipient=recipient, status=ToolStatus.FAILED.value, message="", retryable=True)
            self.db.add(record)
        def save(status, message, retryable=False):
            record.status = status
            record.message = message
            record.retryable = retryable
            self.db.commit()
            return record
        if mode == "log":
            return save(ToolStatus.SIMULATED.value, "模拟预警已记录，未发送邮件；deliveryMode=log")
        if mode != "smtp":
            return save(ToolStatus.FAILED.value, f"未知投递模式 {mode}")
        missing = self._missing_email_config()
        if missing:
            return save(ToolStatus.FAILED.value, f"缺少配置 {', '.join(missing)}")
        require_email_capacity(self.settings)
        # Persist intent before any external side effect. An orphaned SENDING is
        # deliberately UNKNOWN, never silently resent after process death.
        save(ToolStatus.SENDING.value, "正在提交 SMTP")
        self._smtp_submitted = False
        try:
            self._send_alert_email(report, case)
        except DeliveryUnknown as exc:
            return save(ToolStatus.UNKNOWN.value, str(exc))
        except smtplib.SMTPResponseException as exc:
            if self._smtp_submitted:
                return save(ToolStatus.SUCCESS.value, "SMTP 已接受邮件；退出连接失败不影响已接受结果")
            return save(ToolStatus.FAILED.value, f"SMTP 拒绝：{exc.smtp_code}", 400 <= exc.smtp_code < 500)
        except smtplib.SMTPRecipientsRefused as exc:
            codes = [value[0] for value in exc.recipients.values()]
            return save(ToolStatus.FAILED.value, "SMTP 拒绝所有收件人", bool(codes) and all(400 <= code < 500 for code in codes))
        except (OSError, smtplib.SMTPException) as exc:
            if self._smtp_submitted:
                return save(ToolStatus.SUCCESS.value, "SMTP 已接受邮件；退出连接失败不影响已接受结果")
            return save(ToolStatus.FAILED.value, f"SMTP 连接或认证失败：{type(exc).__name__}", True)
        return save(ToolStatus.SUCCESS.value, "SMTP 已接受邮件；不代表已投递到收件箱或已被阅读")

    def _add_case_note(self, case_id: int, actor: str, note: str) -> CaseNote:
        if not note:
            raise RuntimeError("case note cannot be empty")
        record = CaseNote(case_id=case_id, actor=actor, note=note)
        self.db.add(record)
        return record

    def _missing_email_config(self) -> list[str]:
        missing = []
        if not self.settings.smtp_host.strip():
            missing.append("SMTP_HOST")
        if not self._sender():
            missing.append("ALERT_EMAIL_FROM 或 SMTP_USERNAME")
        if not self._recipients():
            missing.append("ALERT_EMAIL_TO")
        return missing

    def _send_alert_email(self, report: PsychologicalReport, case: RiskCase | None = None) -> None:
        # 创建邮件对象
        message = EmailMessage()
        # 设置邮件头
        message["Subject"] = f"{self.settings.alert_email_subject_prefix} reportId={report.id}"
        message["From"] = self._sender()
        message["To"] = ", ".join(self._recipients())
        message.set_content(self._email_body(report, case))
        # 创建 TLS 安全上下文，使用系统默认的安全证书和 TLS 配置。
        context = ssl.create_default_context()
        # SSL 模式 SMTP_SSL + 465 端口
        if self.settings.smtp_use_ssl:
            with smtplib.SMTP_SSL(
                self.settings.smtp_host,
                self.settings.smtp_port,
                timeout=self.settings.smtp_timeout_seconds,
                context=context,
            ) as server:
                self._send_message(server, message)
            return
        # 普通 SMTP + STARTTLS 模式
        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=self.settings.smtp_timeout_seconds) as server:
            # 连接普通 SMTP，向服务器声明客户端能力。
            server.ehlo()
            # 如果配置要求 TLS：升级为加密连接。SMTP + STARTTLS + 587 端口
            if self.settings.smtp_use_tls:
                server.starttls(context=context)
                server.ehlo()
            # 登录并发送
            self._send_message(server, message)

    def _send_message(self, server: smtplib.SMTP, message: EmailMessage) -> None:
        if self.settings.smtp_username:
            server.login(self.settings.smtp_username, self.settings.smtp_password)
        try:
            refused = server.send_message(message)
        except (smtplib.SMTPServerDisconnected, OSError) as exc:
            raise DeliveryUnknown("SMTP 提交期间连接中断，投递结果未知；需人工核查") from exc
        if refused:
            raise DeliveryUnknown("SMTP 部分收件人被拒绝；其他收件人可能已接受，需人工核查")
        self._smtp_submitted = True

# 构造邮件正文：作用是把报告、学生、个案和交接摘要组合成邮件正文。
    def _email_body(self, report: PsychologicalReport, case: RiskCase | None = None) -> str:
        user = self.db.get(UserAccount, report.user_id)
        username = user.username if user else f"userId={report.user_id}"
        display_name = user.display_name if user else ""
        # 获取交接摘要
        handoff = case.handoff_summary if case else MindBridgeSkillLibrary.counselor_handoff_summary(report, user)
        return "\n".join(
            [
                "CareTrace 检测到一条高风险心理预警，请尽快安排辅导员或管理员跟进。",
                "",
                f"个案ID：{case.id}" if case else "个案ID：未创建",
                f"报告ID：{report.id}",
                f"学生：{display_name} ({username})" if display_name else f"学生：{username}",
                f"风险等级：{report.risk_level}",
                f"情绪标签：{report.emotion}",
                f"置信度：{report.confidence}",
                f"摘要：{report.summary}",
                f"创建时间：{report.created_at.isoformat()}",
                "",
                "交接摘要：",
                handoff,
            ]
        )

    def _primary_owner(self) -> str:
        recipients = self._recipients()
        if recipients:
            return recipients[0]
        return "unassigned"

    def _sender(self) -> str:
        return self.settings.alert_email_from.strip() or self.settings.smtp_username.strip()

    def _recipients(self) -> list[str]:
        normalized = self.settings.alert_email_to.replace(";", ",")
        return [recipient.strip() for recipient in normalized.split(",") if recipient.strip()]
