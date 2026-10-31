"""Offline additive migration for existing MySQL/SQLite installations.

Stop all application/worker processes before running. Historical rows are kept;
active duplicate jobs are retired and dependencies redirected to the canonical job.
"""
from collections import defaultdict
import hashlib

from sqlalchemy import inspect, text


def migrate_tool_schema(engine, settings):
    from app.models.entities import Base
    from app.services.tool_execution import operation_key

    # create_all handles entirely new databases and the new receipt table.
    Base.metadata.create_all(engine)
    additions = {
        "tool_jobs": {
            "idempotency_key": "VARCHAR(191)", "trace_id": "VARCHAR(64) DEFAULT ''",
            "worker_id": "VARCHAR(64)", "lease_token": "VARCHAR(64)", "lease_until": "DATETIME",
            "result_json": "TEXT",
        },
        "alert_records": {"delivery_key": "VARCHAR(64)", "retryable": "BOOLEAN DEFAULT 1"},
    }
    with engine.begin() as conn:
        for table, columns in additions.items():
            existing = {item["name"] for item in inspect(conn).get_columns(table)}
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
        if engine.dialect.name == "mysql":
            types = {column["name"]: column["type"] for column in inspect(conn).get_columns("tool_jobs")}
            for name in ("run_after", "lease_until"):
                if getattr(types[name], "fsp", None) != 6:
                    conn.execute(text(f"ALTER TABLE tool_jobs MODIFY COLUMN {name} DATETIME(6)"))
        # Never convert a simulated historical record into a real delivery.
        conn.execute(text("UPDATE alert_records SET status='SIMULATED', channel='log' WHERE message LIKE '%deliveryMode=log%'"))
        alerts = conn.execute(text("SELECT id, report_id, recipient, status, channel, delivery_key FROM alert_records ORDER BY id")).mappings().all()
        used = {row['delivery_key'] for row in alerts if row['delivery_key']}
        for row in alerts:
            if row['delivery_key'] or row['status'] not in {'SUCCESS', 'SIMULATED'}:
                continue
            mode = 'log' if row['channel'] == 'log' else 'smtp'
            recipient = ','.join(sorted(set(x.strip() for x in row['recipient'].replace(';', ',').split(',') if x.strip())))
            if recipient == 'log':
                recipient = 'unconfigured'
            key = hashlib.sha256(f"{row['report_id']}:v1:{mode}:{recipient}".encode()).hexdigest()
            if key not in used:
                conn.execute(text('UPDATE alert_records SET delivery_key=:key WHERE id=:id'), {'key': key, 'id': row['id']})
                used.add(key)
        conn.execute(text("UPDATE risk_cases SET status='OPEN' WHERE status='ALERT_SENT' AND EXISTS (SELECT 1 FROM alert_records a WHERE a.report_id=risk_cases.report_id AND a.status='SIMULATED') AND NOT EXISTS (SELECT 1 FROM alert_records a WHERE a.report_id=risk_cases.report_id AND a.status='SUCCESS')"))
        rows = conn.execute(text("SELECT id, report_id, kind, status, idempotency_key FROM tool_jobs ORDER BY id")).mappings().all()
        groups = defaultdict(list)
        for row in rows:
            groups[(row['kind'], row['report_id'])].append(row)
        for (kind, report_id), jobs in groups.items():
            unmigrated = [job for job in jobs if not job['idempotency_key']]
            if not unmigrated:
                continue
            key_settings = settings
            if kind in {"ALERT_SEND", "RISK_ALERT"}:
                history = [a for a in alerts if a["report_id"] == report_id and a["status"] in {"SUCCESS", "SIMULATED"}]
                if history:
                    delivered = next((a for a in history if a["status"] == "SUCCESS"), history[0])
                    mode = "log" if delivered["status"] == "SIMULATED" else "smtp"
                    recipient = delivered["recipient"] if delivered["recipient"] != "log" else ""
                    key_settings = settings.model_copy(update={"alert_email_delivery_mode": mode, "alert_email_to": recipient})
                    if mode == "log":
                        conn.execute(text("UPDATE tool_jobs SET status='SIMULATED' WHERE report_id=:report AND kind=:kind AND status='SUCCESS'"), {"report": report_id, "kind": kind})
            key = operation_key(kind, report_id, key_settings)
            keyed = next((job for job in jobs if job['idempotency_key'] == key), None)
            canonical = keyed or min(unmigrated, key=lambda job: (job['status'] != 'SUCCESS', job['id']))
            if not keyed:
                conn.execute(text('UPDATE tool_jobs SET idempotency_key=:key WHERE id=:id'), {'key': key, 'id': canonical['id']})
            for job in unmigrated:
                if job['id'] == canonical['id']:
                    continue
                conn.execute(text("UPDATE tool_jobs SET idempotency_key=:key, status='DEAD', last_error='MIGRATION_DUPLICATE: superseded historical job' WHERE id=:id"), {'key': f"legacy:{job['id']}", 'id': job['id']})
                conn.execute(text('UPDATE tool_jobs SET depends_on_job_id=:canonical WHERE depends_on_job_id=:old'), {'canonical': canonical['id'], 'old': job['id']})
        for table, name, columns, unique in [
            ('tool_jobs', 'uq_tool_jobs_idempotency', 'idempotency_key', True),
            ('alert_records', 'uq_alert_delivery', 'delivery_key', True),
            ('tool_jobs', 'ix_tool_jobs_lease_until', 'lease_until', False),
            ('tool_jobs', 'ix_tool_jobs_trace_id', 'trace_id', False),
        ]:
            existing = inspect(conn).get_indexes(table) + inspect(conn).get_unique_constraints(table)
            if not any(item.get('column_names') == [columns] and (not unique or item.get('unique', True)) for item in existing):
                qualifier = 'UNIQUE ' if unique else ''
                conn.execute(text(f'CREATE {qualifier}INDEX {name} ON {table} ({columns})'))
    return True
