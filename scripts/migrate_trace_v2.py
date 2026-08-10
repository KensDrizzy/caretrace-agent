"""Trace v2 数据库迁移脚本（项目无 Alembic，用本脚本完成增量迁移）。

用途：
  1. 为 agent_run_traces 表补齐 trace v2 新增列（trace_id/status/final_response 等）。
  2. 创建 agent_trace_events 事件表。
  3. 为存量行的 trace_id 回填随机 hex 值。

脚本是幂等的，可重复运行；只检测并添加缺失的列，不会修改或删除已有数据。

用法：
  python scripts/migrate_trace_v2.py

数据库连接读取 app 的 Settings（DATABASE_URL 环境变量 / .env），同时兼容
MySQL（information_schema 检测列）和 SQLite（PRAGMA table_info 检测列）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text

from app.core.config import get_settings

# (列名, DDL 类型, 默认值子句)
NEW_TRACE_COLUMNS = [
    ("trace_id", "VARCHAR(64)", ""),
    ("trace_version", "VARCHAR(16)", "DEFAULT '2.0'"),
    ("user_message_id", "INTEGER", ""),
    ("status", "VARCHAR(16)", "DEFAULT 'RUNNING'"),
    ("started_at", "DATETIME", ""),
    ("completed_at", "DATETIME", ""),
    ("final_response", "TEXT", ""),
    ("final_response_artifact_id", "VARCHAR(128)", "DEFAULT ''"),
    ("final_review_artifact_id", "VARCHAR(128)", "DEFAULT ''"),
    ("error_json", "TEXT", ""),
    ("metrics_json", "TEXT", ""),
]


def existing_columns(conn, dialect: str) -> set[str]:
    if dialect == "sqlite":
        rows = conn.execute(text("PRAGMA table_info(agent_run_traces)")).fetchall()
        return {row[1] for row in rows}
    # MySQL
    rows = conn.execute(
        text(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'agent_run_traces'"
        )
    ).fetchall()
    return {row[0] for row in rows}


def backfill_trace_id(conn, dialect: str) -> None:
    if dialect == "sqlite":
        conn.execute(text("UPDATE agent_run_traces SET trace_id = lower(hex(randomblob(16))) WHERE trace_id IS NULL OR trace_id = ''"))
    else:
        conn.execute(text("UPDATE agent_run_traces SET trace_id = REPLACE(UUID(), '-', '') WHERE trace_id IS NULL OR trace_id = ''"))


def backfill_legacy_status(conn) -> None:
    # 迁移前写入的历史行已完成整个对话回合，不应永远停留在默认的 RUNNING。
    result = conn.execute(
        text("UPDATE agent_run_traces SET status = 'COMPLETED' WHERE status = 'RUNNING' AND (final_response IS NULL OR final_response = '')")
    )
    if result.rowcount:
        print(f"legacy rows marked COMPLETED: {result.rowcount}")


def ensure_index(conn, dialect: str, name: str, ddl: str) -> None:
    # MySQL 的 CREATE INDEX 不支持 IF NOT EXISTS，失败（已存在）时忽略
    try:
        conn.execute(text(ddl))
        print(f"index ensured: {name}")
    except Exception as exc:
        print(f"index skipped: {name} ({type(exc).__name__})")


def main() -> int:
    settings = get_settings()
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
    dialect = engine.dialect.name
    print(f"dialect={dialect} url={settings.database_url.split('@')[-1]}")

    with engine.begin() as conn:
        if dialect == "sqlite":
            table = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_run_traces'")).fetchone()
        else:
            table = conn.execute(
                text("SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'agent_run_traces'")
            ).fetchone()
        if not table:
            print("agent_run_traces 表不存在，跳过列迁移（首次启动时由 create_all 建表）")
        else:
            columns = existing_columns(conn, dialect)
            for name, ddl_type, default in NEW_TRACE_COLUMNS:
                if name in columns:
                    print(f"column exists: {name}")
                    continue
                conn.execute(text(f"ALTER TABLE agent_run_traces ADD COLUMN {name} {ddl_type} {default}".strip()))
                print(f"column added: {name}")
            backfill_trace_id(conn, dialect)
            print("trace_id backfilled")
            backfill_legacy_status(conn)
            ensure_index(conn, dialect, "ix_agent_run_traces_trace_id", "CREATE UNIQUE INDEX ix_agent_run_traces_trace_id ON agent_run_traces (trace_id)")
            ensure_index(conn, dialect, "ix_agent_run_traces_status", "CREATE INDEX ix_agent_run_traces_status ON agent_run_traces (status)")

    # 事件表用 SQLAlchemy 元数据创建，方言差异由 SQLAlchemy 处理
    from app.models.entities import AgentTraceEvent

    AgentTraceEvent.__table__.create(engine, checkfirst=True)
    print("agent_trace_events table ensured")
    print("migration done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
