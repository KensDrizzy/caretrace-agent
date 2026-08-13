from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import Base, engine
from app.core.security import hash_password
from app.models.entities import UserAccount
from app.services.knowledge import KnowledgeService


def create_schema() -> None:
    Base.metadata.create_all(bind=engine)
    # Existing MySQL deployments may still have 64 KiB TEXT columns. Complete
    # multi-turn prompts plus RAG evidence can exceed that size, so widen only
    # the trace payload columns. SQLite/test environments need no migration.
    if engine.dialect.name == "mysql":
        with engine.begin() as connection:
            rows = connection.execute(text(
                "SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'agent_run_traces' "
                "AND COLUMN_NAME IN ('agent_steps_json', 'retrieved_knowledge_json', 'response_messages_json')"
            )).all()
            data_types = {row[0]: str(row[1]).lower() for row in rows}
            for column in ("agent_steps_json", "retrieved_knowledge_json", "response_messages_json"):
                if data_types.get(column) != "mediumtext":
                    connection.execute(text(f"ALTER TABLE agent_run_traces MODIFY COLUMN {column} MEDIUMTEXT NOT NULL"))


def seed_data(db: Session) -> None:
    if db.query(UserAccount).count() == 0:
        admin = UserAccount(
            username="admin",
            display_name="Counselor Admin",
            password_hash=hash_password("admin123"),
        )
        admin.roles = {"ROLE_ADMIN", "ROLE_USER"}
        student = UserAccount(
            username="student",
            display_name="Demo Student",
            password_hash=hash_password("student123"),
        )
        student.roles = {"ROLE_USER"}
        db.add_all([admin, student])
        db.commit()

    service = KnowledgeService(db, get_settings())
    root = Path(__file__).resolve().parents[1]
    for file in sorted((root / "knowledge").glob("*.md")):
        # RAG 知识库的增量入库判断：先对整篇文档算 SHA-256。数据库里如果已经有同一个 source，
        # 而且保存的整篇文档 hash 一样，就不重新切块、不重新生成向量；只有内容发生变化才重新 ingest。
        service.ensure_source(file.name, file.read_text(encoding="utf-8"))
