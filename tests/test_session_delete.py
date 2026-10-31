"""Session deletion is student-side only; retained records remain auditable."""
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.harness import MindBridgeAgentHarness
from app.api.routes import router
from app.core.database import Base, get_db
from app.core.security import hash_password
from app.models.entities import AgentRunTrace, ChatMessage, ChatSession, PsychologicalReport, UserAccount
from app.services.memory import RedisShortTermMemoryStore
from app.services.report import ReportService


@pytest.fixture
def env(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    owner = UserAccount(username="owner", display_name="Owner", password_hash=hash_password("test"))
    stranger = UserAccount(username="stranger", display_name="Stranger", password_hash=hash_password("test"))
    db.add_all([owner, stranger])
    db.flush()
    session = ChatSession(public_id="delete-me", title="测试会话", user_id=owner.id)
    other = ChatSession(public_id="keep-me", title="保留会话", user_id=owner.id)
    db.add_all([session, other])
    db.flush()
    db.add(ChatMessage(user_id=owner.id, session_id=session.id, role="USER", content="测试消息"))
    db.add(AgentRunTrace(user_id=owner.id, session_id=session.id, intent="CHAT", original_input="测试消息", sanitized_input="测试消息"))
    db.add(PsychologicalReport(user_id=owner.id, session_id=session.id, content="测试", intent="CONSULT", emotion="ANXIETY", emotion_score=1, risk_level="LOW", confidence=.9, summary="测试报告"))
    db.commit()
    clear_calls = []
    monkeypatch.setattr(RedisShortTermMemoryStore, "_connect", lambda self: None)
    monkeypatch.setattr(RedisShortTermMemoryStore, "clear", lambda self, public_id: clear_calls.append(public_id))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        yield db, owner, session, client, clear_calls
    db.close()
    engine.dispose()


def test_delete_hides_session_but_preserves_messages_reports_and_traces(env):
    db, owner, session, client, clear_calls = env
    auth = ("owner", "test")
    assert client.delete("/api/chat/sessions/delete-me", auth=auth).status_code == 204
    assert clear_calls == ["delete-me"]
    assert session.deleted_at is not None
    listed = client.get("/api/chat/sessions", auth=auth).json()
    assert [item["sessionId"] for item in listed] == ["keep-me"]
    assert client.get("/api/chat/sessions/delete-me", auth=auth).status_code == 404
    assert client.patch("/api/chat/sessions/delete-me", json={"title": "恢复"}, auth=auth).status_code == 404
    assert client.post("/api/chat/stream", json={"sessionId": "delete-me", "message": "继续"}, auth=auth).status_code == 404
    assert db.query(ChatMessage).count() == 1
    assert db.query(PsychologicalReport).count() == 1
    assert db.query(AgentRunTrace).count() == 1
    service = ReportService(db)
    assert service.conversation("delete-me").messages[0].content == "测试消息"
    assert service.latest_reports()[0].sessionId == "delete-me"
    assert service.agent_run_traces()[0].sessionId == "delete-me"
    harness = MindBridgeAgentHarness.__new__(MindBridgeAgentHarness)
    harness.db = db
    with pytest.raises(ValueError, match="Session not found"):
        harness._resolve_session(owner, "delete-me", "继续")


def test_delete_is_idempotent_and_retries_cache_cleanup(env):
    db, owner, session, client, clear_calls = env
    for _ in range(2):
        response = client.delete("/api/chat/sessions/delete-me", auth=("owner", "test"))
        assert response.status_code == 204
        assert response.content == b""
    assert clear_calls == ["delete-me", "delete-me"]
    assert db.query(ChatSession).count() == 2


def test_delete_requires_auth_and_ownership(env):
    db, owner, session, client, clear_calls = env
    assert client.delete("/api/chat/sessions/delete-me").status_code == 401
    assert client.delete("/api/chat/sessions/delete-me", auth=("stranger", "test")).status_code == 404
    assert client.delete("/api/chat/sessions/missing", auth=("owner", "test")).status_code == 404
    assert session.deleted_at is None
    assert clear_calls == []
    assert len(ReportService(db).user_sessions(owner.id)) == 2


def test_startup_adds_deleted_at_to_legacy_schema_idempotently(env, monkeypatch):
    db, _, session, _, _ = env
    from app.core import bootstrap
    engine = db.get_bind()
    db.expunge_all()
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE chat_sessions DROP COLUMN deleted_at"))
    monkeypatch.setattr(bootstrap, "engine", engine)
    bootstrap.create_schema()
    bootstrap.create_schema()
    assert "deleted_at" in {column["name"] for column in inspect(engine).get_columns("chat_sessions")}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM chat_sessions WHERE deleted_at IS NULL")).scalar() == 2


def test_cache_clear_uses_session_key_and_tolerates_redis_outage():
    memory = RedisShortTermMemoryStore.__new__(RedisShortTermMemoryStore)
    memory.client = Mock()
    memory.clear("delete-me")
    memory.client.delete.assert_called_once_with("mindbridge:short-term-memory:delete-me")
    memory.client.delete.side_effect = ConnectionError("offline")
    memory.clear("delete-me")
    memory.client = None
    memory.clear("delete-me")
