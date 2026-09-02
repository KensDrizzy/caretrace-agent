from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.entities import AgentRunTrace, ChatSession, UserAccount
from app.schemas.dtos import ChatSessionRenameRequest
from app.services.report import ReportService


def make_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def add_user(db, username: str) -> UserAccount:
    user = UserAccount(username=username, display_name=username, password_hash="test")
    db.add(user)
    db.flush()
    return user


def test_rename_session_keeps_public_id_and_trace_association():
    db = make_db()
    try:
        user = add_user(db, "student")
        chat_session = ChatSession(public_id="session-public-id", title="旧名称", user_id=user.id)
        db.add(chat_session)
        db.flush()
        trace = AgentRunTrace(
            user_id=user.id,
            session_id=chat_session.id,
            intent="CHAT",
            original_input="你好",
            sanitized_input="你好",
        )
        db.add(trace)
        db.commit()
        original_session_pk = chat_session.id
        original_updated_at = chat_session.updated_at

        renamed = ReportService(db).rename_user_session(user.id, chat_session.public_id, "我的新会话")

        db.refresh(trace)
        assert renamed.sessionId == "session-public-id"
        assert renamed.title == "我的新会话"
        assert trace.session_id == original_session_pk
        assert chat_session.updated_at == original_updated_at
        assert ReportService(db).agent_run_traces()[0].sessionId == "session-public-id"
    finally:
        db.close()


def test_rename_session_enforces_owner():
    db = make_db()
    try:
        owner = add_user(db, "owner")
        stranger = add_user(db, "stranger")
        db.add(ChatSession(public_id="private-session", title="私有会话", user_id=owner.id))
        db.commit()

        try:
            ReportService(db).rename_user_session(stranger.id, "private-session", "越权修改")
        except ValueError as exc:
            assert str(exc) == "Session not found"
        else:
            raise AssertionError("Expected cross-user rename to fail")
    finally:
        db.close()


def test_rename_request_normalizes_whitespace():
    request = ChatSessionRenameRequest(title="  考试   压力\n记录  ")
    assert request.title == "考试 压力 记录"
