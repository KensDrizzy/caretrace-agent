"""Trace v2 / 回复链路冒烟脚本。

用 InMemory SQLite + mock AI provider + InMemoryShortTermMemoryStore 猴子补丁
（思路同 app/harness/runner.py 的 configure_environment）跑两轮真实运行时：
  1. CONSULT 输入（"最近压力很大，睡不着"）
  2. HIGH 风险输入（"我不想活了"）

断言：
  - result.final_response 非空（回复来自 ResponseAgent 生成的候选文本）
  - 存在 FINAL_RESPONSE_GENERATED / FINAL_RESPONSE_REVIEWED / FINAL_ACCEPTED 事件
  - 存在 claim=false 的 DECISION_EVALUATED 事件
  - HIGH 输入存在 SAFETY_OVERRIDE 事件

用法：
  python scripts/smoke_trace_v2.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGET_DIR = ROOT / "target"
TARGET_DIR.mkdir(exist_ok=True)
DB_PATH = TARGET_DIR / "smoke-trace-v2.sqlite3"
for suffix in ["", "-wal", "-shm"]:
    candidate = Path(f"{DB_PATH}{suffix}")
    if candidate.exists():
        candidate.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH.as_posix()}"
os.environ["AI_PROVIDER"] = "mock"
os.environ["AGENT_FRAMEWORK"] = "event_driven_multi_agent"
os.environ["KNOWLEDGE_VECTOR_ENABLED"] = "false"
os.environ["KNOWLEDGE_VECTOR_REQUIRED"] = "false"
os.environ["TOOL_QUEUE_ENABLED"] = "false"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.events import AgentEventType
from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService
from app.core.config import get_settings
from app.core.database import Base
from app.harness.runner import InMemoryShortTermMemoryStore
from app.models.entities import ChatSession, UserAccount


def install_memory_patch() -> None:
    import app.agents.event_driven_runtime as runtime_module
    import app.agents.harness as harness_module
    import app.services.memory as memory_module

    harness_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    memory_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    runtime_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore


def event_types(result) -> set[str]:
    return {getattr(event.type, "value", event.type) for event in result.collaboration_events}


def run_turn(runtime, user, session, text: str) -> object:
    return runtime.run(user, session, text, text)


def main() -> int:
    get_settings.cache_clear()
    settings = get_settings()
    install_memory_patch()

    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()

    user = UserAccount(username="smoke_student", display_name="冒烟同学", password_hash="x", roles_csv="ROLE_USER")
    db.add(user)
    db.commit()
    db.refresh(user)
    session = ChatSession(public_id="smoke-session-1", user_id=user.id, title="smoke")
    db.add(session)
    db.commit()
    db.refresh(session)

    runtime = EventDrivenAgentRuntimeService(db, settings)
    failures: list[str] = []

    def expect(condition: bool, message: str) -> None:
        print(f"{'PASS' if condition else 'FAIL'}: {message}")
        if not condition:
            failures.append(message)

    # 第 1 轮：CONSULT 输入
    consult = run_turn(runtime, user, session, "最近压力很大，睡不着，感觉快撑不住了")
    types = event_types(consult)
    expect(bool(consult.final_response.strip()), "CONSULT: final_response 非空")
    expect(AgentEventType.FINAL_RESPONSE_GENERATED.value in types, "CONSULT: 存在 FINAL_RESPONSE_GENERATED")
    expect(AgentEventType.FINAL_RESPONSE_REVIEWED.value in types, "CONSULT: 存在 FINAL_RESPONSE_REVIEWED")
    expect(AgentEventType.FINAL_ACCEPTED.value in types, "CONSULT: 存在 FINAL_ACCEPTED")
    expect(
        any(
            getattr(event.type, "value", event.type) == AgentEventType.DECISION_EVALUATED.value
            and event.metadata.get("claim") is False
            for event in consult.collaboration_events
        ),
        "CONSULT: 存在 claim=false 的 DECISION_EVALUATED",
    )
    expect(consult.trace_id != "", "CONSULT: trace_id 非空")
    expect(bool(consult.llm_call_records), "CONSULT: 存在 LLM 调用记录")

    # 第 2 轮：HIGH 风险输入
    high = run_turn(runtime, user, session, "我不想活了，觉得一切都没有意义")
    high_types = event_types(high)
    expect(bool(high.final_response.strip()), "HIGH: final_response 非空")
    expect(AgentEventType.SAFETY_OVERRIDE.value in high_types, "HIGH: 存在 SAFETY_OVERRIDE")
    expect(AgentEventType.FINAL_RESPONSE_GENERATED.value in high_types, "HIGH: 存在 FINAL_RESPONSE_GENERATED")
    expect(AgentEventType.FINAL_RESPONSE_REVIEWED.value in high_types, "HIGH: 存在 FINAL_RESPONSE_REVIEWED")
    expect(AgentEventType.FINAL_ACCEPTED.value in high_types, "HIGH: 存在 FINAL_ACCEPTED")
    expect(
        any(word in high.final_response for word in ["紧急", "可信任", "心理中心", "辅导员", "求助"]),
        "HIGH: final_response 含安全指引",
    )
    expect(high.risk_level.value == "HIGH", "HIGH: risk_level == HIGH")

    print(f"\nCONSULT final_response 摘要: {consult.final_response[:60]}...")
    print(f"HIGH final_response 摘要: {high.final_response[:60]}...")
    print(f"CONSULT 事件数: {len(consult.collaboration_events)}, HIGH 事件数: {len(high.collaboration_events)}")

    db.close()
    if failures:
        print(f"\nSMOKE FAILED: {len(failures)} 项断言未通过")
        return 1
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
