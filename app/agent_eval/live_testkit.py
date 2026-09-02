"""Live 回放：在 InMemory SQLite 中跑真实 runtime，但模型走「真实 provider」。

与 testkit.run_case 的区别：
  - run_case（确定性回归）：注入 ScriptedModelRegistry 假模型，不发起任何真实调用；
  - run_case_live（消融评测）：保留真实 AgentModelRegistry，按 profile/.env 解析到
    Ollama 本地微调模型或 DeepSeek/OpenAI API，用于 A/B 对比。

两者共用同一套 InMemory SQLite + 进程内记忆隔离，以及同一套 Invariant/Rubric 评分层，
因此 A/B 的唯一变量就是「各 Agent 实际调用的模型」。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent_eval.schemas import GoldenCase, TraceRecord
from app.agent_eval.testkit import install_replay_patches
from app.agent_eval.trace_parser import parse_trace
from app.core.config import Settings
from app.core.database import Base
from app.core.enums import MessageRole
from app.models.entities import ChatSession, UserAccount
from app.schemas.dtos import ChatRequest
from app.services.trace import build_trace_record

# 系统内全部可独立配置模型的 profile 别名
ALL_MODEL_ALIASES = ("coordinator", "understanding", "safety", "safety_review", "context", "response")
# 除情绪分类外，默认都走 API 大模型的 profile
API_ALIASES = ("coordinator", "understanding", "safety_review", "context", "response")


def live_eval_settings(profile: str = "env", keep_db: bool = False, db_path: Path | None = None) -> Settings:
    """构造 live 评测配置：读取 .env 真实密钥，再叠加评测安全覆盖与 A/B 路由 profile。"""
    overrides = {
        "database_url": f"sqlite:///{db_path.as_posix()}" if db_path else "sqlite:///:memory:",
        "agent_framework": "event_driven_multi_agent",
        # 消融只对比「模型」差异：关闭向量库，避免 Chroma/embedding 成为混杂因素
        "knowledge_vector_enabled": False,
        "knowledge_vector_required": False,
        "tool_queue_enabled": True,
        "alert_email_delivery_mode": "log",
        # 固定随机性，让 A/B 差异可归因于模型而非采样
        "ai_temperature": 0.0,
    }
    settings = Settings(**overrides)
    apply_ablation_profile(settings, profile)
    return settings


def apply_ablation_profile(settings: Settings, profile: str) -> None:
    """把消融 profile 写入 settings（不依赖外部环境变量，保证 A/B 可复现）。

    A：全 API 基线（所有 Agent 都走 DeepSeek）。
    B：混合（SafetyAgent 情绪/风险分类走本地 LoRA 微调模型，其余走 DeepSeek）。
    env：不覆盖，完全按 .env 的 AGENT_MODEL_* 配置。
    """
    profile = (profile or "env").lower()
    if profile == "env":
        return

    api_model = settings.deepseek_model
    if profile == "a":
        # 基线：全部走 API
        for alias in ALL_MODEL_ALIASES:
            setattr(settings, f"agent_model_{alias}_provider", "deepseek")
            setattr(settings, f"agent_model_{alias}_model", api_model)
        settings.ai_provider = "deepseek"
    elif profile == "b":
        # 实验组：情绪/风险分类走本地微调模型，其余（含语义审核）走 API
        settings.agent_model_safety_provider = "ollama"
        settings.agent_model_safety_model = settings.finetuned_model_name
        for alias in API_ALIASES:
            setattr(settings, f"agent_model_{alias}_provider", "deepseek")
            setattr(settings, f"agent_model_{alias}_model", api_model)
        settings.ai_provider = "deepseek"  # 全局默认仍为 API，仅 safety 被覆盖
    else:
        raise ValueError(f"未知消融 profile: {profile}（支持 A/B/env）")


def resolved_routing_table(settings: Settings) -> dict[str, dict[str, str]]:
    """返回当前 settings 下每个 profile 实际解析到的 provider/model，写入报告便于审计。"""
    from app.services.agent_models import AgentModelRegistry

    registry = AgentModelRegistry(settings)
    table: dict[str, dict[str, str]] = {}
    for alias in ALL_MODEL_ALIASES:
        prof = registry.profile_for_alias(alias)
        table[alias] = {"provider": prof.provider, "model": prof.model}
    return table


def run_case_live(case: GoldenCase, settings: Settings, keep_db: bool = False) -> TraceRecord:
    """用真实模型回放单条 case 的最后一轮，返回解析好的 v2 TraceRecord。"""
    from app.agents.harness import MindBridgeAgentHarness
    from app.harness.runner import InMemoryShortTermMemoryStore

    install_replay_patches()
    InMemoryShortTermMemoryStore.reset()

    db_path: Path | None = None
    if keep_db:
        db_dir = Path("target/agent_eval_live")
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / f"{case.caseId}.sqlite3"
        for suffix in ["", "-wal", "-shm"]:
            candidate = Path(f"{db_path}{suffix}")
            if candidate.exists():
                candidate.unlink()

    if keep_db and db_path:
        engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    else:
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        user = UserAccount(
            username=f"live_{case.caseId}",
            display_name="评测同学",
            password_hash="x",
            roles_csv="ROLE_USER",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        session = ChatSession(public_id=f"live-session-{case.caseId}", user_id=user.id, title=case.caseId[:36])
        db.add(session)
        db.commit()
        db.refresh(session)

        harness = MindBridgeAgentHarness(db, settings)
        for turn in case.conversation[:-1]:
            harness.save_message(user, session, MessageRole(turn.role.upper()), turn.content)
        last_turn = case.conversation[-1]
        if last_turn.role != "user":
            raise ValueError(f"case {case.caseId} 最后一条消息必须是 user，实际是 {last_turn.role}")

        outcome = harness.run(user, ChatRequest(message=last_turn.content, sessionId=session.public_id))
        tool_records = asyncio.run(harness.dispatch_tools(outcome.tool_plan))
        harness.finalize_trace(outcome, tool_records)

        agent_run = outcome.agent_run
        assert agent_run is not None
        record = build_trace_record(
            agent_run,
            original_input=outcome.original_input,
            sanitized_input=outcome.model_input,
            memory_brief=agent_run.memory_brief,
            report_id=outcome.report_id,
        )
        return parse_trace(record)
    finally:
        db.close()
        engine.dispose()
