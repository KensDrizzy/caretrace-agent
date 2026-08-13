"""确定性回放 testkit：在 InMemory 环境里跑真实 runtime，产出 v2 TraceRecord。

runner 和 pytest 都依赖本模块，公开 API 保持稳定：
  - ScriptedAiClient(settings, script=None, metrics_hook=None)
  - ScriptedModelRegistry(settings, script=None).client_for(agent_name, metrics_hook=None)
  - run_case(case, settings=None, keep_db=False) -> TraceRecord
"""

from __future__ import annotations

import asyncio
import itertools
import time
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent_eval.schemas import GoldenCase, ModelScript, TraceRecord
from app.agent_eval.trace_parser import parse_trace
from app.core.config import Settings
from app.core.enums import MessageRole
from app.models.entities import ChatSession, UserAccount
from app.schemas.dtos import ChatRequest
from app.services.agent_models import AgentModelRegistry
from app.services.ai import (
    AiClient,
    _mock_rewrite_query,
    has_consult_signal,
    has_high_risk_signal,
)
from app.services.trace import build_trace_record

_MEMORY_PATCHED = False

HIGH_RISK_DEFAULT_RESPONSE = (
    "我听到你现在已经痛苦到觉得撑不下去了。现在最重要的是先保证你的安全：请马上联系身边可信任的人，"
    "或者直接联系辅导员、学校心理中心；如果情况紧急，请拨打当地紧急求助电话（110/120）。"
    "接下来 10 分钟，请先把自己移到有人在的地方，把可能伤害自己的东西放远一点。"
    "如果可以，回我一句：你现在身边有没有可以马上联系的人？"
)
CONSULT_DEFAULT_RESPONSE = (
    "我听到你最近压力很大，还影响到了睡眠，这种状态确实让人很消耗。你可以先做两件具体的小事："
    "今晚把最担心的事情写成清单，只选一个最小步骤先处理；睡前 30 分钟把手机和学习任务放远一点，"
    "用缓慢呼吸帮身体降下来。建议先试三天看看变化。如果这种状态持续一周以上，建议联系学校心理中心或辅导员一起看一看。"
)
CHAT_DEFAULT_RESPONSE = "我在。这个问题可以直接拆开来看：先说结论，再分步骤说明。你想先从哪一部分开始？"
GENERIC_FALLBACK_RESPONSE = (
    "我在。先把你现在最具体的困扰说出来，我们可以一步一步拆开。"
    "如果情况已经影响安全，请马上联系身边可信任的人或学校心理中心。"
)

# 确定性语义审核（对应 SafetyAgent._llm_semantic_review）：扮演"审核模型"，
# 用固定模式识别贬低语气/诊断表述/危险认同，输出与生产一致的 JSON 结论。
REVIEW_UNSAFE_PATTERNS: list[tuple[list[str], str]] = [
    (["ResponseAgent", "CoordinatorAgent", "SafetyAgent", "共享黑板", "黑板上的", "artifact"], "INTERNAL_PROCESS_DISCLOSURE"),
    (["尊重你的决定", "解脱", "怎么舒服怎么来", "别硬撑"], "UNSAFE_CONTENT"),
    (["八成是", "是抑郁症", "是焦虑症", "赶紧吃药", "做个量表"], "DIAGNOSTIC_CLAIM"),
    (["矫情", "别人比", "就你不行", "好高骛远", "学会感恩", "少玩点手机", "睡一觉就好", "谁大学"], "DISMISSIVE_TONE"),
]


class ScriptedAiClient(AiClient):
    """按 system prompt 内容分派的确定性 fake 模型。"""

    def __init__(self, settings: Settings, script: ModelScript | None = None, metrics_hook=None, response_cursor=None):
        super().__init__(settings, metrics_hook=metrics_hook)
        self.script = script or ModelScript()
        # responses 序列的共享游标：AgentModelRegistry.client_for 每次新建 client，
        # 游标必须由 registry 注入才能跨多次 complete() 调用保持推进。
        self._response_cursor = response_cursor if response_cursor is not None else itertools.count()

    def complete(self, messages) -> str:
        started = time.perf_counter()
        content = self._dispatch(messages)
        self._emit_metrics("mock", started, status="OK")
        return content

    def _dispatch(self, messages) -> str:
        system = " ".join(m.content for m in messages if m.role == "system")
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if "意图分类器" in system:
            return self._intent(last)
        if "严格 JSON" in system:
            return self._assessment_json(last)
        if "只输出查询词" in system:
            current = last.split("当前输入：")[-1].strip().split("\n")[0].strip() if "当前输入：" in last else last
            return _mock_rewrite_query(current)
        if "只输出 1-3 条中文记忆要点" in system:
            return f"用户近期提到：{last.split(chr(10))[0][:40] or '情绪困扰'}。"
        if "审核一段即将发送给学生的回复" in system:
            return self._review_json(last)
        if "你的整段输出会原样展示给学生" in system:
            return self._response(system, last)
        return GENERIC_FALLBACK_RESPONSE

    def _review_json(self, user_message: str) -> str:
        text = user_message.split("待审核回复：")[-1] if "待审核回复：" in user_message else user_message
        for words, reason_code in REVIEW_UNSAFE_PATTERNS:
            hit = next((word for word in words if word in text), None)
            if hit:
                return f'{{"approved": false, "reasonCode": "{reason_code}", "evidence": "{hit}"}}'
        return '{"approved": true, "reasonCode": "OK", "evidence": ""}'

    def _intent(self, text: str) -> str:
        if self.script.intent:
            return self.script.intent.upper()
        if has_high_risk_signal(text):
            return "RISK"
        if has_consult_signal(text):
            return "CONSULT"
        return "CHAT"

    def _assessment_json(self, text: str) -> str:
        risk = (self.script.risk or "").upper()
        if risk not in {"LOW", "MEDIUM", "HIGH"}:
            if has_high_risk_signal(text):
                risk = "HIGH"
            elif has_consult_signal(text):
                lowered = text.lower()
                risk = "MEDIUM" if any(word in lowered for word in ["抑郁", "低落", "崩溃", "难过", "depress", "hopeless"]) else "LOW"
            else:
                risk = "LOW"
        emotion = {"HIGH": "HIGH_RISK", "MEDIUM": "DEPRESSED", "LOW": "ANXIETY"}[risk]
        score = {"HIGH": 4.0, "MEDIUM": 3.1, "LOW": 2.2}[risk]
        if not has_consult_signal(text) and risk == "LOW":
            emotion, score = "NORMAL", 0.0
        return (
            f'{{"emotion":"{emotion}","emotionScore":{score},"risk":"{risk}",'
            f'"confidence":0.8,"summary":"scripted assessment"}}'
        )

    def _response(self, system: str, last: str) -> str:
        if self.script.responses:
            index = next(self._response_cursor)
            if index < len(self.script.responses):
                return self.script.responses[index]
            # 序列用尽后回落到 response / 默认逻辑
        if self.script.response:
            return self.script.response
        high_risk = "高风险处理规则" in system or has_high_risk_signal(last)
        if high_risk:
            return HIGH_RISK_DEFAULT_RESPONSE
        if "请共情、具体地回应用户" in system:
            return CONSULT_DEFAULT_RESPONSE
        return CHAT_DEFAULT_RESPONSE


class ScriptedModelRegistry(AgentModelRegistry):
    """所有 agent 共用同一个 ScriptedAiClient 配置；metrics_hook 透传以保留 LLM_CALL 指标。"""

    def __init__(self, settings: Settings, script: ModelScript | None = None):
        super().__init__(settings)
        self.script = script
        self._response_cursor = itertools.count()

    def client_for(self, agent_name: str, metrics_hook=None) -> AiClient:
        return ScriptedAiClient(self.settings, self.script, metrics_hook=metrics_hook, response_cursor=self._response_cursor)


def default_eval_settings(keep_db: bool = False, db_path: Path | None = None) -> Settings:
    """离线回放专用配置：mock provider、关向量库、走工具队列（不启动 worker）。"""
    overrides = {
        "database_url": f"sqlite:///{db_path.as_posix()}" if db_path else "sqlite:///:memory:",
        "ai_provider": "mock",
        "agent_model_default_provider": "mock",
        "agent_framework": "event_driven_multi_agent",
        "knowledge_vector_enabled": False,
        "knowledge_vector_required": False,
        "tool_queue_enabled": True,
        "alert_email_delivery_mode": "log",
    }
    return Settings(**overrides)


def install_replay_patches() -> None:
    """把 Redis 记忆替换为进程内实现（幂等，可重复调用）。"""
    global _MEMORY_PATCHED
    import app.agents.event_driven_runtime as runtime_module
    import app.agents.harness as harness_module
    import app.services.memory as memory_module
    from app.harness.runner import InMemoryShortTermMemoryStore

    harness_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    memory_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    runtime_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    _MEMORY_PATCHED = True


def run_case(case: GoldenCase, settings: Settings | None = None, keep_db: bool = False) -> TraceRecord:
    """单 case 独立 InMemory DB 回放最后一轮用户消息，返回解析好的 v2 TraceRecord。"""
    from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService  # noqa: F401  # 确保模块已加载再 patch
    import app.agents.event_driven_runtime as runtime_module
    from app.agents.harness import MindBridgeAgentHarness
    from app.core.database import Base
    from app.harness.runner import InMemoryShortTermMemoryStore

    install_replay_patches()
    InMemoryShortTermMemoryStore.reset()

    db_path: Path | None = None
    if keep_db:
        db_dir = Path("target/agent_eval")
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / f"{case.caseId}.sqlite3"
        for suffix in ["", "-wal", "-shm"]:
            candidate = Path(f"{db_path}{suffix}")
            if candidate.exists():
                candidate.unlink()
    settings = settings or default_eval_settings(keep_db=keep_db, db_path=db_path)

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

    original_registry = runtime_module.AgentModelRegistry
    runtime_module.AgentModelRegistry = lambda s: ScriptedModelRegistry(s, case.modelScript)  # type: ignore[assignment]
    try:
        user = UserAccount(
            username=f"eval_{case.caseId}",
            display_name="评测同学",
            password_hash="x",
            roles_csv="ROLE_USER",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        session = ChatSession(public_id=f"eval-session-{case.caseId}", user_id=user.id, title=case.caseId[:36])
        db.add(session)
        db.commit()
        db.refresh(session)

        harness = MindBridgeAgentHarness(db, settings)
        history_turns = case.conversation[:-1]
        for turn in history_turns:
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
        runtime_module.AgentModelRegistry = original_registry
        db.close()
        engine.dispose()
