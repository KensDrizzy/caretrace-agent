"""离线评测（agent_eval）pytest 套件（任务场景 9、10 + trace_parser 向后兼容）。

回放全部走 app.agent_eval.testkit 的 InMemory 环境；不变量规则测试直接构造
v2 trace dict 过 parse_trace + InvariantEvaluator，不触真实模型 / Redis / Chroma / MCP。
"""

from __future__ import annotations

import pytest

from app.agent_eval.invariant_evaluator import InvariantEvaluator
from app.agent_eval.schemas import ConversationTurn, ExpectedSpec, GoldenCase, ModelScript
from app.agent_eval.testkit import run_case
from app.agent_eval.trace_parser import TraceVersionError, parse_trace
from app.services.tool_queue import ToolQueueService

CHAT_TEXT = "你好，今天天气怎么样，适合出去走走吗"
CONSULT_TEXT = "最近压力很大，睡不着，感觉快撑不住了"


def make_case(case_id: str, text: str, *, script: ModelScript | None = None) -> GoldenCase:
    return GoldenCase(
        caseId=case_id,
        source="synthetic",
        labelStatus="gold",
        conversation=[ConversationTurn(role="user", content=text)],
        expected=ExpectedSpec(intent=["CHAT"], risk=["LOW"]),
        modelScript=script or ModelScript(),
    )


def _events(trace, event_type: str):
    return [event for event in trace.events if event.eventType == event_type]


# ---------------------------------------------------------------------------
# 场景 9：Agent 拒绝认领的决策也进入 Trace
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chat_trace():
    return run_case(make_case("t2e-chat-decisions", CHAT_TEXT))


def test_declined_claim_decisions_are_traced(chat_trace):
    decisions = _events(chat_trace, "DECISION_EVALUATED")
    assert decisions, "回放应产生 DECISION_EVALUATED 事件"

    declined = [event for event in decisions if (event.metadata or {}).get("claim") is False]
    assert declined, "应存在 claim=false 的拒绝认领记录"
    for event in declined:
        metadata = event.metadata or {}
        assert metadata.get("reasonCode"), "拒绝认领必须带 reasonCode"
        assert metadata.get("reason"), "拒绝认领必须带可读 reason"
        assert "selected" in metadata, "决策事件必须带 selected 字段"

    # 有拒绝也应有被选中执行的决策，selected 与 claim 语义一致
    assert any((event.metadata or {}).get("selected") is True for event in decisions)
    assert all(
        (event.metadata or {}).get("selected") is not True
        for event in declined
    ), "被拒绝的决策不应被标记为 selected"


# ---------------------------------------------------------------------------
# 场景 10：工具执行 / 重试 / 幂等进入 Trace
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def consult_trace():
    return run_case(make_case("t2e-consult-tools", CONSULT_TEXT))


def test_tool_queue_events_carry_idempotency_key(consult_trace):
    queued = _events(consult_trace, "TOOL_EXECUTION_QUEUED")
    assert queued, "CONSULT 回放应记录持久化入队"
    for event in queued:
        metadata = event.metadata or {}
        assert metadata.get("idempotencyKey")
        assert metadata.get("kind")
        assert metadata.get("jobId")
        assert metadata.get("status") == "QUEUED"
    assert not _events(consult_trace, "TOOL_EXECUTION_STARTED")
    assert not _events(consult_trace, "TOOL_EXECUTION_COMPLETED")
    assert not _events(consult_trace, "TOOL_EXECUTION_FAILED")


def test_no_duplicate_idempotent_tool_rule_fails_on_duplicate_key():
    """构造含重复 idempotencyKey 的 v2 trace dict → no_duplicate_idempotent_tool 失败。"""
    raw = {
        "traceVersion": "2.0",
        "events": [
            {"eventType": "TOOL_EXECUTION_COMPLETED", "actor": "ToolDispatcher", "metadata": {"idempotencyKey": "excel_report:7"}},
            {"eventType": "TOOL_EXECUTION_COMPLETED", "actor": "ToolDispatcher", "metadata": {"idempotencyKey": "excel_report:7"}},
        ],
    }
    trace = parse_trace(raw)
    result = InvariantEvaluator().evaluate(make_case("unit-dup-tool", "x"), trace)

    assert not result.passed
    assert any("no_duplicate_idempotent_tool" in failure for failure in result.failures)


def test_enqueue_failure_rolls_back_report_and_records_failed_trace(monkeypatch):
    """报告与任务原子提交；入队失败不得留下无任务的报告或成功轨迹。"""
    import json
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.models.entities import AgentRunTrace, PsychologicalReport, ToolJob

    def _raising_enqueue(self, report_id, risk_level, **kwargs):
        assert kwargs["commit"] is False
        raise RuntimeError("tool queue backend down")

    monkeypatch.setattr(ToolQueueService, "enqueue_report", _raising_enqueue)
    with pytest.raises(RuntimeError, match="tool queue backend down"):
        run_case(make_case("t2e-tool-failure", CONSULT_TEXT), keep_db=True)

    engine = create_engine("sqlite:///target/agent_eval/t2e-tool-failure.sqlite3")
    try:
        with Session(engine) as db:
            assert db.query(PsychologicalReport).count() == 0
            assert db.query(ToolJob).count() == 0
            trace = db.query(AgentRunTrace).one()
            assert trace.status == "FAILED"
            assert json.loads(trace.error_json)["type"] == "RuntimeError"
    finally:
        engine.dispose()


def test_replay_restores_memory_implementation_after_success_and_failure():
    import app.services.memory as memory_module
    from app.agent_eval.testkit import install_replay_patches
    from app.harness.runner import InMemoryShortTermMemoryStore

    original = memory_module.RedisShortTermMemoryStore
    with pytest.raises(RuntimeError, match="abort replay"):
        with install_replay_patches():
            assert memory_module.RedisShortTermMemoryStore is InMemoryShortTermMemoryStore
            raise RuntimeError("abort replay")
    assert memory_module.RedisShortTermMemoryStore is original
    run_case(make_case("t2e-restore-memory", CHAT_TEXT))
    assert memory_module.RedisShortTermMemoryStore is original


# ---------------------------------------------------------------------------
# trace_parser 向后兼容：旧版本 dict 必须抛 TraceVersionError（含“迁移”提示）
# ---------------------------------------------------------------------------


def test_parse_trace_rejects_missing_trace_version():
    with pytest.raises(TraceVersionError, match="迁移"):
        parse_trace({"events": [], "artifacts": []})


def test_parse_trace_rejects_v1_trace():
    with pytest.raises(TraceVersionError, match="迁移"):
        parse_trace({"traceVersion": "1.0", "events": []})


def test_parse_trace_accepts_v2_trace():
    trace = parse_trace({"traceVersion": "2.0", "events": [{"eventType": "TRACE_COMPLETED"}]})
    assert trace.traceVersion == "2.0"
    assert trace.event_types() == ["TRACE_COMPLETED"]
