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
    started = _events(consult_trace, "TOOL_EXECUTION_STARTED")
    finished = [
        event
        for event in consult_trace.events
        if event.eventType in {"TOOL_EXECUTION_COMPLETED", "TOOL_EXECUTION_FAILED"}
    ]
    assert started, "CONSULT 回放（report + tool queue）应产生 TOOL_EXECUTION_STARTED"
    assert finished, "应产生 TOOL_EXECUTION_COMPLETED/FAILED"
    for event in [*started, *finished]:
        metadata = event.metadata or {}
        assert metadata.get("idempotencyKey"), f"{event.eventType} 必须带 idempotencyKey"
        assert metadata.get("kind"), f"{event.eventType} 必须带工具 kind"
    # queue 路径的幂等键形如 {kind}:{reportId}
    keys = {event.metadata["idempotencyKey"] for event in finished}
    assert all(":" in key for key in keys)


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


def test_tool_execution_failed_event_metadata_structure(monkeypatch):
    """工具入队失败：TOOL_EXECUTION_FAILED 事件 metadata 含 errorType / retryCount / idempotencyKey。"""

    def _raising_enqueue(self, report_id, risk_level):
        raise RuntimeError("tool queue backend down")

    monkeypatch.setattr(ToolQueueService, "enqueue_report", _raising_enqueue)
    trace = run_case(make_case("t2e-tool-failure", CONSULT_TEXT))

    failed = _events(trace, "TOOL_EXECUTION_FAILED")
    assert failed, "enqueue 失败应产生 TOOL_EXECUTION_FAILED 事件"
    metadata = failed[0].metadata or {}
    assert metadata.get("errorType") == "RuntimeError"
    assert "retryCount" in metadata and isinstance(metadata["retryCount"], int)
    assert metadata.get("idempotencyKey")
    assert metadata.get("status") == "FAILED"
    # 工具失败不中断主流程
    assert trace.status == "COMPLETED"
    assert "TRACE_COMPLETED" in trace.event_types()


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
