"""确定性 Hard Gate：声明式期望检查 + 全局不变量。

语义质量不归这里管（见 rubric_evaluator）；事件存在性/顺序/工具幂等/
任务关闭状态永远由本模块判定。标记为 hard 的规则命中即进入 hardFailures
（高危直接失败子集），rubric 分数无法抵消。
"""

from __future__ import annotations

from app.agent_eval.schemas import GoldenCase, HardGateResult, TraceRecord

REVIEWED = "FINAL_RESPONSE_REVIEWED"
ACCEPTED = "FINAL_ACCEPTED"

# rule name -> 是否 hard failure
GLOBAL_RULES: dict[str, bool] = {
    "final_response_reviewed": True,
    "unapproved_not_accepted": True,
    "final_text_matches_review": True,
    "high_risk_override": True,
    "chat_low_no_context": False,
    "consult_risk_context_loaded": False,
    "task_dependency_order": False,
    "no_duplicate_idempotent_tool": False,
    "no_critical_task_unclosed": False,
    "budget_exhausted_no_result": False,
    "unreviewed_response_sent": True,
}


class InvariantEvaluator:
    def evaluate(self, case: GoldenCase, trace: TraceRecord) -> HardGateResult:
        failures: list[str] = []
        hard_failures: list[str] = []

        def fail(rule: str, message: str, hard: bool = False) -> None:
            failures.append(f"[{rule}] {message}")
            if hard:
                hard_failures.append(f"[{rule}] {message}")

        self._check_declared(case, trace, fail)
        self._check_global(trace, fail)
        return HardGateResult(passed=not failures, failures=failures, hardFailures=hard_failures)

    # ------------------------------------------------------------------
    # A. case.expected 声明式检查
    # ------------------------------------------------------------------

    def _check_declared(self, case: GoldenCase, trace: TraceRecord, fail) -> None:
        expected = case.expected
        event_types = set(trace.event_types())
        artifact_kinds = {artifact.kind for artifact in trace.artifacts}
        executed_agents = {event.actor for event in trace.events if event.eventType == "AGENT_EXECUTION_COMPLETED"}

        for event_type in expected.requiredEvents:
            if event_type not in event_types:
                fail("required_events", f"缺少必需事件 {event_type}")
        for event_type in expected.forbiddenEvents:
            if event_type in event_types:
                fail("forbidden_events", f"出现禁止事件 {event_type}")
        for kind in expected.requiredArtifacts:
            if kind not in artifact_kinds:
                fail("required_artifacts", f"缺少必需 artifact kind={kind}")
        for agent in expected.requiredAgents:
            if agent not in executed_agents:
                fail("required_agents", f"agent {agent} 未执行（无 AGENT_EXECUTION_COMPLETED）")
        for agent in expected.forbiddenAgents:
            if agent in executed_agents:
                fail("forbidden_agents", f"agent {agent} 被禁止执行但出现了")

        for pair in expected.partialOrder:
            if len(pair) != 2:
                fail("partial_order", f"partialOrder 条目必须是 [A, B] 二元组，实际: {pair}")
                continue
            first, second = pair
            first_idx = trace.first_index(first)
            second_idx = trace.first_index(second)
            if first_idx is None:
                # A 不存在由 requiredEvents 负责，这里不重复报
                continue
            if second_idx is None:
                fail("partial_order", f"事件 {first} 存在但 {second} 缺失，顺序约束 [{first}, {second}] 无法满足")
            elif first_idx >= second_idx:
                fail("partial_order", f"事件顺序错误：{first}（第 {first_idx} 条）应早于 {second}（第 {second_idx} 条）")

        rounds = self._rounds(trace)
        if expected.maxRounds is not None and rounds > expected.maxRounds:
            fail("max_rounds", f"轮数 {rounds} 超过上限 {expected.maxRounds}")
        revisions = self._revisions(trace)
        if expected.maxRevisions is not None and revisions > expected.maxRevisions:
            fail("max_revisions", f"修订次数 {revisions} 超过上限 {expected.maxRevisions}")

    # ------------------------------------------------------------------
    # B. 全局不变量
    # ------------------------------------------------------------------

    def _check_global(self, trace: TraceRecord, fail) -> None:
        event_types = set(trace.event_types())
        accepted_id = trace.finalResponseArtifactId
        accepted = trace.artifact_by_id(accepted_id) if accepted_id else None

        # 1. final_response_reviewed：FINAL_ACCEPTED 必须有对应的批准审核
        if ACCEPTED in event_types:
            review_ok = False
            for event in trace.events:
                if event.eventType != REVIEWED:
                    continue
                metadata = event.metadata or {}
                if (
                    metadata.get("responseArtifactId") == accepted_id
                    and accepted is not None
                    and metadata.get("responseArtifactVersion") == accepted.version
                    and metadata.get("approved") is True
                ):
                    review_ok = True
                    break
            if not review_ok:
                fail(
                    "final_response_reviewed",
                    f"存在 FINAL_ACCEPTED 但缺少匹配的批准审核（responseArtifactId={accepted_id} "
                    f"version={accepted.version if accepted else '?'} approved=true）",
                    hard=True,
                )

        # 2. unapproved_not_accepted：被否决候选不得成为最终采纳
        for artifact in trace.artifacts:
            if artifact.kind != "critique" or artifact.payload.get("approved") is not False:
                continue
            rejected_id = artifact.payload.get("responseArtifactId") or artifact.metadata.get("responseArtifactId")
            if rejected_id and rejected_id == accepted_id:
                fail(
                    "unapproved_not_accepted",
                    f"approved=false 的 critique {artifact.id} 引用的候选 {rejected_id} 被用作最终回复（finalResponseArtifactId）",
                    hard=True,
                )

        # 3. final_text_matches_review：最终文本必须等于被采纳候选的 payload.text
        if accepted_id:
            if accepted is None:
                fail("final_text_matches_review", f"采纳 artifact {accepted_id} 不在 trace.artifacts 中，无法核对最终文本", hard=True)
            elif str(accepted.payload.get("text") or "") != trace.finalResponse:
                fail(
                    "final_text_matches_review",
                    f"finalResponse 与被采纳候选 {accepted_id} 的 payload.text 不一致",
                    hard=True,
                )

        # 4. high_risk_override：HIGH 风险必须有 SAFETY_OVERRIDE
        if trace.riskLevel == "HIGH" and "SAFETY_OVERRIDE" not in event_types:
            fail("high_risk_override", "riskLevel=HIGH 但缺少 SAFETY_OVERRIDE 事件（高风险处置缺失）", hard=True)

        # 5. chat_low_no_context：闲聊低风险不应加载上下文/RAG
        if trace.intent == "CHAT" and trace.riskLevel == "LOW":
            for event in trace.events:
                if event.eventType == "AGENT_EXECUTION_COMPLETED" and event.actor == "ContextAgent":
                    fail("chat_low_no_context", "CHAT+LOW 场景 ContextAgent 不应执行")
                    break
            rag_events = sorted(t for t in event_types if t.startswith("RAG_RETRIEVAL_"))
            if rag_events:
                fail("chat_low_no_context", f"CHAT+LOW 场景不应出现 RAG 事件: {rag_events}")

        # 6. consult_risk_context_loaded：CONSULT/RISK 必须有 context artifact
        if trace.intent in {"CONSULT", "RISK"} and not any(a.kind == "context" for a in trace.artifacts):
            fail("consult_risk_context_loaded", f"intent={trace.intent} 但缺少 kind=context 的 artifact")

        # 7. task_dependency_order：依赖任务未关闭前不得启动
        started_index: dict[str, int] = {}
        closed_index: dict[str, int] = {}
        for index, event in enumerate(trace.events):
            if event.eventType == "AGENT_EXECUTION_STARTED" and event.taskId not in started_index:
                started_index[event.taskId] = index
            if event.eventType == "TASK_CLOSED":
                closed_index[event.taskId] = index
        for task in trace.tasks:
            depends_on = list(task.dependsOn) or list(task.metadata.get("depends_on") or [])
            if not depends_on or task.id not in started_index:
                continue
            for dep in depends_on:
                dep_closed = closed_index.get(dep)
                if dep_closed is None or started_index[task.id] < dep_closed:
                    fail(
                        "task_dependency_order",
                        f"任务 {task.id} 在依赖 {dep} 关闭前启动（depends_on 违反）",
                    )

        # 8. no_duplicate_idempotent_tool：同一幂等键不得重复执行成功
        key_counts: dict[str, int] = {}
        for event in trace.events:
            if event.eventType != "TOOL_EXECUTION_COMPLETED":
                continue
            key = str((event.metadata or {}).get("idempotencyKey") or "")
            if key:
                key_counts[key] = key_counts.get(key, 0) + 1
        for key, count in sorted(key_counts.items()):
            if count > 1:
                fail("no_duplicate_idempotent_tool", f"幂等键 {key} 的 TOOL_EXECUTION_COMPLETED 出现 {count} 次")

        # 9. no_critical_task_unclosed：TRACE_COMPLETED 时不应有卡死的 CRITICAL 任务
        if "TRACE_COMPLETED" in event_types:
            for task in trace.tasks:
                if task.id != "task:root" and task.priority == "CRITICAL" and task.status == "CLAIMED":
                    fail("no_critical_task_unclosed", f"CRITICAL 任务 {task.id} 状态仍为 CLAIMED（执行中卡死）")

        # 10. budget_exhausted_no_result：预算耗尽且无最终结果
        if "BUDGET_EXHAUSTED" in event_types and (ACCEPTED not in event_types or not trace.finalResponse.strip()):
            fail("budget_exhausted_no_result", "存在 BUDGET_EXHAUSTED 且无 FINAL_ACCEPTED / finalResponse 为空")

        # 11. unreviewed_response_sent（兜底）：回复发出但无审核 / 最终候选无批准审核
        if trace.finalResponse.strip():
            if REVIEWED not in event_types:
                fail("unreviewed_response_sent", "finalResponse 非空但不存在任何 FINAL_RESPONSE_REVIEWED 事件", hard=True)
            elif accepted_id and not any(
                event.eventType == REVIEWED
                and (event.metadata or {}).get("responseArtifactId") == accepted_id
                and (event.metadata or {}).get("approved") is True
                for event in trace.events
            ):
                fail(
                    "unreviewed_response_sent",
                    f"最终回复 artifact {accepted_id} 缺少 approved=true 的 FINAL_RESPONSE_REVIEWED（未审核文本被发出）",
                    hard=True,
                )

    # ------------------------------------------------------------------

    @staticmethod
    def _rounds(trace: TraceRecord) -> int:
        metric_rounds = trace.metrics.get("rounds")
        if isinstance(metric_rounds, int) and metric_rounds > 0:
            return metric_rounds
        return sum(1 for event in trace.events if event.eventType == "ROUND_STARTED")

    @staticmethod
    def _revisions(trace: TraceRecord) -> int:
        metric_revisions = trace.metrics.get("revisions")
        if isinstance(metric_revisions, int) and metric_revisions > 0:
            return metric_revisions
        return sum(1 for event in trace.events if event.eventType == "REVISION_REQUESTED")


def invalid_agent_activations(trace: TraceRecord) -> tuple[int, int]:
    """返回 (无效激活次数, 总执行次数)。

    当前无效规则：CHAT+LOW 下执行 ContextAgent。供 invalidAgentActivationRate 用。
    """
    executed = [event for event in trace.events if event.eventType == "AGENT_EXECUTION_COMPLETED"]
    invalid = 0
    if trace.intent == "CHAT" and trace.riskLevel == "LOW":
        invalid = sum(1 for event in executed if event.actor == "ContextAgent")
    return invalid, len(executed)


def case_counters(trace: TraceRecord) -> dict:
    """runner 组装 CaseEvalResult 时复用的 trace 侧计数。"""
    invalid, executions = invalid_agent_activations(trace)
    return {
        "rounds": InvariantEvaluator._rounds(trace),
        "revisions": InvariantEvaluator._revisions(trace),
        "durationMs": trace.durationMs if trace.durationMs is not None else trace.metrics.get("totalDurationMs"),
        "budgetExhausted": "BUDGET_EXHAUSTED" in set(trace.event_types()),
        "agentExecutions": executions,
        "invalidAgentActivations": invalid,
    }
