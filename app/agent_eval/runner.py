"""离线评测 CLI。

用法：
  python -m app.agent_eval.runner \
    --dataset tests/eval/caretrace_gold.jsonl \
    --output reports/eval_report.json \
    [--split dev|heldout|all] [--limit N] [--judge off|mock|llm] \
    [--ai-provider mock] [--cases id1,id2] [--keep-db]

流程：逐 case 用 testkit.run_case 确定性回放 -> InvariantEvaluator（Hard Gate）
-> 可选 RubricJudge。单个 case 异常不中断整体，记为 failed case。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from app.agent_eval.dataset_loader import DatasetLoadError, load_dataset
from app.agent_eval.invariant_evaluator import InvariantEvaluator, case_counters
from app.agent_eval.report import build_report
from app.agent_eval.rubric_evaluator import AiClientRubricJudge, MockRubricJudge, RubricJudge
from app.agent_eval.schemas import CaseEvalResult, GoldenCase, HardGateResult, RubricScore
from app.agent_eval.testkit import run_case
from app.core.config import Settings


def _build_judge(mode: str, ai_provider: str | None) -> RubricJudge | None:
    if mode == "off":
        return None
    if mode == "mock":
        return MockRubricJudge()
    from app.services.ai import AiClient

    settings = Settings()
    if ai_provider:
        settings.ai_provider = ai_provider
    return AiClientRubricJudge(AiClient(settings))


def evaluate_case(
    case: GoldenCase,
    evaluator: InvariantEvaluator,
    judge: RubricJudge | None,
    keep_db: bool = False,
) -> CaseEvalResult:
    base = {
        "caseId": case.caseId,
        "split": case.split,
        "labelStatus": case.labelStatus,
        "category": case.category,
        "expectedIntent": list(case.expected.intent),
        "expectedRisk": list(case.expected.risk),
    }
    try:
        trace = run_case(case, keep_db=keep_db)
    except Exception as exc:
        return CaseEvalResult(
            **base,
            hardGate=HardGateResult(passed=False, failures=[f"[replay_error] 回放异常: {type(exc).__name__}: {str(exc)[:300]}"]),
            passed=False,
            failureDetails=[f"回放异常: {type(exc).__name__}: {str(exc)[:300]}", traceback.format_exc()[-500:]],
        )

    hard_gate = evaluator.evaluate(case, trace)
    rubric_score: RubricScore | None = None
    if judge is not None and case.rubric is not None:
        try:
            rubric_score = judge.evaluate(case.conversation[-1].content, trace, trace.finalResponse, case.rubric)
        except Exception as exc:
            rubric_score = RubricScore(
                scores={dim: None for dim in ("risk_alignment", "relevance", "empathy_boundary", "actionability", "groundedness", "trajectory_efficiency")},
                judgeErrors=[f"judge 调用异常: {type(exc).__name__}: {str(exc)[:200]}"],
                available=False,
                passed=False,
            )

    intent_hit = trace.intent in case.expected.intent
    risk_hit = trace.riskLevel in case.expected.risk
    counters = case_counters(trace)
    # rubric 分不能抵消 hard failure；rubric 不可用（judgeError）时不阻塞 passed
    passed = hard_gate.passed and (rubric_score is None or not rubric_score.available or rubric_score.passed)
    failure_details = list(hard_gate.failures)
    if not intent_hit:
        failure_details.append(f"intent 未命中: 期望 {case.expected.intent} 实际 {trace.intent}")
    if not risk_hit:
        failure_details.append(f"risk 未命中: 期望 {case.expected.risk} 实际 {trace.riskLevel}")
    if rubric_score is not None and rubric_score.judgeErrors:
        failure_details.extend(rubric_score.judgeErrors)

    return CaseEvalResult(
        **base,
        actualIntent=trace.intent,
        actualRisk=trace.riskLevel,
        intentHit=intent_hit,
        riskHit=risk_hit,
        hardGate=hard_gate,
        rubric=rubric_score,
        passed=passed,
        failureDetails=failure_details,
        finalResponse=trace.finalResponse,
        **counters,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CareTrace agent 离线评测（Trace v2）")
    parser.add_argument("--dataset", required=True, help="golden 数据集 JSONL 路径")
    parser.add_argument("--output", required=True, help="报告 JSON 输出路径")
    parser.add_argument("--split", choices=["dev", "heldout", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", choices=["off", "mock", "llm"], default="mock")
    parser.add_argument("--ai-provider", default=None, help="--judge llm 时的 provider（默认读 settings/env）")
    parser.add_argument("--cases", default=None, help="只跑指定 caseId（逗号分隔）")
    parser.add_argument("--keep-db", action="store_true", help="保留每个 case 的 sqlite 回放库到 target/agent_eval/")
    args = parser.parse_args(argv)

    try:
        cases = load_dataset(args.dataset)
    except DatasetLoadError as exc:
        print(f"数据集加载失败: {exc}", file=sys.stderr)
        return 2

    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
    if args.cases:
        wanted = {item.strip() for item in args.cases.split(",") if item.strip()}
        cases = [case for case in cases if case.caseId in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print("没有匹配的 case", file=sys.stderr)
        return 2

    judge = _build_judge(args.judge, args.ai_provider)
    evaluator = InvariantEvaluator()
    results: list[CaseEvalResult] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] 回放 {case.caseId} ...", file=sys.stderr, flush=True)
        result = evaluate_case(case, evaluator, judge, keep_db=args.keep_db)
        status = "PASS" if result.passed else "FAIL"
        print(f"    -> {status} intent={result.actualIntent} risk={result.actualRisk}", file=sys.stderr, flush=True)
        results.append(result)

    report = build_report(results, dataset=str(args.dataset), judge=args.judge)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary = report["summary"]
    print(f"\n评测完成: {summary['passed']}/{summary['totalCases']} 通过（passRate={summary['passRate']}）")
    for key in ("hardGatePassRate", "intentAccuracy", "intentMacroF1", "riskAccuracy", "riskMacroF1",
                "highRiskRecall", "highRiskFalseNegativeRate", "avgRubricScore", "avgRounds",
                "avgLatencyMs", "p95LatencyMs", "budgetExhaustionRate", "invalidAgentActivationRate"):
        if key in summary:
            print(f"  {key}: {summary[key]}")
    print(f"报告已写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
