"""Live 消融评测 CLI：用真实模型跑 gold case，产出与离线 runner 同构的报告。

用法：
  # 基线 A：所有 Agent 走 DeepSeek API
  python -m app.agent_eval.live_runner \
    --dataset tests/eval/caretrace_hard.jsonl --profile A \
    --output reports/ablation_A_all_api.json --judge off

  # 实验组 B：情绪分类走本地 LoRA，其余走 DeepSeek
  python -m app.agent_eval.live_runner \
    --dataset tests/eval/caretrace_hard.jsonl --profile B \
    --output reports/ablation_B_lora_classify.json --judge off

  # 对比两份报告：
  python scripts/ablation_diff.py reports/ablation_A_all_api.json reports/ablation_B_lora_classify.json

注意：profile=B 需要本地 Ollama 已加载 FINETUNED_MODEL_NAME 指定的模型。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.agent_eval.dataset_loader import DatasetLoadError, load_dataset
from app.agent_eval.invariant_evaluator import InvariantEvaluator
from app.agent_eval.live_testkit import live_eval_settings, resolved_routing_table, run_case_live
from app.agent_eval.report import build_report
from app.agent_eval.runner import _build_judge, evaluate_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CareTrace live 模型消融评测（真实 provider）")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=["A", "B", "env"], default="B", help="A=全API基线 B=本地分类+API env=按.env")
    parser.add_argument("--split", choices=["dev", "heldout", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cases", default=None, help="只跑指定 caseId（逗号分隔）")
    parser.add_argument("--judge", choices=["off", "mock", "llm"], default="off", help="rubric 打分，live 消融建议固定用 llm 且对 A/B 一致")
    parser.add_argument("--judge-provider", default=None, help="--judge llm 时的 judge provider")
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args(argv)

    try:
        cases = load_dataset(args.dataset)
    except DatasetLoadError as exc:
        print(f"数据集加载失败: {exc}", file=sys.stderr)
        return 2

    if args.split != "all":
        cases = [c for c in cases if c.split == args.split]
    if args.cases:
        wanted = {x.strip() for x in args.cases.split(",") if x.strip()}
        cases = [c for c in cases if c.caseId in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print("没有匹配的 case", file=sys.stderr)
        return 2

    settings = live_eval_settings(profile=args.profile, keep_db=args.keep_db)
    routing = resolved_routing_table(settings)
    print(f"[profile={args.profile}] 模型路由：", file=sys.stderr)
    for alias, resolved in routing.items():
        print(f"  {alias:12s} -> {resolved['provider']:8s} {resolved['model']}", file=sys.stderr)

    judge = _build_judge(args.judge, args.judge_provider)
    evaluator = InvariantEvaluator()
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] live 回放 {case.caseId} ...", file=sys.stderr, flush=True)
        result = evaluate_case(
            case,
            evaluator,
            judge,
            keep_db=args.keep_db,
            run_fn=lambda c: run_case_live(c, settings, keep_db=args.keep_db),
        )
        status = "PASS" if result.passed else "FAIL"
        print(f"    -> {status} intent={result.actualIntent} risk={result.actualRisk}", file=sys.stderr, flush=True)
        results.append(result)

    report = build_report(results, dataset=str(args.dataset), judge=args.judge)
    report["ablationProfile"] = args.profile
    report["modelRouting"] = routing
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary = report["summary"]
    print(f"\n[profile={args.profile}] 评测完成: {summary['passed']}/{summary['totalCases']} 通过（passRate={summary['passRate']}）")
    for key in ("hardGatePassRate", "intentAccuracy", "intentMacroF1", "riskAccuracy", "riskMacroF1",
                "highRiskRecall", "highRiskFalseNegativeRate", "avgRubricScore", "avgRounds",
                "avgLatencyMs", "p95LatencyMs", "budgetExhaustionRate"):
        if key in summary:
            print(f"  {key}: {summary[key]}")
    print(f"报告已写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
