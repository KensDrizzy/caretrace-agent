#!/usr/bin/env python3
"""对比两份 live 消融报告（A 基线 vs B 实验组）。

用法：
  python scripts/ablation_diff.py reports/ablation_A_all_api.json reports/ablation_B_lora_classify.json
  # 额外输出 markdown：
  python scripts/ablation_diff.py A.json B.json --output reports/ablation_diff.md

对比内容：
  1) summary 核心指标逐项差值；
  2) 按 category 的 risk 命中差异；
  3) 配对翻转：A对B错 / A错B对 的具体 case（risk 维度）；
  4) McNemar 精确检验（risk/intent/passed 三个维度），判断差异是否显著。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_cases(report: dict) -> dict[str, dict]:
    return {c["caseId"]: c for c in report.get("cases", [])}


METRIC_KEYS = [
    "passRate", "hardGatePassRate", "intentAccuracy", "intentMacroF1",
    "riskAccuracy", "riskMacroF1", "highRiskRecall", "highRiskFalseNegativeRate",
    "avgRubricScore", "avgRounds", "avgRevisions", "avgLatencyMs", "p95LatencyMs",
    "budgetExhaustionRate", "invalidAgentActivationRate",
]


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def diff_metrics(a: dict, b: dict) -> list[tuple[str, str, str, str]]:
    sa, sb = a.get("summary", {}), b.get("summary", {})
    rows = []
    for key in METRIC_KEYS:
        if key not in sa and key not in sb:
            continue
        va, vb = sa.get(key), sb.get(key)
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = vb - va
            delta = f"{d:+.4f}"
        rows.append((key, fmt(va), fmt(vb), delta))
    return rows


def paired_flips(ia: dict, ib: dict, field: str) -> tuple[int, int, list, list]:
    """返回 (A对B错数, A错B对数, A对B错case列表, A错B对case列表)。"""
    a_only_wrong = []  # B 对、A 错（实验组救回）
    b_only_wrong = []  # A 对、B 错（实验组引入的退化）
    for cid, ca in ia.items():
        cb = ib.get(cid)
        if cb is None:
            continue
        ha, hb = bool(ca.get(field)), bool(cb.get(field))
        if ha and not hb:
            b_only_wrong.append((cid, ca.get("category"), ca.get("actualRisk"), cb.get("actualRisk")))
        elif not ha and hb:
            a_only_wrong.append((cid, ca.get("category"), ca.get("actualRisk"), cb.get("actualRisk")))
    return len(b_only_wrong), len(a_only_wrong), b_only_wrong, a_only_wrong


def mcnemar_exact(b: int, c: int) -> float | None:
    """McNemar 双侧精确 p 值（二项检验），b/c 为不一致对数量。"""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def category_risk(ia: dict) -> dict[str, tuple[int, int]]:
    stats: dict[str, list[int]] = {}
    for c in ia.values():
        cat = c.get("category") or "(none)"
        hit, total = stats.setdefault(cat, [0, 0])
        stats[cat] = [hit + int(bool(c.get("riskHit"))), total + 1]
    return {k: tuple(v) for k, v in stats.items()}


def build_markdown(a: dict, b: dict) -> str:
    ia, ib = index_cases(a), index_cases(b)
    lines = []
    pa, pb = a.get("ablationProfile", "A"), b.get("ablationProfile", "B")
    lines.append(f"# 消融对比：{pa}（基线） vs {pb}（实验组）\n")
    lines.append(f"- A 报告：{a.get('dataset','')}，{a['summary'].get('totalCases')} 条")
    lines.append(f"- B 报告：{b.get('dataset','')}，{b['summary'].get('totalCases')} 条\n")

    lines.append("## 1. 核心指标差值（B - A）\n")
    lines.append("| 指标 | A 基线 | B 实验组 | 差值 |")
    lines.append("|---|---|---|---|")
    for key, va, vb, delta in diff_metrics(a, b):
        lines.append(f"| {key} | {va} | {vb} | {delta} |")

    lines.append("\n## 2. 配对翻转与 McNemar 检验\n")
    lines.append("| 维度 | A对B错(退化) | A错B对(救回) | McNemar p |")
    lines.append("|---|---|---|---|")
    for field, name in [("riskHit", "风险分类"), ("intentHit", "意图分类"), ("passed", "整体通过")]:
        bw, aw, _, _ = paired_flips(ia, ib, field)
        p = mcnemar_exact(bw, aw)
        lines.append(f"| {name} | {bw} | {aw} | {f'{p:.4f}' if p is not None else '-'} |")
    lines.append("\n> p<0.05 表示差异在配对样本上统计显著；不一致对太少时检验不具说服力。")

    lines.append("\n## 3. 各 category 风险命中率（A → B）\n")
    ca, cb = category_risk(ia), category_risk(ib)
    lines.append("| category | A 命中 | B 命中 |")
    lines.append("|---|---|---|")
    for cat in sorted(set(ca) | set(cb)):
        ha, ta = ca.get(cat, (0, 0))
        hb, tb = cb.get(cat, (0, 0))
        lines.append(f"| {cat} | {ha}/{ta} | {hb}/{tb} |")

    for field, title in [("riskHit", "风险分类"), ("intentHit", "意图分类")]:
        bw, aw, bw_list, aw_list = paired_flips(ia, ib, field)
        lines.append(f"\n## 4. {title}翻转明细\n")
        lines.append(f"### B 救回（A 错 → B 对），共 {aw} 条")
        for cid, cat, ar, br in aw_list:
            lines.append(f"- `{cid}` ({cat})：A={ar} → B={br}")
        lines.append(f"\n### B 退化（A 对 → B 错），共 {bw} 条")
        for cid, cat, ar, br in bw_list:
            lines.append(f"- `{cid}` ({cat})：A={ar} → B={br}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_a")
    parser.add_argument("report_b")
    parser.add_argument("--output", default=None, help="可选：输出 markdown 路径")
    args = parser.parse_args()

    a, b = load(args.report_a), load(args.report_b)
    ia, ib = index_cases(a), index_cases(b)

    print(f"=== 核心指标（{a.get('ablationProfile','A')} → {b.get('ablationProfile','B')}，差值=B-A）===")
    print(f"{'指标':32s}{'A':>12s}{'B':>12s}{'delta':>12s}")
    for key, va, vb, delta in diff_metrics(a, b):
        print(f"{key:32s}{va:>12s}{vb:>12s}{delta:>12s}")

    print("\n=== 配对翻转 / McNemar ===")
    for field, name in [("riskHit", "风险分类"), ("intentHit", "意图分类"), ("passed", "整体通过")]:
        bw, aw, _, _ = paired_flips(ia, ib, field)
        p = mcnemar_exact(bw, aw)
        pstr = f"{p:.4f}" if p is not None else "-"
        print(f"{name:8s} 退化(A对B错)={bw}  救回(A错B对)={aw}  McNemar p={pstr}")

    if args.output:
        md = build_markdown(a, b)
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"\nMarkdown 对比报告已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
