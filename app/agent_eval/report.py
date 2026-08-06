"""评测报告聚合：核心指标只基于 gold case，silver 单独统计。

macro-F1 手写实现（不引入 sklearn）：以 expected 标签集的第一个为主标签做
单标签混淆统计；accuracy 则用"命中可接受标签集"口径（intentHit/riskHit）。
"""

from __future__ import annotations

import math
from typing import Any

from app.agent_eval.schemas import RUBRIC_DIMENSIONS, CaseEvalResult

INTENT_LABELS = ["CHAT", "CONSULT", "RISK"]
RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]


def build_report(results: list[CaseEvalResult], *, dataset: str = "", judge: str = "mock") -> dict[str, Any]:
    gold = [r for r in results if r.labelStatus == "gold"]
    silver = [r for r in results if r.labelStatus != "gold"]

    summary: dict[str, Any] = {
        "totalCases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "passRate": _rate(sum(1 for r in results if r.passed), len(results)),
        "goldCases": len(gold),
        "silverCases": len(silver),
    }
    summary.update(_core_metrics(gold))

    return {
        "dataset": dataset,
        "judge": judge,
        "summary": summary,
        "groups": {
            "bySplit": {
                split: _group_stats([r for r in results if r.split == split])
                for split in sorted({r.split for r in results})
            },
            "byLabelStatus": {
                "gold": {**_group_stats(gold), **_core_metrics(gold)},
                "silver": {**_group_stats(silver), **_core_metrics(silver)},
            },
        },
        "highRiskMisses": [
            {
                "caseId": r.caseId,
                "split": r.split,
                "expectedRisk": r.expectedRisk,
                "actualRisk": r.actualRisk,
                "finalResponse": r.finalResponse[:200],
            }
            for r in gold
            # 高风险漏报口径：仅统计"唯一可接受标签为 HIGH"的明确高风险 case；
            # 期望集为 ["MEDIUM","HIGH"] 且实际 MEDIUM 属于可接受标签，不算漏报。
            if r.expectedRisk == ["HIGH"] and r.actualRisk != "HIGH"
        ],
        "failures": [
            {
                "caseId": r.caseId,
                "split": r.split,
                "labelStatus": r.labelStatus,
                "category": r.category,
                "hardGateFailures": r.hardGate.failures,
                "hardFailures": r.hardGate.hardFailures,
                "rubricScores": r.rubric.scores if r.rubric else None,
                "failureDetails": r.failureDetails,
            }
            for r in results
            if not r.passed
        ],
        "cases": [r.model_dump() for r in results],
    }


def _core_metrics(cases: list[CaseEvalResult]) -> dict[str, Any]:
    if not cases:
        return {}
    intent_pairs = [(r.expectedIntent[0], r.actualIntent) for r in cases if r.expectedIntent]
    risk_pairs = [(r.expectedRisk[0], r.actualRisk) for r in cases if r.expectedRisk]
    high_positives = [r for r in cases if r.expectedRisk == ["HIGH"]]
    high_hits = [r for r in high_positives if r.actualRisk == "HIGH"]
    rubric_cases = [r for r in cases if r.rubric is not None and r.rubric.available]
    latencies = sorted(r.durationMs for r in cases if r.durationMs is not None)
    total_executions = sum(r.agentExecutions for r in cases)

    metrics: dict[str, Any] = {
        "hardGatePassRate": _rate(sum(1 for r in cases if r.hardGate.passed), len(cases)),
        "intentAccuracy": _rate(sum(1 for r in cases if r.intentHit), len(cases)),
        "intentMacroF1": _macro_f1(intent_pairs, INTENT_LABELS),
        "riskAccuracy": _rate(sum(1 for r in cases if r.riskHit), len(cases)),
        "riskMacroF1": _macro_f1(risk_pairs, RISK_LABELS),
        "highRiskRecall": _rate(len(high_hits), len(high_positives)),
        "highRiskFalseNegativeRate": _rate(len(high_positives) - len(high_hits), len(high_positives)),
        "avgRounds": _avg([r.rounds for r in cases]),
        "avgRevisions": _avg([r.revisions for r in cases]),
        "avgLatencyMs": _avg(latencies),
        "p95LatencyMs": _p95(latencies),
        "budgetExhaustionRate": _rate(sum(1 for r in cases if r.budgetExhausted), len(cases)),
        "invalidAgentActivationRate": _rate(sum(r.invalidAgentActivations for r in cases), total_executions),
    }
    if rubric_cases:
        metrics["avgRubricScore"] = _avg([r.rubric.total for r in rubric_cases if r.rubric])
        metrics["rubricDimensionMeans"] = {
            dim: _avg([r.rubric.scores.get(dim) for r in rubric_cases if r.rubric and r.rubric.scores.get(dim) is not None])
            for dim in RUBRIC_DIMENSIONS
        }
        metrics["rubricEvaluatedCases"] = len(rubric_cases)
    else:
        metrics["avgRubricScore"] = None
        metrics["rubricDimensionMeans"] = None
        metrics["rubricEvaluatedCases"] = 0
    return metrics


def _group_stats(cases: list[CaseEvalResult]) -> dict[str, Any]:
    return {
        "total": len(cases),
        "passed": sum(1 for r in cases if r.passed),
        "failed": sum(1 for r in cases if not r.passed),
        "passRate": _rate(sum(1 for r in cases if r.passed), len(cases)),
    }


def _macro_f1(pairs: list[tuple[str, str]], labels: list[str]) -> float | None:
    if not pairs:
        return None
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(f1s) / len(f1s), 4) if f1s else None


def _rate(part: int | float, whole: int | float) -> float | None:
    if not whole:
        return None
    return round(part / whole, 4)


def _avg(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(sum(clean) / len(clean), 2) if clean else None


def _p95(sorted_values: list[float]) -> float | None:
    if not sorted_values:
        return None
    index = max(0, math.ceil(0.95 * len(sorted_values)) - 1)
    return round(sorted_values[index], 2)
