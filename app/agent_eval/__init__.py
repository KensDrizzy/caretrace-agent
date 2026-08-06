"""CareTrace 离线评测模块（Trace v2）。

- dataset_loader: golden JSONL 加载与校验
- testkit: 确定性回放（ScriptedAiClient / run_case）
- trace_parser: v2 trace 解析（旧版本抛 TraceVersionError）
- invariant_evaluator: 确定性 Hard Gate
- rubric_evaluator: LLM-as-a-Judge（Mock/AiClient 两种实现）
- runner: CLI 入口
- report: 指标聚合（核心指标只基于 gold）
"""

from __future__ import annotations

from app.agent_eval.dataset_loader import DatasetLoadError, load_dataset
from app.agent_eval.invariant_evaluator import InvariantEvaluator
from app.agent_eval.rubric_evaluator import AiClientRubricJudge, MockRubricJudge, RubricJudge
from app.agent_eval.schemas import (
    CaseEvalResult,
    GoldenCase,
    HardGateResult,
    RubricScore,
    RubricSpec,
    TraceRecord,
)
from app.agent_eval.trace_parser import TraceVersionError, parse_trace

__all__ = [
    "AiClientRubricJudge",
    "CaseEvalResult",
    "DatasetLoadError",
    "GoldenCase",
    "HardGateResult",
    "InvariantEvaluator",
    "MockRubricJudge",
    "RubricJudge",
    "RubricScore",
    "RubricSpec",
    "TraceRecord",
    "TraceVersionError",
    "load_dataset",
    "parse_trace",
]
