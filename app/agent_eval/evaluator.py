from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.agent_eval.judge import LLMResponseJudge
from app.agent_eval.rubric import AgenticRubric, EvaluationScores
from app.core.config import Settings
from app.models.entities import AgentRunTrace
from app.services.ai import AiClient

logger = logging.getLogger(__name__)


@dataclass
class EvaluatedTrace:
    trace_id: int
    session_id: int
    user_id: int
    original_input: str
    intent: str
    risk_level: str
    scores: EvaluationScores


@dataclass
class EvaluationReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    average_scores: dict[str, float] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "average_scores": self.average_scores,
            "results": self.results,
        }


class AgentTraceEvaluator:
    def __init__(self, settings: Settings, db: Session, use_llm: bool = False):
        self.settings = settings
        self.db = db
        llm_judge = LLMResponseJudge(AiClient(settings)) if use_llm else None
        self.rubric = AgenticRubric(use_llm=use_llm, llm_judge=llm_judge)

    def evaluate_latest(self, limit: int = 100) -> EvaluationReport:
        traces = (
            self.db.query(AgentRunTrace)
            .order_by(AgentRunTrace.created_at.desc())
            .limit(limit)
            .all()
        )
        return self._evaluate_traces(traces)

    def evaluate_by_session(self, session_id: int) -> EvaluationReport:
        traces = (
            self.db.query(AgentRunTrace)
            .filter(AgentRunTrace.session_id == session_id)
            .order_by(AgentRunTrace.created_at.desc())
            .all()
        )
        return self._evaluate_traces(traces)

    def evaluate_dataset(self, dataset: list[dict[str, Any]]) -> EvaluationReport:
        """Evaluate traces provided as a JSON dataset (no DB required)."""
        report = EvaluationReport()
        for item in dataset:
            trace = item.get("trace", item)
            ground_truth = item.get("ground_truth", {})
            evaluated = self._evaluate_one(trace, ground_truth)
            report.results.append(self._result_dict(evaluated))
            report.total += 1
            if evaluated.scores.passed:
                report.passed += 1
            else:
                report.failed += 1
        report.average_scores = self._compute_averages(report.results)
        return report

    def _evaluate_traces(self, traces: list[AgentRunTrace]) -> EvaluationReport:
        report = EvaluationReport()
        for trace in traces:
            trace_dict = self._trace_to_dict(trace)
            evaluated = self._evaluate_one(trace_dict, {})
            report.results.append(self._result_dict(evaluated))
            report.total += 1
            if evaluated.scores.passed:
                report.passed += 1
            else:
                report.failed += 1
        report.average_scores = self._compute_averages(report.results)
        return report

    def _evaluate_one(self, trace: dict[str, Any], ground_truth: dict[str, Any]) -> EvaluatedTrace:
        scores = self.rubric.evaluate(trace, ground_truth)
        return EvaluatedTrace(
            trace_id=trace.get("id", 0),
            session_id=trace.get("session_id", 0),
            user_id=trace.get("user_id", 0),
            original_input=trace.get("original_input", ""),
            intent=trace.get("intent", "CHAT"),
            risk_level=trace.get("risk_level", "LOW"),
            scores=scores,
        )

    @staticmethod
    def _trace_to_dict(trace: AgentRunTrace) -> dict[str, Any]:
        return {
            "id": trace.id,
            "user_id": trace.user_id,
            "session_id": trace.session_id,
            "report_id": trace.report_id,
            "intent": trace.intent,
            "risk_level": trace.risk_level,
            "original_input": trace.original_input,
            "sanitized_input": trace.sanitized_input,
            "memory_brief": trace.memory_brief,
            "agent_steps": AgentTraceEvaluator._load_json(trace.agent_steps_json),
            "retrieved_knowledge": AgentTraceEvaluator._load_json(trace.retrieved_knowledge_json),
            "response_messages": AgentTraceEvaluator._load_json(trace.response_messages_json),
            "assessment": AgentTraceEvaluator._load_json(trace.assessment_json),
        }

    @staticmethod
    def _load_json(value: str | None) -> Any:
        if not value:
            return []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON: %s", value[:200])
            return []

    @staticmethod
    def _result_dict(evaluated: EvaluatedTrace) -> dict[str, Any]:
        return {
            "trace_id": evaluated.trace_id or evaluated.session_id,
            "session_id": evaluated.session_id,
            "original_input": evaluated.original_input,
            "intent": evaluated.intent,
            "risk_level": evaluated.risk_level,
            "passed": evaluated.scores.passed,
            "scores": asdict(evaluated.scores),
        }

    @staticmethod
    def _compute_averages(results: list[dict[str, Any]]) -> dict[str, float]:
        if not results:
            return {}
        keys = ["intent_score", "risk_score", "retrieval_score", "response_score", "safety_score", "overall_score"]
        averages = {}
        for key in keys:
            values = [r["scores"][key] for r in results if key in r.get("scores", {})]
            averages[key] = round(sum(values) / len(values), 2) if values else 0.0
        return averages
