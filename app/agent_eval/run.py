from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agent_eval.evaluator import AgentTraceEvaluator
from app.agent_eval.report import EvaluationReportWriter
from app.core.config import get_settings
from app.core.database import SessionLocal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Agent Run Traces with rubrics.")
    parser.add_argument("--dataset", type=Path, help="Path to JSON dataset of traces + ground truth.")
    parser.add_argument("--session-id", type=int, help="Evaluate traces for a specific session ID.")
    parser.add_argument("--limit", type=int, default=100, help="Max traces to evaluate from DB.")
    parser.add_argument("--llm", action="store_true", help="Enable LLM-as-a-Judge for response quality.")
    parser.add_argument("--output", type=Path, default="target/agent-trace-eval-report.json")
    args = parser.parse_args(argv)

    settings = get_settings()

    evaluator = AgentTraceEvaluator(settings, SessionLocal(), use_llm=args.llm)

    if args.dataset:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        report = evaluator.evaluate_dataset(dataset)
    elif args.session_id:
        report = evaluator.evaluate_by_session(args.session_id)
    else:
        report = evaluator.evaluate_latest(limit=args.limit)

    writer = EvaluationReportWriter(args.output)
    writer.write(report)
    writer.print_summary(report)
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
