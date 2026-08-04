from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agent_eval.evaluator import EvaluationReport


class EvaluationReportWriter:
    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)

    def write(self, report: EvaluationReport) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.output_path

    @staticmethod
    def print_summary(report: EvaluationReport) -> None:
        print("=" * 60)
        print(f"Agent Trace Evaluation Report")
        print("=" * 60)
        print(f"Total traces : {report.total}")
        print(f"Passed       : {report.passed}")
        print(f"Failed       : {report.failed}")
        print(f"Pass rate    : {report.passed / report.total * 100:.1f}%" if report.total else "N/A")
        print("-" * 60)
        print("Average scores:")
        for key, value in report.average_scores.items():
            print(f"  {key:20s}: {value}")
        print("=" * 60)
