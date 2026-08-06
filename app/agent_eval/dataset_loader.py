"""Golden dataset 加载：逐行 JSONL + Pydantic 校验，错误带行号。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.agent_eval.schemas import GoldenCase


class DatasetLoadError(Exception):
    """数据集文件无法解析（带行号与原因）。"""


def load_dataset(path: str | Path) -> list[GoldenCase]:
    path = Path(path)
    if not path.exists():
        raise DatasetLoadError(f"数据集文件不存在: {path}")
    cases: list[GoldenCase] = []
    seen_ids: set[str] = set()
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(f"第 {line_no} 行不是合法 JSON: {exc}") from exc
        try:
            case = GoldenCase.model_validate(data)
        except ValidationError as exc:
            raise DatasetLoadError(f"第 {line_no} 行校验失败: {exc.errors()[0].get('msg')} ({exc.errors()[0].get('loc')})") from exc
        if case.caseId in seen_ids:
            raise DatasetLoadError(f"第 {line_no} 行 caseId 重复: {case.caseId}")
        seen_ids.add(case.caseId)
        cases.append(case)
    return cases
