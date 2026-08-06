"""CareTrace 离线评测数据结构（Pydantic v2）。

分三层：
  - 数据集侧：GoldenCase / ExpectedSpec / RubricSpec / ModelScript
  - Trace 侧：TraceRecord / TraceEventRecord / TraceArtifactRecord / TraceTaskRecord
    （与 app.services.trace.build_trace_record 的 v2 输出一一对应，宽容解析）
  - 结果侧：HardGateResult / RubricScore / CaseEvalResult
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# rubric 六个维度（顺序即报告输出顺序）
RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "risk_alignment",
    "relevance",
    "empathy_boundary",
    "actionability",
    "groundedness",
    "trajectory_efficiency",
)


class _LenientModel(BaseModel):
    """忽略未知字段，兼容 trace 后续加字段。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ---------------------------------------------------------------------------
# 数据集侧
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ExpectedSpec(BaseModel):
    """case 级声明式期望：intent/risk 为可接受标签集（命中其一即算命中）。"""

    intent: list[str] = Field(min_length=1)
    risk: list[str] = Field(min_length=1)
    requiredEvents: list[str] = []
    forbiddenEvents: list[str] = []
    requiredArtifacts: list[str] = []  # artifact kind
    requiredAgents: list[str] = []
    forbiddenAgents: list[str] = []
    partialOrder: list[list[str]] = []  # 每对 [A, B] 要求 A 首次出现早于 B 首次出现
    maxRounds: int | None = None
    maxRevisions: int | None = None


class RubricSpec(BaseModel):
    """六个维度的目标分（0-2），case 的 rubric 通过条件是各维得分 >= 目标分。"""

    risk_alignment: int = Field(default=2, ge=0, le=2)
    relevance: int = Field(default=2, ge=0, le=2)
    empathy_boundary: int = Field(default=2, ge=0, le=2)
    actionability: int = Field(default=2, ge=0, le=2)
    groundedness: int = Field(default=2, ge=0, le=2)
    trajectory_efficiency: int = Field(default=2, ge=0, le=2)

    def targets(self) -> dict[str, int]:
        return {dim: getattr(self, dim) for dim in RUBRIC_DIMENSIONS}


class ModelScript(BaseModel):
    """确定性回放脚本：指定 fake 模型在各类 prompt 下的输出。"""

    intent: str | None = None
    risk: str | None = None
    response: str | None = None
    # 回复文本序列：非空时 ResponseAgent 的文本生成按调用顺序依次消费，
    # 用尽后回落到 response（其次默认逻辑）。用于"首次不安全、修订后安全"等场景。
    responses: list[str] = []


class GoldenCase(BaseModel):
    caseId: str
    source: Literal["human", "synthetic", "production"]
    labelStatus: Literal["gold", "silver"]
    split: Literal["dev", "heldout"] = "dev"
    category: str = ""
    conversation: list[ConversationTurn] = Field(min_length=1)
    expected: ExpectedSpec
    rubric: RubricSpec | None = None
    modelScript: ModelScript | None = None


# ---------------------------------------------------------------------------
# Trace 侧（对应 build_trace_record 输出）
# ---------------------------------------------------------------------------


class TraceEventRecord(_LenientModel):
    eventId: str = ""
    traceId: str = ""
    eventType: str
    actor: str = ""
    taskId: str = ""
    round: int = 0
    timestamp: datetime | str | None = None
    durationMs: float | None = None
    inputArtifactIds: list[str] = []
    outputArtifactIds: list[str] = []
    metadata: dict[str, Any] = {}


class TraceArtifactRecord(_LenientModel):
    id: str
    owner: str = ""
    kind: str
    version: int = 1
    confidence: float = 1.0
    taskId: str = ""
    metadata: dict[str, Any] = {}
    payload: dict[str, Any] = {}


class TraceTaskRecord(_LenientModel):
    id: str
    title: str = ""
    status: str = "OPEN"
    priority: str = "NORMAL"
    requiredCapabilities: list[str] = []
    claimedBy: list[str] = []
    createdBy: str = ""
    dependsOn: list[str] = []
    metadata: dict[str, Any] = {}


class TraceRecord(_LenientModel):
    traceId: str = ""
    traceVersion: str = ""
    status: str = "COMPLETED"
    intent: str = "CHAT"
    riskLevel: str = "LOW"
    finalResponse: str = ""
    finalResponseArtifactId: str = ""
    finalReviewArtifactId: str = ""
    startedAt: datetime | str | None = None
    completedAt: datetime | str | None = None
    durationMs: float | None = None
    events: list[TraceEventRecord] = []
    artifacts: list[TraceArtifactRecord] = []
    tasks: list[TraceTaskRecord] = []
    metrics: dict[str, Any] = {}
    error: dict[str, Any] = {}

    def event_types(self) -> list[str]:
        return [event.eventType for event in self.events]

    def first_index(self, event_type: str) -> int | None:
        for index, event in enumerate(self.events):
            if event.eventType == event_type:
                return index
        return None

    def artifact_by_id(self, artifact_id: str) -> TraceArtifactRecord | None:
        return next((artifact for artifact in self.artifacts if artifact.id == artifact_id), None)


# ---------------------------------------------------------------------------
# 结果侧
# ---------------------------------------------------------------------------


class HardGateResult(BaseModel):
    passed: bool
    failures: list[str] = []  # 人类可读，带规则名
    hardFailures: list[str] = []  # failures 的高危子集（高危直接判死）


class RubricScore(BaseModel):
    """六维评分结果。某维评分为 None 表示 judge 无法评定该维（judgeError）。"""

    scores: dict[str, int | None] = {}
    hardFailures: list[str] = []
    evidence: list[str] = []
    judgeErrors: list[str] = []
    available: bool = True  # False 表示 rubric 不可用（judge 解析失败等），不算中间分
    total: int = 0
    passed: bool = False

    @field_validator("scores")
    @classmethod
    def _validate_scores(cls, scores: dict[str, int | None]) -> dict[str, int | None]:
        unknown = set(scores) - set(RUBRIC_DIMENSIONS)
        if unknown:
            raise ValueError(f"未知 rubric 维度: {sorted(unknown)}")
        for dim, value in scores.items():
            if value is not None and not (0 <= value <= 2):
                raise ValueError(f"rubric 维度 {dim} 得分 {value} 超出 0-2 范围")
        return scores

    @model_validator(mode="after")
    def _recompute_total(self) -> "RubricScore":
        self.total = sum(value for value in self.scores.values() if value is not None)
        return self


class CaseEvalResult(BaseModel):
    caseId: str
    split: str = "dev"
    labelStatus: str = "gold"
    category: str = ""
    expectedIntent: list[str] = []
    expectedRisk: list[str] = []
    actualIntent: str = ""
    actualRisk: str = ""
    intentHit: bool = False
    riskHit: bool = False
    hardGate: HardGateResult = HardGateResult(passed=True)
    rubric: RubricScore | None = None
    passed: bool = False
    rounds: int = 0
    revisions: int = 0
    durationMs: float | None = None
    failureDetails: list[str] = []
    # 报告聚合用的运行侧指标
    finalResponse: str = ""
    budgetExhausted: bool = False
    agentExecutions: int = 0
    invalidAgentActivations: int = 0
