from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.enums import EmotionLabel, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, PromptTemplates, has_consult_signal, has_high_risk_signal


# ---------------------------------------------------------------------------
# 本地 LoRA 微调情绪分类模型适配层
#
# 微调训练数据（psychqa_synthetic.jsonl）的 output 是纯中文标签：
#   正常 / 焦虑 / 低落 / 高风险
# 与面向 API 大模型的 psychology_prompt（要求输出英文 JSON）格式不同。
# 因此当解析到的客户端是本地微调模型时，必须用「训练时一模一样的 instruction」
# 调用，并把中文标签映射回系统内部枚举，否则 json.loads 会失败并退化成 heuristic。
# ---------------------------------------------------------------------------

# 与训练集 instruction 严格对齐（见 data/lora/psychqa_synthetic.jsonl）
FINETUNED_EMOTION_INSTRUCTION = "分析用户文本情绪，只能输出：正常、焦虑、低落、高风险"

# 中文标签 -> 内部情绪枚举
CHINESE_EMOTION_LABELS: dict[str, EmotionLabel] = {
    "正常": EmotionLabel.NORMAL,
    "焦虑": EmotionLabel.ANXIETY,
    "低落": EmotionLabel.DEPRESSED,
    "高风险": EmotionLabel.HIGH_RISK,
}

# 包含匹配时的优先级：「高风险」需要先于其它标签判定
_LABEL_MATCH_ORDER = ("高风险", "低落", "焦虑", "正常")

# 本地模型不输出置信度，给一个固定的校准置信度（仅用于排序/展示，不参与安全决策）
LOCAL_MODEL_CONFIDENCE = 0.9


@dataclass
class PsychologyAssessment:
    emotion: EmotionLabel
    emotion_score: float
    risk: RiskLevel
    confidence: float
    summary: str


def match_chinese_emotion_label(raw: str) -> str | None:
    """从模型原始输出中稳健地抽取中文情绪标签。

    优先精确匹配；模型若带了标点/前后缀（如「情绪：焦虑。」）再退化为包含匹配。
    无法识别返回 None，由调用方决定兜底策略。
    """
    text = (raw or "").strip()
    if not text:
        return None
    if text in CHINESE_EMOTION_LABELS:
        return text
    for label in _LABEL_MATCH_ORDER:
        if label in text:
            return label
    return None


class PsychologicalAssessmentService:
    def __init__(self, ai: AiClient):
        self.ai = ai

    def assess(self, text: str, history: list[AiMessage] | None = None) -> PsychologyAssessment:
        # 高风险硬规则始终在最前，且不调用任何模型（与既有安全策略一致）
        if has_high_risk_signal(text):
            return PsychologyAssessment(EmotionLabel.HIGH_RISK, 4.0, RiskLevel.HIGH, 0.95, "检测到明确高风险表达")

        # 本地 LoRA 微调模型：用训练对齐的纯中文标签格式
        if self._is_local_finetuned():
            local = self._assess_with_finetuned(text)
            if local is not None:
                return local
            # 微调模型不可用或输出无法解析时，直接走确定性 heuristic，
            # 不再用英文 JSON prompt 去问一个没被这样训练过的模型。
            return heuristic(text)

        # API 大模型：英文结构化 JSON 路径
        try:
            raw = self.ai.complete(PromptTemplates.psychology_prompt(history or [], text))
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start:end + 1] if start >= 0 and end > start else raw)
            emotion = EmotionLabel(data.get("emotion", "NORMAL").upper())
            score = float(data.get("emotionScore", score_for_emotion(emotion)))
            risk = RiskLevel(data.get("risk", risk_from_score(score).value).upper())
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.75))))
            score_risk = risk_from_score(score)
            if risk_order(score_risk) > risk_order(risk):
                risk = score_risk
            # 如果情绪标签是 HIGH_RISK，强制将风险设为 HIGH。
            if emotion == EmotionLabel.HIGH_RISK:
                risk = RiskLevel.HIGH
            return PsychologyAssessment(emotion, score, risk, confidence, data.get("summary", "模型评估结果"))
        except Exception:
            # 如果 LLM 调用或解析失败，就执行确定性 heuristic
            return heuristic(text)

    # ------------------------------------------------------------------
    # 本地微调模型路径
    # ------------------------------------------------------------------

    def _is_local_finetuned(self) -> bool:
        """判断当前 AiClient 是否解析到了本地 LoRA 微调模型（Ollama）。

        AgentModelRegistry.client_for 会把 per-agent profile 写回 client.settings，
        因此这里能读到该 Agent 实际使用的 provider/model。
        """
        settings = getattr(self.ai, "settings", None)
        if settings is None:
            return False
        provider = str(getattr(settings, "ai_provider", "")).lower()
        model = str(getattr(settings, "ollama_model", ""))
        target = str(getattr(settings, "finetuned_model_name", ""))
        return provider == "ollama" and bool(target) and model == target

    def _assess_with_finetuned(self, text: str) -> PsychologyAssessment | None:
        """用训练对齐的 instruction 调本地模型，解析中文标签。任何异常都返回 None。"""
        try:
            raw = self.ai.complete(
                [
                    AiMessage(role="system", content=FINETUNED_EMOTION_INSTRUCTION),
                    AiMessage(role="user", content=text),
                ]
            )
        except Exception:
            return None
        label = match_chinese_emotion_label(raw)
        if label is None:
            return None
        emotion = CHINESE_EMOTION_LABELS[label]
        score = score_for_emotion(emotion)
        risk = risk_from_score(score)
        if emotion == EmotionLabel.HIGH_RISK:
            risk = RiskLevel.HIGH
        return PsychologyAssessment(emotion, score, risk, LOCAL_MODEL_CONFIDENCE, f"本地微调模型情绪识别：{label}")


def heuristic(text: str) -> PsychologyAssessment:
    if has_consult_signal(text):
        if any(word in text.lower() for word in ["抑郁", "低落", "崩溃", "难过", "depress", "hopeless"]):
            return PsychologyAssessment(EmotionLabel.DEPRESSED, 3.1, RiskLevel.MEDIUM, 0.75, "检测到低落或抑郁相关表达")
        return PsychologyAssessment(EmotionLabel.ANXIETY, 2.2, RiskLevel.LOW, 0.72, "检测到焦虑或压力相关表达")
    return PsychologyAssessment(EmotionLabel.NORMAL, 0.0, RiskLevel.LOW, 0.66, "未检测到明显风险信号")


def score_for_emotion(emotion: EmotionLabel) -> float:
    return {
        EmotionLabel.HIGH_RISK: 4.0,
        EmotionLabel.DEPRESSED: 3.0,
        EmotionLabel.ANXIETY: 2.0,
        EmotionLabel.NORMAL: 0.0,
    }[emotion]


def risk_from_score(score: float) -> RiskLevel:
    if score >= 4:
        return RiskLevel.HIGH
    if score >= 3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def risk_order(risk: RiskLevel) -> int:
    return {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}[risk]
