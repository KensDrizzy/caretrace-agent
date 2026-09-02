"""混合模型路由 + 本地 LoRA 情绪分类适配层的单元测试。"""

import unittest

from app.core.config import Settings
from app.core.enums import EmotionLabel, RiskLevel
from app.services.agent_models import AgentModelRegistry
from app.services.assessment import (
    PsychologicalAssessmentService,
    match_chinese_emotion_label,
    FINETUNED_EMOTION_INSTRUCTION,
)


class _FakeSettings:
    def __init__(self, provider="ollama", ollama_model="mindbridge-qwen2.5-7b-ft:latest",
                 finetuned_name="mindbridge-qwen2.5-7b-ft:latest"):
        self.ai_provider = provider
        self.ollama_model = ollama_model
        self.finetuned_model_name = finetuned_name


class _FakeAi:
    """记录调用 prompt、返回固定内容的假 AiClient。"""

    def __init__(self, reply, settings=None):
        self.settings = settings or _FakeSettings()
        self.reply = reply
        self.messages_seen = []

    def complete(self, messages):
        self.messages_seen.append(messages)
        return self.reply


class ChineseLabelMatchTests(unittest.TestCase):
    def test_exact_label(self):
        self.assertEqual(match_chinese_emotion_label("焦虑"), "焦虑")
        self.assertEqual(match_chinese_emotion_label("高风险"), "高风险")

    def test_fuzzy_label_with_punctuation(self):
        self.assertEqual(match_chinese_emotion_label("情绪：焦虑。"), "焦虑")
        self.assertEqual(match_chinese_emotion_label(" 正常\n"), "正常")

    def test_high_risk_priority(self):
        self.assertEqual(match_chinese_emotion_label("判定为高风险倾向"), "高风险")

    def test_unrecognized_returns_none(self):
        self.assertIsNone(match_chinese_emotion_label(""))
        self.assertIsNone(match_chinese_emotion_label("我不知道"))


class LocalFinetunedAssessmentTests(unittest.TestCase):
    def _service(self, reply):
        ai = _FakeAi(reply)
        return PsychologicalAssessmentService(ai), ai

    def test_local_uses_training_aligned_instruction(self):
        service, ai = self._service("焦虑")
        result = service.assess("下周答辩，我总觉得没准备好，很紧张。")
        self.assertEqual(result.emotion, EmotionLabel.ANXIETY)
        self.assertEqual(result.risk, RiskLevel.LOW)
        # system prompt 必须与训练 instruction 严格对齐
        system_msg = ai.messages_seen[0][0]
        self.assertEqual(system_msg.content, FINETUNED_EMOTION_INSTRUCTION)

    def test_local_depressed_maps_to_medium(self):
        service, _ = self._service("低落")
        result = service.assess("对什么都提不起劲，心里闷闷的。")
        self.assertEqual(result.emotion, EmotionLabel.DEPRESSED)
        self.assertEqual(result.risk, RiskLevel.MEDIUM)

    def test_local_high_risk_label(self):
        service, _ = self._service("高风险")
        result = service.assess("想找个方式让这一切彻底停下来。")
        self.assertEqual(result.emotion, EmotionLabel.HIGH_RISK)
        self.assertEqual(result.risk, RiskLevel.HIGH)

    def test_local_garbage_falls_back_to_heuristic(self):
        # 本地模型输出无法解析时不应抛异常，退化为确定性 heuristic
        service, _ = self._service("###无法理解的输出###")
        result = service.assess("最近压力很大，经常失眠。")
        self.assertIn(result.risk, {RiskLevel.LOW, RiskLevel.MEDIUM})

    def test_high_risk_keyword_hard_guards_before_local_model(self):
        class Exploding(_FakeAi):
            def complete(self, messages):
                raise AssertionError("命中高风险关键词时不应调用模型")
        service = PsychologicalAssessmentService(Exploding(""))
        result = service.assess("我不想活了，想结束生命")
        self.assertEqual(result.risk, RiskLevel.HIGH)

    def test_non_finetuned_client_still_uses_json_path(self):
        settings = _FakeSettings(provider="deepseek", ollama_model="other", finetuned_name="ft")
        ai = _FakeAi(
            '{"emotion":"ANXIETY","emotionScore":2.0,"risk":"LOW","confidence":0.8,"summary":"x"}',
            settings=settings,
        )
        result = PsychologicalAssessmentService(ai).assess("普通的一句压力描述")
        self.assertEqual(result.emotion, EmotionLabel.ANXIETY)
        # 走的是 JSON prompt，而不是训练 instruction
        self.assertIn("JSON", ai.messages_seen[0][0].content)


class PerTaskModelRoutingTests(unittest.TestCase):
    def _settings(self):
        return Settings(
            agent_model_safety_provider="ollama",
            agent_model_safety_model="mindbridge-qwen2.5-7b-ft:latest",
            agent_model_safety_review_provider="deepseek",
            agent_model_safety_review_model="deepseek-v4-flash",
            agent_model_response_provider="deepseek",
            agent_model_response_model="deepseek-v4-flash",
            ai_provider="deepseek",
        )

    def test_safety_classification_uses_local(self):
        registry = AgentModelRegistry(self._settings())
        profile = registry.profile_for("SafetyAgent")
        self.assertEqual(profile.provider, "ollama")
        self.assertEqual(profile.model, "mindbridge-qwen2.5-7b-ft:latest")

    def test_safety_semantic_review_uses_api(self):
        registry = AgentModelRegistry(self._settings())
        profile = registry.profile_for_alias("safety_review")
        self.assertEqual(profile.provider, "deepseek")
        self.assertEqual(profile.model, "deepseek-v4-flash")

    def test_client_for_alias_writes_back_provider(self):
        registry = AgentModelRegistry(self._settings())
        local_client = registry.client_for_alias("safety")
        review_client = registry.client_for_alias("safety_review")
        self.assertEqual(local_client.settings.ai_provider, "ollama")
        self.assertEqual(local_client.settings.ollama_model, "mindbridge-qwen2.5-7b-ft:latest")
        self.assertEqual(review_client.settings.ai_provider, "deepseek")
        self.assertEqual(review_client.settings.deepseek_model, "deepseek-v4-flash")

    def test_empty_profile_falls_back_to_global(self):
        # Ignore the developer's local .env so this test really exercises an
        # empty per-agent profile falling back to the global provider.
        settings = Settings(_env_file=None, ai_provider="deepseek", deepseek_model="deepseek-v4-flash")
        registry = AgentModelRegistry(settings)
        profile = registry.profile_for_alias("context")
        self.assertEqual(profile.provider, "deepseek")
        self.assertEqual(profile.model, "deepseek-v4-flash")


if __name__ == "__main__":
    unittest.main()
