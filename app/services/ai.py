from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import httpx

from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.core.http_client import get_async_client, get_sync_client
from app.schemas.dtos import AiMessage


class PromptTemplates:
    @staticmethod
    def intent_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        return [
            AiMessage(role="system", content=(
                "你是一个用户意图分类器，只做意图识别，不回答问题。"
                "只输出 CHAT、CONSULT、RISK 之一。CHAT 包含普通闲聊、学习、编程、作业、校园事务；"
                "CONSULT 包含压力、焦虑、低落、失眠、情绪倾诉；RISK 包含自杀、自残、伤人或即时危险信号。"
            )),
            AiMessage(role="user", content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}"),
        ]

    @staticmethod
    def psychology_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        return [
            AiMessage(role="system", content=(
                "你负责分析校园心理健康消息。只返回严格 JSON："
                '{"emotion":"NORMAL|ANXIETY|DEPRESSED|HIGH_RISK","emotionScore":0.0,'
                '"risk":"LOW|MEDIUM|HIGH","confidence":0.0,"summary":"short reason"}'
            )),
            AiMessage(role="user", content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}"),
        ]

    @staticmethod
    def answer_system_prompt(intent: IntentType, risk: RiskLevel, context: str, display_name: str, skill_context: str = "") -> AiMessage:
        if intent == IntentType.CHAT:
            content = (
                "你是 CareTrace，一个面向学生的日常陪伴与校园生活助手。"
                "普通学习、编程、校园事务和通用问题请自然、准确、直接地回答。"
                "不要主动做心理测评，不要输出风险等级、心理标签、诊断结论或报告口吻。"
                f"学生显示名：{display_name}"
            )
            return AiMessage(role="system", content=content)
        crisis_rule = ""
        if risk == RiskLevel.HIGH:
            crisis_rule = (
                "\n高风险处理规则：先回应情绪，再关注当前安全；鼓励用户立刻联系身边可信任的人、"
                "学校辅导员/心理中心或当地紧急救助；不提供任何危险操作细节。"
            )
        content = (
            "你是 CareTrace，一个面向学生的校园心理关怀智能体。"
            "回答要共情、谨慎、非评判，不诊断疾病，不开药，不替代持证心理咨询师。"
            "不要向学生输出风险等级、报告分数或后台标签。"
            "下面提供了检索到的校园心理知识，你必须直接基于这些知识给出具体、可操作的建议，"
            "不要只给泛泛安慰，也不要在已有明确建议时继续反问用户。"
            "如果检索知识为空或明显不足，再给出安全的通用支持。"
            f"\n学生显示名：{display_name}\n检索知识：\n{context}\n\n可用 skill 指引：\n{skill_context or '无'}{crisis_rule}"
        )
        return AiMessage(role="system", content=content)


class AiError(Exception):
    """Base class for AI provider failures."""


class ModelConnectionError(AiError):
    """Provider is unreachable or timed out."""


class ModelTimeoutError(ModelConnectionError):
    """Provider did not respond in time."""


class ModelNotFoundError(AiError):
    """Requested model does not exist on the provider."""


@dataclass(frozen=True)
class LlmCallMetrics:
    model_name: str
    duration_ms: float
    status: str  # "OK" / "ERROR"
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_type: str | None = None
    retry_count: int = 0
    truncated: bool = False  # finish_reason/done_reason == "length"，输出被 max_tokens 截断


class AiClient:
    def __init__(self, settings: Settings, metrics_hook: Callable[[LlmCallMetrics], None] | None = None):
        self.settings = settings
        self.metrics_hook = metrics_hook

    def complete(self, messages: list[AiMessage]) -> str:
        provider = self.settings.ai_provider.lower()
        started = time.perf_counter()
        try:
            if provider == "ollama":
                content, input_tokens, output_tokens, truncated = self._ollama(messages, stream=False)
            elif provider in {"openai", "deepseek"}:
                content, input_tokens, output_tokens, truncated = self._openai(messages, stream=False, use_deepseek=provider == "deepseek")
            else:
                content, input_tokens, output_tokens, truncated = self._mock(messages), None, None, False
        except Exception as exc:
            self._emit_metrics(provider, started, status="ERROR", error_type=type(exc).__name__)
            raise
        self._emit_metrics(provider, started, status="OK", input_tokens=input_tokens, output_tokens=output_tokens, truncated=truncated)
        return content

    def _emit_metrics(
        self,
        provider: str,
        started: float,
        status: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error_type: str | None = None,
        truncated: bool = False,
    ) -> None:
        if self.metrics_hook is None:
            return
        try:
            self.metrics_hook(
                LlmCallMetrics(
                    model_name=self._model_name(provider),
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    status=status,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error_type=error_type,
                    truncated=truncated,
                )
            )
        except Exception:
            pass

    def _model_name(self, provider: str) -> str:
        if provider == "ollama":
            return self.settings.ollama_model
        if provider == "openai":
            return self.settings.openai_model
        if provider == "deepseek":
            return self.settings.deepseek_model
        return "mock"

    async def stream(self, messages: list[AiMessage]):
        provider = self.settings.ai_provider.lower()
        if provider == "ollama":
            async for token in self._ollama_stream(messages):
                yield token
            return
        if provider in {"openai", "deepseek"}:
            async for token in self._openai_stream(messages, use_deepseek=provider == "deepseek"):
                yield token
            return
        text = self._mock(messages)
        for chunk in split_text(text, 12):
            yield chunk

    def _ollama(self, messages: list[AiMessage], stream: bool) -> tuple[str, int | None, int | None, bool]:
        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "options": {"temperature": self.settings.ai_temperature, "num_predict": self.settings.ai_max_tokens},
        }
        try:
            response = get_sync_client(self.settings).post(f"{self.settings.ollama_base_url}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError("模型服务响应超时，请稍后重试") from exc
        except httpx.ConnectError as exc:
            raise ModelConnectionError("无法连接到模型服务，请确认 Ollama 已启动") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ModelNotFoundError(f"模型未找到：{self.settings.ollama_model}。请运行 create-finetuned-model.sh 或检查 Ollama 模型列表。") from exc
            raise AiError(f"模型服务错误 ({exc.response.status_code})") from exc
        data = response.json()
        truncated = data.get("done_reason") == "length"
        return data["message"]["content"], data.get("prompt_eval_count"), data.get("eval_count"), truncated

    async def _ollama_stream(self, messages: list[AiMessage]):
        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "options": {"temperature": self.settings.ai_temperature, "num_predict": self.settings.ai_max_tokens},
        }
        try:
            async with get_async_client(self.settings).stream("POST", f"{self.settings.ollama_base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError("模型服务响应超时，请稍后重试") from exc
        except httpx.ConnectError as exc:
            raise ModelConnectionError("无法连接到模型服务，请确认 Ollama 已启动") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ModelNotFoundError(f"模型未找到：{self.settings.ollama_model}。请运行 create-finetuned-model.sh 或检查 Ollama 模型列表。") from exc
            raise AiError(f"模型服务错误 ({exc.response.status_code})") from exc

    def _openai(self, messages: list[AiMessage], stream: bool, use_deepseek: bool = False) -> tuple[str, int | None, int | None, bool]:
        base_url = self.settings.deepseek_base_url if use_deepseek else self.settings.openai_base_url
        api_key = self.settings.deepseek_api_key if use_deepseek else self.settings.openai_api_key
        model = self.settings.deepseek_model if use_deepseek else self.settings.openai_model
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
        }
        if _is_gpt5_model(model):
            payload["max_completion_tokens"] = self.settings.ai_max_tokens
            payload["reasoning_effort"] = self.settings.openai_reasoning_effort
        else:
            payload["temperature"] = self.settings.ai_temperature
            payload["max_tokens"] = self.settings.ai_max_tokens
        try:
            response = get_sync_client(self.settings).post(f"{base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError("模型服务响应超时，请稍后重试") from exc
        except httpx.ConnectError as exc:
            raise ModelConnectionError("无法连接到模型服务，请检查网络或 API 地址") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ModelNotFoundError(f"模型未找到：{model}") from exc
            raise AiError(f"模型服务错误 ({exc.response.status_code})") from exc
        data = response.json()
        usage = data.get("usage") or {}
        choice = data["choices"][0]
        truncated = choice.get("finish_reason") == "length"
        return choice["message"]["content"], usage.get("prompt_tokens"), usage.get("completion_tokens"), truncated

    async def _openai_stream(self, messages: list[AiMessage], use_deepseek: bool = False):
        base_url = self.settings.deepseek_base_url if use_deepseek else self.settings.openai_base_url
        api_key = self.settings.deepseek_api_key if use_deepseek else self.settings.openai_api_key
        model = self.settings.deepseek_model if use_deepseek else self.settings.openai_model
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
        }
        if _is_gpt5_model(model):
            payload["max_completion_tokens"] = self.settings.ai_max_tokens
            payload["reasoning_effort"] = self.settings.openai_reasoning_effort
        else:
            payload["temperature"] = self.settings.ai_temperature
            payload["max_tokens"] = self.settings.ai_max_tokens
        try:
            async with get_async_client(self.settings).stream("POST", f"{base_url}/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line.removeprefix("data: ").strip()
                    if raw == "[DONE]":
                        break
                    data = json.loads(raw)
                    token = data["choices"][0].get("delta", {}).get("content", "")
                    if token:
                        yield token
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError("模型服务响应超时，请稍后重试") from exc
        except httpx.ConnectError as exc:
            raise ModelConnectionError("无法连接到模型服务，请检查网络或 API 地址") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ModelNotFoundError(f"模型未找到：{model}") from exc
            raise AiError(f"模型服务错误 ({exc.response.status_code})") from exc

    def _mock(self, messages: list[AiMessage]) -> str:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system = " ".join(m.content for m in messages if m.role == "system")
        if "严格 JSON" in system:
            if has_high_risk_signal(last):
                return '{"emotion":"HIGH_RISK","emotionScore":4.0,"risk":"HIGH","confidence":0.95,"summary":"检测到明确高风险表达"}'
            if has_consult_signal(last):
                return '{"emotion":"ANXIETY","emotionScore":2.5,"risk":"LOW","confidence":0.72,"summary":"检测到压力或情绪求助表达"}'
            return '{"emotion":"NORMAL","emotionScore":0.0,"risk":"LOW","confidence":0.66,"summary":"未检测到明显风险信号"}'
        if "意图分类器" in system:
            if has_high_risk_signal(last):
                return "RISK"
            if has_consult_signal(last):
                return "CONSULT"
            return "CHAT"
        if "high_risk_safety_plan" in system and has_high_risk_signal(last):
            return "我听到你现在已经痛苦到觉得撑不下去了。现在最重要的是先让你不要一个人扛：请马上联系身边可信任的人，或者直接联系辅导员、学校心理中心、校园保卫/当地紧急服务。接下来 10 分钟，请先把自己移到有人在的地方，并把可能伤害自己的东西放远一点。如果可以，回我一句：你现在身边有没有可以马上联系或走过去找的人？"
        if "当前由 ResponseAgent 以 support mode" in system:
            return "我听到你最近压力很大，还影响到了睡眠，这种状态确实会让人很消耗。你可以先做两件小事：今晚把最担心的事情写成清单，先只选一个最小步骤处理；睡前 30 分钟把手机和学习任务放远一点，用缓慢呼吸或热水澡帮身体降下来。如果这种失眠持续一周以上，建议联系学校心理中心或辅导员一起看一看。"
        if "当前由 ResponseAgent 以 normal_chat mode" in system:
            return "我在。这个问题可以直接拆开来看，我们先从你最想解决的那一部分开始。"
        if "ContextAgent" in system and "只输出查询词" in system:
            if "当前输入：" in last:
                current = last.split("当前输入：")[-1].strip().split("\n")[0].strip()
                return _mock_rewrite_query(current)
            return _mock_rewrite_query(last)
        if "ContextAgent" in system and "SUFFICIENT" in system:
            return "SUFFICIENT"
        if "ContextAgent" in system:
            return last[:40] or "校园心理支持"
        return "我在。先把你现在最具体的困扰说出来，我们可以一步一步拆开。如果情况已经影响安全，请马上联系身边可信任的人或学校心理中心。"


def format_history(history: list[AiMessage]) -> str:
    if not history:
        return "无"
    return "\n".join(f"{m.role}: {m.content}" for m in history[-20:])


HIGH_RISK_WORDS = ["自杀", "自残", "不想活", "结束生命", "伤害自己", "轻生", "suicide", "kill myself", "self harm"]
CONSULT_WORDS = [
    "焦虑", "抑郁", "压力", "失眠", "难过", "崩溃", "痛苦", "无助", "心理", "咨询",
    "anxious", "depress", "stress",
    # 学业/生涯压力也纳入咨询信号，避免"改论文+实习+秋招"被误判为普通任务
    "秋招", "实习", "毕设", "找工作", "学业", "挂科", "保研", "考研",
]


def has_high_risk_signal(text: str) -> bool:
    normalized = text.lower()
    return any(word in normalized for word in HIGH_RISK_WORDS)


def has_consult_signal(text: str) -> bool:
    normalized = text.lower()
    return any(word in normalized for word in CONSULT_WORDS)


def split_text(text: str, size: int) -> Iterable[str]:
    for index in range(0, len(text), size):
        yield text[index:index + size]


def _is_gpt5_model(model: str) -> bool:
    """GPT-5-family Chat Completions use reasoning/max-completion parameters."""
    return model.strip().lower().startswith("gpt-5")


def _mock_rewrite_query(current: str) -> str:
    """Deterministic query rewrite used by the mock provider in tests.

    Mirrors a real LLM rewrite by expanding common synonyms so that the
    rewritten query is more likely to hit the knowledge base.
    """
    current = current.strip() or "校园心理支持"
    expansions = [
        ("睡不着", "失眠"),
        ("失眠", "睡不着"),
        ("心跳", "呼吸"),
        ("喘不过气", "呼吸"),
        ("想伤害自己", "自杀"),
        ("自杀", "自伤"),
    ]
    extras: list[str] = []
    for trigger, keyword in expansions:
        if trigger in current and keyword not in current and keyword not in extras:
            extras.append(keyword)
    if extras:
        query = f"{current} {' '.join(extras)}"
    else:
        query = current
    return query[:60]
