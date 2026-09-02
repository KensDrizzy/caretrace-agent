from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from app.core.config import Settings
from app.services.ai import AiClient, LlmCallMetrics


AGENT_MODEL_ALIASES = {
    "CoordinatorAgent": "coordinator",
    "UnderstandingAgent": "understanding",
    "SafetyAgent": "safety",
    "ContextAgent": "context",
    "ResponseAgent": "response",
}


@dataclass(frozen=True)
class AgentModelProfile:
    provider: str
    model: str
    temperature: float
    max_tokens: int


class AgentModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings

    def profile_for(self, agent_name: str) -> AgentModelProfile:
        alias = self.alias_for(agent_name)
        return self.profile_for_alias(alias)

    def alias_for(self, agent_name: str) -> str:
        return AGENT_MODEL_ALIASES.get(agent_name, _snake(agent_name.removesuffix("Agent")))

    def profile_for_alias(self, alias: str) -> AgentModelProfile:
        """按 profile 别名（understanding/safety/safety_review/context/response/...）解析模型配置。"""
        provider = self._setting(f"agent_model_{alias}_provider", self._default_provider())
        model = self._setting(f"agent_model_{alias}_model", self._default_model(provider))
        temperature = float(self._setting(f"agent_model_{alias}_temperature", getattr(self.settings, "ai_temperature", 0.35)))
        max_tokens = int(self._setting(f"agent_model_{alias}_max_tokens", getattr(self.settings, "ai_max_tokens", 512)))
        return AgentModelProfile(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)

    def client_for(self, agent_name: str, metrics_hook: Callable[[LlmCallMetrics], None] | None = None) -> AiClient:
        return self.client_for_alias(self.alias_for(agent_name), metrics_hook=metrics_hook)

    def client_for_alias(self, alias: str, metrics_hook: Callable[[LlmCallMetrics], None] | None = None) -> AiClient:
        """同一个 Agent 内部不同任务可用不同 profile（如 Safety 分类走本地、语义审核走 API）。"""
        profile = self.profile_for_alias(alias)
        settings = copy.copy(self.settings)
        settings.ai_provider = profile.provider
        settings.ai_temperature = profile.temperature
        settings.ai_max_tokens = profile.max_tokens
        if profile.provider == "openai":
            settings.openai_model = profile.model
        elif profile.provider == "deepseek":
            settings.deepseek_model = profile.model
        else:
            settings.ollama_model = profile.model
        return AiClient(settings, metrics_hook=metrics_hook)

    def _setting(self, name: str, fallback: Any) -> Any:
        value = getattr(self.settings, name, None)
        if value in {None, ""}:
            return fallback
        return value

    def _default_provider(self) -> str:
        return self._setting("agent_model_default_provider", getattr(self.settings, "ai_provider", "mock")).lower()

    def _default_model(self, provider: str) -> str:
        configured = self._setting("agent_model_default_model", "")
        if configured:
            return configured
        if provider == "openai":
            return getattr(self.settings, "openai_model", "gpt-4o-mini")
        if provider == "deepseek":
            return getattr(self.settings, "deepseek_model", "deepseek-v4-flash")
        if provider == "ollama":
            return getattr(self.settings, "ollama_model", "mindbridge-qwen2.5-7b-ft:latest")
        return "mock"


def _snake(value: str) -> str:
    chars = []
    for index, char in enumerate(value):
        if char.isupper() and index > 0:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)
