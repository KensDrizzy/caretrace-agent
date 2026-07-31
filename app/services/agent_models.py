from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.services.ai import AiClient


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
        alias = AGENT_MODEL_ALIASES.get(agent_name, _snake(agent_name.removesuffix("Agent")))
        provider = self._setting(f"agent_model_{alias}_provider", self._default_provider())
        model = self._setting(f"agent_model_{alias}_model", self._default_model(provider))
        temperature = float(self._setting(f"agent_model_{alias}_temperature", getattr(self.settings, "ai_temperature", 0.35)))
        max_tokens = int(self._setting(f"agent_model_{alias}_max_tokens", getattr(self.settings, "ai_max_tokens", 512)))
        return AgentModelProfile(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)

    def client_for(self, agent_name: str) -> AiClient:
        profile = self.profile_for(agent_name)
        settings = copy.copy(self.settings)
        settings.ai_provider = profile.provider
        settings.ai_temperature = profile.temperature
        settings.ai_max_tokens = profile.max_tokens
        if profile.provider == "openai":
            settings.openai_model = profile.model
        else:
            settings.ollama_model = profile.model
        return AiClient(settings)

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
