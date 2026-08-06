from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from app.agents.events import AgentTask, AgentTurnResult, CollaborationBlackboard


class AgentCapability(str, Enum):
    UNDERSTANDING = "UNDERSTANDING"
    SAFETY = "SAFETY"
    CONTEXT = "CONTEXT"
    RESPONSE = "RESPONSE"
    COORDINATION = "COORDINATION"


@dataclass(frozen=True)
class AgentProfile:
    name: str
    capabilities: frozenset[AgentCapability] = field(default_factory=frozenset)
    system_prompt: str = ""
    memory_policy: str = "none"
    model_profile: str = "default"
    tool_permissions: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class AgentDecision:
    claim: bool
    confidence: float = 0.0
    reason: str = ""
    reason_code: str = ""


class AutonomousAgent(Protocol):
    profile: AgentProfile

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        ...

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        ...


@dataclass(frozen=True)
class AgentCandidate:
    agent: AutonomousAgent
    decision: AgentDecision


class AgentRegistry:
    def __init__(self, agents: list[AutonomousAgent]):
        self._agents = list(agents)

    @property
    def agents(self) -> list[AutonomousAgent]:
        return list(self._agents)

    def candidates_for(self, task: AgentTask, board: CollaborationBlackboard) -> list[AutonomousAgent]:
        return [candidate.agent for candidate in self.candidate_decisions_for(task, board)]

    def candidate_decisions_for(self, task: AgentTask, board: CollaborationBlackboard) -> list[AgentCandidate]:
        return [candidate for candidate in self.evaluate_decisions_for(task, board) if candidate.decision.claim]

    def evaluate_decisions_for(self, task: AgentTask, board: CollaborationBlackboard) -> list[AgentCandidate]:
        """Evaluate every agent against the task, including non-claiming ones.

        Agents lacking the required capabilities are not asked to decide; they
        receive a deterministic CAPABILITY_MISMATCH decision instead. Claiming
        candidates are sorted by confidence (desc) ahead of the rest.
        """
        claiming: list[AgentCandidate] = []
        declined: list[AgentCandidate] = []
        for agent in self._agents:
            if not self._has_required_capability(agent, task):
                declined.append(
                    AgentCandidate(
                        agent,
                        AgentDecision(False, 0.0, "capability mismatch", reason_code="CAPABILITY_MISMATCH"),
                    )
                )
                continue
            decision = agent.decide(task, board)
            if decision.claim:
                claiming.append(AgentCandidate(agent, decision))
            else:
                declined.append(AgentCandidate(agent, decision))
        claiming.sort(key=lambda item: item.decision.confidence, reverse=True)
        return [*claiming, *declined]

    def _has_required_capability(self, agent: AutonomousAgent, task: AgentTask) -> bool:
        if not task.required_capabilities:
            return True
        agent_capabilities = {capability.value for capability in agent.profile.capabilities}
        return set(task.required_capabilities).issubset(agent_capabilities)
