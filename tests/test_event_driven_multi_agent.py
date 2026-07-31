import unittest
from types import SimpleNamespace

from app.agents.coordinator import EventDrivenCoordinator
from app.agents.events import (
    AgentArtifact,
    AgentEvent,
    AgentEventType,
    AgentTask,
    AgentTurnResult,
    CollaborationBlackboard,
    TaskPriority,
    TaskStatus,
)
from app.agents.registry import AgentCapability, AgentDecision, AgentProfile, AgentRegistry
from app.services.agent_models import AgentModelRegistry


class EventDrivenProtocolTests(unittest.TestCase):
    def test_blackboard_is_append_only(self):
        board = CollaborationBlackboard(turn_id="t1")
        task = AgentTask(id="task-1", title="Resolve")
        event = AgentEvent(type=AgentEventType.TASK_CREATED, actor="CoordinatorAgent", task_id=task.id)

        updated = board.add_task(task).append_event(event)

        self.assertEqual(board.tasks, {})
        self.assertEqual(updated.tasks["task-1"].status, TaskStatus.OPEN)
        self.assertEqual(updated.events[0].actor, "CoordinatorAgent")

    def test_artifacts_do_not_overwrite_each_other(self):
        first = AgentArtifact(id="a1", owner="UnderstandingAgent", kind="intent", payload={"intent": "CHAT"})
        second = AgentArtifact(id="a2", owner="SafetyAgent", kind="risk", payload={"risk": "HIGH"})
        board = CollaborationBlackboard(turn_id="t1").add_artifact(first).add_artifact(second)

        self.assertEqual(len(board.artifacts), 2)
        self.assertEqual(board.artifacts_by_kind("risk")[0].payload["risk"], "HIGH")


class DemoAgent:
    def __init__(self, name, capability, confidence):
        self.profile = AgentProfile(
            name=name,
            capabilities=frozenset({capability}),
            system_prompt=f"{name} prompt",
            memory_policy="private",
            model_profile=name,
        )
        self.confidence = confidence

    def decide(self, task, board):
        return AgentDecision(True, self.confidence, f"{self.profile.name} claims")

    def act(self, task, board):
        return AgentTurnResult(
            artifacts=(
                AgentArtifact(
                    id=f"{self.profile.name}:artifact",
                    owner=self.profile.name,
                    kind="demo",
                    payload={"agent": self.profile.name},
                    task_id=task.id,
                ),
            )
        )


class RegistryAndCoordinatorTests(unittest.TestCase):
    def test_registry_sorts_by_claim_confidence_not_list_order(self):
        low = DemoAgent("LowConfidenceAgent", AgentCapability.UNDERSTANDING, 0.2)
        high = DemoAgent("HighConfidenceAgent", AgentCapability.UNDERSTANDING, 0.9)
        registry = AgentRegistry([low, high])
        task = AgentTask(id="task-1", title="Understand", required_capabilities=frozenset({AgentCapability.UNDERSTANDING.value}))

        candidates = registry.candidates_for(task, CollaborationBlackboard(turn_id="t1"))

        self.assertEqual([agent.profile.name for agent in candidates], ["HighConfidenceAgent", "LowConfidenceAgent"])

    def test_coordinator_uses_claims_for_open_tasks(self):
        settings = SimpleNamespace(
            agent_max_rounds=1,
            agent_max_claims_per_round=2,
            agent_max_claims_per_agent=1,
            agent_final_acceptance_min_confidence=0.6,
        )
        coordinator_agent = SimpleNamespace(
            name="CoordinatorAgent",
            root_task=lambda board: AgentTask(
                id="task:root",
                title="Resolve user turn",
                description=board.user_input,
                priority=TaskPriority.NORMAL,
                metadata={"kind": "root"},
            ),
            remember_acceptance=lambda artifact_id, reason: None,
        )
        registry = AgentRegistry([
            DemoAgent("AgentA", AgentCapability.UNDERSTANDING, 0.8),
            DemoAgent("AgentB", AgentCapability.SAFETY, 0.7),
        ])
        board = CollaborationBlackboard(turn_id="t1", user_input="hello", model_input="hello")

        result = EventDrivenCoordinator(registry, coordinator_agent, settings).run(board)

        claimed = [event.actor for event in result.events if event.type == AgentEventType.TASK_CLAIMED]
        self.assertIn("AgentA", claimed)
        self.assertIn("AgentB", claimed)


class AgentModelRegistryTests(unittest.TestCase):
    def test_agent_can_override_model_without_changing_global_default(self):
        settings = SimpleNamespace(
            ai_provider="ollama",
            ollama_model="default-model",
            openai_model="default-openai",
            ai_temperature=0.35,
            ai_max_tokens=512,
            agent_model_understanding_model="small-understanding-model",
            agent_model_understanding_provider="ollama",
            agent_model_safety_model="safety-model",
            agent_model_safety_provider="ollama",
        )

        registry = AgentModelRegistry(settings)

        self.assertEqual(registry.profile_for("UnderstandingAgent").model, "small-understanding-model")
        self.assertEqual(registry.profile_for("ResponseAgent").model, "default-model")


if __name__ == "__main__":
    unittest.main()
