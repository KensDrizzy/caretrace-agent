from __future__ import annotations

import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from app.agents.autonomous import CoordinatorAgent, _latest_review_for
from app.agents.events import (
    AgentEvent,
    AgentEventType,
    AgentTask,
    AgentTurnResult,
    CollaborationBlackboard,
    PRIORITY_ORDER,
    TaskPriority,
)
from app.agents.registry import AgentCapability, AgentCandidate, AgentRegistry
from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel


class EventDrivenCoordinator:
    """Claim-based coordinator.

    This class owns budgets and acceptance policy. It does not encode an agent
    chain; all worker execution comes from agents claiming open tasks.
    """

    _PARALLEL_AGENT_NAMES = frozenset({"UnderstandingAgent", "SafetyAgent"})

    def __init__(self, registry: AgentRegistry, coordinator_agent: CoordinatorAgent, settings: Settings):
        self.registry = registry
        self.coordinator_agent = coordinator_agent
        self.settings = settings
        self.max_rounds = int(getattr(settings, "agent_max_rounds", 8))
        self.max_claims_per_round = int(getattr(settings, "agent_max_claims_per_round", 4))
        self.max_claims_per_agent = int(getattr(settings, "agent_max_claims_per_agent", 3))
        self.final_min_confidence = float(getattr(settings, "agent_final_acceptance_min_confidence", 0.6))

    def run(self, board: CollaborationBlackboard) -> CollaborationBlackboard:
        board = self._ensure_root_task(board)
        claim_counts: dict[str, int] = defaultdict(int)
        for round_number in range(1, self.max_rounds + 1):
            board = replace(board, current_round=round_number)
            board = board.append_event(
                AgentEvent(
                    type=AgentEventType.ROUND_STARTED,
                    actor=self.coordinator_agent.name,
                    message=f"round={round_number}",
                    metadata={"round": round_number},
                    round=round_number,
                )
            )
            # 1. 根据黑板状态，推导还缺哪些任务
            board = self._derive_missing_work(board)
            # 2. 检查是否已经有可采纳的最终回复
            board = self._try_accept_final(board)
            if board.final_artifact_id:
                return board
            # 3. 让各个 Agent 决定去不去认领任务
            candidates, decision_events = self._claim_candidates(board, claim_counts)
            if not candidates:
                board = self._derive_missing_work(board, force_response=True)
                retry_candidates, retry_events = self._claim_candidates(board, claim_counts)
                candidates = retry_candidates
                decision_events = [*decision_events, *retry_events]
                if not candidates:
                    board = self._append_events(board, decision_events)
                    break
            # 决策事件在任务认领事件之前落板，保证 trace 时序可读
            board = self._append_events(board, decision_events)
            # 4. 执行被认领的任务：在这里调用 agent.act()
            executions = self._execute_candidate_acts(board, candidates)
            for index, (task, candidate) in enumerate(candidates):
                result, started_event, finished_event = executions[index]
                current_task = board.tasks.get(task.id, task)
                board = board.update_task(current_task.claim(candidate.agent.profile.name)).append_event(
                    AgentEvent(
                        type=AgentEventType.TASK_CLAIMED,
                        actor=candidate.agent.profile.name,
                        task_id=task.id,
                        message=candidate.decision.reason,
                        metadata={"confidence": candidate.decision.confidence},
                        round=round_number,
                    )
                )
                # 顺序保证 STARTED → (agent 自己的事件) → COMPLETED/FAILED
                board = board.append_event(started_event)
                board = board.apply_turn_result(current_task, candidate.agent.profile.name, result)
                board = board.append_event(finished_event)
                claim_counts[candidate.agent.profile.name] += 1
            board = self._derive_missing_work(board)
            board = self._try_accept_final(board)
            if board.final_artifact_id:
                return board
        return board.append_event(
            AgentEvent(
                type=AgentEventType.BUDGET_EXHAUSTED,
                actor=self.coordinator_agent.name,
                message="event-driven agent budget exhausted before final acceptance",
                round=board.current_round,
            )
        )

    @staticmethod
    def _append_events(board: CollaborationBlackboard, events: list[AgentEvent]) -> CollaborationBlackboard:
        for event in events:
            board = board.append_event(event)
        return board

    def _execute_candidate_acts(self, board: CollaborationBlackboard, candidates: list[tuple[AgentTask, AgentCandidate]]) -> list:
        if len(candidates) <= 1:
            return [self._act_candidate(board, task, candidate) for task, candidate in candidates]
        parallel: list[tuple[AgentTask, AgentCandidate]] = []
        sequential: list[tuple[AgentTask, AgentCandidate]] = []
        for task, candidate in candidates:
            if candidate.agent.profile.name in self._PARALLEL_AGENT_NAMES:
                parallel.append((task, candidate))
            else:
                sequential.append((task, candidate))

        results_by_key: dict[tuple[str, str],] = {}

        if len(parallel) > 1:
            with ThreadPoolExecutor(max_workers=len(parallel)) as executor:
                futures = {
                    executor.submit(self._act_candidate, board, task, candidate): (task, candidate)
                    for task, candidate in parallel
                }
                for future in futures:
                    task, candidate = futures[future]
                    results_by_key[(task.id, candidate.agent.profile.name)] = future.result()

        for task, candidate in sequential + (parallel if len(parallel) <= 1 else []):
            results_by_key[(task.id, candidate.agent.profile.name)] = self._act_candidate(board, task, candidate)

        return [results_by_key[(task.id, candidate.agent.profile.name)] for task, candidate in candidates]

    def _act_candidate(self, board: CollaborationBlackboard, task: AgentTask, candidate: AgentCandidate):
        current_task = board.tasks.get(task.id, task)
        agent_name = candidate.agent.profile.name
        input_artifact_ids = self._input_artifact_ids(board, current_task)
        started_event = AgentEvent(
            type=AgentEventType.AGENT_EXECUTION_STARTED,
            actor=agent_name,
            task_id=current_task.id,
            message=current_task.title,
            input_artifact_ids=input_artifact_ids,
            round=board.current_round,
        )
        started = time.perf_counter()
        try:
            result = candidate.agent.act(current_task, board)
        except Exception as exc:
            # 单个 Agent 执行失败不应拖垮整条链路：记录 FAILED 事件并让任务保持可重试
            duration_ms = (time.perf_counter() - started) * 1000.0
            failed_event = AgentEvent(
                type=AgentEventType.AGENT_EXECUTION_FAILED,
                actor=agent_name,
                task_id=current_task.id,
                message=str(exc)[:200],
                metadata={"errorType": type(exc).__name__, "errorMessage": str(exc)[:200]},
                duration_ms=duration_ms,
                input_artifact_ids=input_artifact_ids,
                round=board.current_round,
            )
            return AgentTurnResult(close_task=False), started_event, failed_event
        duration_ms = (time.perf_counter() - started) * 1000.0
        completed_event = AgentEvent(
            type=AgentEventType.AGENT_EXECUTION_COMPLETED,
            actor=agent_name,
            task_id=current_task.id,
            message=current_task.title,
            metadata={"status": "OK"},
            duration_ms=duration_ms,
            input_artifact_ids=input_artifact_ids,
            output_artifact_ids=tuple(artifact.id for artifact in result.artifacts),
            round=board.current_round,
        )
        return result, started_event, completed_event

    @staticmethod
    def _input_artifact_ids(board: CollaborationBlackboard, task: AgentTask) -> tuple[str, ...]:
        ids: list[str] = []
        for key in ("responseArtifactId", "revisionOf"):
            referenced = str(task.metadata.get(key) or "")
            if referenced:
                ids.append(referenced)
        latest_by_kind: dict[str, str] = {}
        for artifact in board.artifacts:
            latest_by_kind[artifact.kind] = artifact.id
        for artifact_id in latest_by_kind.values():
            if artifact_id not in ids:
                ids.append(artifact_id)
        return tuple(ids)

    def _ensure_root_task(self, board: CollaborationBlackboard) -> CollaborationBlackboard:
        if board.tasks:
            return board
        root = self.coordinator_agent.root_task(board)
        return board.add_task(root).append_event(
            AgentEvent(type=AgentEventType.TASK_CREATED, actor=self.coordinator_agent.name, task_id=root.id, message=root.title)
        )

    def _derive_missing_work(self, board: CollaborationBlackboard, force_response: bool = False) -> CollaborationBlackboard:
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="intent",
            task_id="task:understand",
            title="Understand user turn",
            capability=AgentCapability.UNDERSTANDING,
            priority=TaskPriority.HIGH,
            condition=board.user_input != "",
        )
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="risk",
            task_id="task:assess-safety",
            title="Assess safety risk",
            capability=AgentCapability.SAFETY,
            priority=TaskPriority.CRITICAL if _hard_high_risk(board.user_input) else TaskPriority.HIGH,
            condition=board.user_input != "",
        )
        intent = _intent_value(board)
        risk = _risk_value(board)
        needs_context = intent in {IntentType.CONSULT, IntentType.RISK} or risk in {RiskLevel.MEDIUM, RiskLevel.HIGH}
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="context",
            task_id="task:gather-context",
            title="Gather contextual evidence",
            capability=AgentCapability.CONTEXT,
            priority=TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.NORMAL,
            condition=needs_context,
        )
        has_response = board.latest_artifact("response_candidate") is not None
        can_request_response = force_response or (
            board.latest_artifact("intent") is not None
            and board.latest_artifact("risk") is not None
            and (not needs_context or board.latest_artifact("context") is not None or risk == RiskLevel.HIGH)
        )
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="response_candidate",
            task_id="task:propose-response",
            title="Propose candidate response",
            capability=AgentCapability.RESPONSE,
            priority=TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.HIGH,
            condition=can_request_response and not has_response,
        )
        response = board.latest_artifact("response_candidate")
        review = board.latest_artifact("safety_review")
        critique = board.latest_artifact("critique")
        # 被否决（critique）的候选也算"已审核"：不再为它派生审核任务，由修订任务接管，
        # 避免对同一旧候选反复重审、烧掉认领预算。
        if response and _latest_review_for(board, response.id) is None:
            board = self._ensure_task(
                board,
                AgentTask(
                    id=f"task:review-response:{response.id}",
                    title="Review candidate response safety",
                    description="Safety review is required before final acceptance.",
                    priority=TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.HIGH,
                    required_capabilities=frozenset({AgentCapability.SAFETY.value}),
                    created_by=self.coordinator_agent.name,
                    metadata={"kind": "safety_review", "responseArtifactId": response.id},
                ),
            )
        if critique and critique.payload.get("approved") is False:
            board = self._ensure_task(
                board,
                AgentTask(
                    id=f"task:revise-response:{critique.id}",
                    title="Revise response after critique",
                    description=str(critique.payload.get("reason", "Safety critique requested revision.")),
                    priority=TaskPriority.CRITICAL,
                    required_capabilities=frozenset({AgentCapability.RESPONSE.value}),
                    created_by=self.coordinator_agent.name,
                    metadata={"kind": "response", "revisionOf": critique.payload.get("responseArtifactId", "")},
                ),
            )
        return board

    def _ensure_task_for_missing_artifact(
        self,
        board: CollaborationBlackboard,
        artifact_kind: str,
        task_id: str,
        title: str,
        capability: AgentCapability,
        priority: TaskPriority,
        condition: bool,
    ) -> CollaborationBlackboard:
        if not condition or board.latest_artifact(artifact_kind) is not None:
            return board
        return self._ensure_task(
            board,
            AgentTask(
                id=task_id,
                title=title,
                description=board.user_input,
                priority=priority,
                required_capabilities=frozenset({capability.value}),
                created_by=self.coordinator_agent.name,
                metadata={"kind": artifact_kind},
            ),
        )

    def _ensure_task(self, board: CollaborationBlackboard, task: AgentTask) -> CollaborationBlackboard:
        if task.id in board.tasks:
            return board
        return board.add_task(task).append_event(
            AgentEvent(type=AgentEventType.TASK_CREATED, actor=self.coordinator_agent.name, task_id=task.id, message=task.title)
        )

    def _claim_candidates(self, board: CollaborationBlackboard, claim_counts: dict[str, int]):
        # 评估所有 agent 对所有开放任务的决定（含不认领的），生成 DECISION_EVALUATED 事件
        evaluations: list[tuple[AgentTask, AgentCandidate]] = []
        task_candidates = []
        for task in board.open_tasks():
            for candidate in self.registry.evaluate_decisions_for(task, board):
                evaluations.append((task, candidate))
                if not candidate.decision.claim:
                    continue
                if claim_counts[candidate.agent.profile.name] >= self.max_claims_per_agent:
                    continue
                task_candidates.append((task, candidate))
        task_candidates.sort(
            key=lambda item: (
                PRIORITY_ORDER[item[0].priority],
                item[1].decision.confidence,
                item[1].agent.profile.name,
            ),
            reverse=True,
        )
        seen = set()
        selected_agents = set()
        selected = []
        for task, candidate in task_candidates:
            key = (task.id, candidate.agent.profile.name)
            if key in seen or candidate.agent.profile.name in selected_agents:
                continue
            selected.append((task, candidate))
            seen.add(key)
            selected_agents.add(candidate.agent.profile.name)
            if len(selected) >= self.max_claims_per_round:
                break
        selected_keys = {(task.id, candidate.agent.profile.name) for task, candidate in selected}
        observed_artifact_ids = [artifact.id for artifact in board.artifacts]
        events: list[AgentEvent] = []
        for task, candidate in evaluations:
            decision = candidate.decision
            events.append(
                AgentEvent(
                    type=AgentEventType.DECISION_EVALUATED,
                    actor=candidate.agent.profile.name,
                    task_id=task.id,
                    message=decision.reason,
                    metadata={
                        "claim": decision.claim,
                        "confidence": decision.confidence,
                        "reasonCode": decision.reason_code,
                        "reason": decision.reason,
                        "requiredCapabilities": sorted(task.required_capabilities),
                        "observedArtifactIds": observed_artifact_ids,
                        "selected": (task.id, candidate.agent.profile.name) in selected_keys,
                    },
                    round=board.current_round,
                )
            )
        for task, candidate in selected:
            events.append(
                AgentEvent(
                    type=AgentEventType.CANDIDATE_SELECTED,
                    actor=candidate.agent.profile.name,
                    task_id=task.id,
                    message=task.title,
                    metadata={"confidence": candidate.decision.confidence, "round": board.current_round},
                    round=board.current_round,
                )
            )
        return selected, events

    def _try_accept_final(self, board: CollaborationBlackboard) -> CollaborationBlackboard:
        if board.final_artifact_id:
            return board
        response = board.latest_artifact("response_candidate")
        review = board.latest_artifact("safety_review")
        if response is None or review is None:
            return board
        if review.metadata.get("responseArtifactId") != response.id:
            return board
        if review.metadata.get("responseArtifactVersion") != response.version:
            return board
        if not review.payload.get("approved"):
            return board
        if response.confidence < self.final_min_confidence:
            return board
        reason = "accepted after autonomous response proposal and SafetyAgent approval"
        self.coordinator_agent.remember_acceptance(response.id, reason)
        return board.accept_final(response.id, self.coordinator_agent.name, reason)


def _intent_value(board: CollaborationBlackboard) -> IntentType:
    artifact = board.latest_artifact("intent")
    if artifact:
        try:
            return IntentType(str(artifact.payload.get("intent", IntentType.CHAT.value)).upper())
        except ValueError:
            return IntentType.CHAT
    if _hard_high_risk(board.user_input):
        return IntentType.RISK
    return IntentType.CHAT


def _risk_value(board: CollaborationBlackboard) -> RiskLevel:
    order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
    highest = RiskLevel.LOW
    for artifact in board.artifacts_by_kind("risk"):
        try:
            risk = RiskLevel(str(artifact.payload.get("risk", RiskLevel.LOW.value)).upper())
        except ValueError:
            risk = RiskLevel.LOW
        if order[risk] > order[highest]:
            highest = risk
    if any(event.type == AgentEventType.SAFETY_OVERRIDE for event in board.events):
        return RiskLevel.HIGH
    return highest


def _hard_high_risk(text: str) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in ["自杀", "自残", "不想活", "结束生命", "伤害自己", "轻生", "suicide", "kill myself", "self harm"])
