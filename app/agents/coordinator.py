from __future__ import annotations

import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from app.agents.autonomous import CoordinatorAgent
from app.agents.events import (
    AgentEvent,
    AgentEventType,
    AgentTask,
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
        # board = self._ensure_root_task(board)
        # 在本次总任务开始执行前，要加入root task，作为总任务的根节点。
        board = self._ensure_root_task(board)
        claim_counts: dict[str, int] = defaultdict(int)
        # 一个总任务里面，可能会有好几轮调度，每轮调度开始之前先加入ROUND_STARTED
        for round_number in range(1, self.max_rounds + 1):
            board = board.append_event(
                AgentEvent(
                    type=AgentEventType.ROUND_STARTED,
                    actor=self.coordinator_agent.name,
                    message=f"round={round_number}",
                    metadata={"round": round_number},
                )
            )
            # 1. 它根据当前黑板上有没有 artifact，创建需要的任务
            # 根据 Blackboard 当前缺了什么 artifact，推导现在应该有哪些任务。
            board = self._derive_missing_work(board)
            # 2. 检查是否已经有可采纳的最终回复
            board = self._try_accept_final(board)
            if board.final_artifact_id:
                return board
            # 3. 实际上是调用registry里面的registry.evaluate_decisions_for，
            # 让各个 Agent 决定去不去认领任务
            # 返回 (task, candidate)
            # task                  # 要执行的任务
            # candidate.agent       # 实际执行任务的 Agent
            # candidate.decision    # 该 Agent 的认领决定
            candidates = self._claim_candidates(board, claim_counts)
            # 如果没有任何 Agent 愿意认领任务，尝试强制生成回复任务 force_response=True
            if not candidates:
                board = self._derive_missing_work(board, force_response=True)
                candidates = self._claim_candidates(board, claim_counts)
                if not candidates:
                    break
            # 4. 执行被认领的任务：在这里调用 agent.act()
            results = self._execute_candidate_acts(board, candidates)
            # 这一步的结果还是局部变量，黑板还没有被写入新的 artifact。
            # 部分 Agent 可能并行执行，不能让多个线程同时直接修改同一个黑板。
            # 所以for循环里面，拿到当前候选对应的原 task，用来：确认这个结果属于哪个 task,标记 task 已被哪个 Agent 认领,记录 task 相关事件
            for index, (task, candidate) in enumerate(candidates):
                # 把 task 标记为已认领
                current_task = board.tasks.get(task.id, task)
                # 更新task的状态，把这个新 Task 塞回 Blackboard。  然后往board.events里追加一个事件，表示这个任务被认领了。
                board = board.update_task(current_task.claim(candidate.agent.profile.name)).append_event(
                    AgentEvent(
                        type=AgentEventType.TASK_CLAIMED,
                        actor=candidate.agent.profile.name,
                        task_id=task.id,
                        message=candidate.decision.reason,
                        metadata={"confidence": candidate.decision.confidence},
                    )
                )
                board = board.apply_turn_result(current_task, candidate.agent.profile.name, results[index])
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
            )
        )

# Agent 执行 task
#   ↓
# 得到临时的 AgentTurnResult
#   ↓
# Coordinator 把 result 拆开
#   ↓
# 分别写入黑板的 artifacts、messages、tasks、events
    def _execute_candidate_acts(self, board: CollaborationBlackboard, candidates: list[tuple[AgentTask, AgentCandidate]]) -> list:
        # 只有一个候选agent，直接调用
        if len(candidates) <= 1:
            return [self._act_candidate(board, task, candidate) for task, candidate in candidates]

        parallel: list[tuple[AgentTask, AgentCandidate]] = []
        sequential: list[tuple[AgentTask, AgentCandidate]] = []
        for task, candidate in candidates:
            # 如果是UnderstandingAgent、SafetyAgent → 他们两个可以并行
            if candidate.agent.profile.name in self._PARALLEL_AGENT_NAMES:
                parallel.append((task, candidate))
            else:
                # 但是ContextAgent、ResponseAgent  → 只能顺序执行
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
            # 它的 key 是：(task.id, candidate.agent.profile.name)
            # value 是 _act_candidate() 返回的三元组：
            # (
            #     result,
            #     started_event,
            #     finished_event,
            # )
            # 其中的result：AgentTurnResultmessages：Agent 发给其他 Agent 的消息
            # artifacts：Agent 产出的结构化结果
            # tasks：Agent 追加创建的后续任务
            # events：Agent 执行过程中产生的事件
            # close_task：是否关闭当前 task
            # started_event：AGENT_EXECUTION_STARTED
            # finished_event：成功时是 AGENT_EXECUTION_COMPLETED，失败时是 AGENT_EXECUTION_FAILED
            results_by_key[(task.id, candidate.agent.profile.name)] = self._act_candidate(board, task, candidate)

        return [results_by_key[(task.id, candidate.agent.profile.name)] for task, candidate in candidates]

    def _act_candidate(self, board: CollaborationBlackboard, task: AgentTask, candidate: AgentCandidate):
        current_task = board.tasks.get(task.id, task)
        return candidate.agent.act(current_task, board)

    def _ensure_root_task(self, board: CollaborationBlackboard) -> CollaborationBlackboard:
        if board.tasks:
            return board
        root = self.coordinator_agent.root_task(board)
        return board.add_task(root).append_event(
            AgentEvent(type=AgentEventType.TASK_CREATED, actor=self.coordinator_agent.name, task_id=root.id, message=root.title)
        )
# 根据黑板当前已有的 artifact，判断还缺哪些工作，并创建对应的任务。
    def _derive_missing_work(self, board: CollaborationBlackboard, force_response: bool = False) -> CollaborationBlackboard:
        # 有用户输入
        # 并且还没有 intent artifact 【在里面做了判断，如果已经有这个artifact的话就不用创建了】
        # → 创建“理解用户意图”任务
        # → 之后由 UnderstandingAgent 认领
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="intent",
            task_id="task:understand",
            title="Understand user turn",
            capability=AgentCapability.UNDERSTANDING,
            priority=TaskPriority.HIGH,
            condition=board.user_input != "",
        )
        # 有用户输入
        # 并且还没有 risk artifact
        # → 创建“评估安全风险”任务
        # → 之后由 SafetyAgent 认领
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="risk",
            task_id="task:assess-safety",
            title="Assess safety risk",
            capability=AgentCapability.SAFETY,
            priority=TaskPriority.CRITICAL if _hard_high_risk(board.user_input) else TaskPriority.HIGH,
            condition=board.user_input != "",
        )
        # 读取当前意图和风险
        intent = _intent_value(board)
        risk = _risk_value(board)
        # 根据 intent 和 risk 的值，判断是否需要上下文
        # intent 是 CONSULT 或 RISK → 需要上下文；
        # 或者 risk 是 MEDIUM 或 HIGH → 需要上下文；
        # 只要任意一个条件成立，needs_context 就是 True。
        needs_context = intent in {IntentType.CONSULT, IntentType.RISK} or risk in {RiskLevel.MEDIUM, RiskLevel.HIGH}
        # 创建收集上下文任务
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="context",
            task_id="task:gather-context",
            title="Gather contextual evidence",
            capability=AgentCapability.CONTEXT,
            priority=TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.NORMAL,
            condition=needs_context,
        )
        # 判断是否已经有回复方案：检查黑板中是否已经存在最新的 response_proposal 产物。
        # 这里的 response_proposal 可以理解为：
        # Agent 提出的候选回复，但还没有经过最终安全审核。
        has_response = board.latest_artifact("response_proposal") is not None
        # 判断是否可以生成回复
        # case1:强制创建，这通常用于系统已经没有可认领任务时，强制推动流程继续。
        # case2:有意图和风险分析结果，
        # 【1.如果不需要上下文，可以继续；
        # 2.如果需要上下文，但已经有上下文 artifact，可以继续；
        # 3.如果风险是 HIGH，即使没有上下文，也可以继续。】，可以生成回复。
        can_request_response = force_response or (
            board.latest_artifact("intent") is not None
            and board.latest_artifact("risk") is not None
            and (not needs_context or board.latest_artifact("context") is not None or risk == RiskLevel.HIGH)
        )
        # 创建候选回复任务
        board = self._ensure_task_for_missing_artifact(
            board,
            artifact_kind="response_proposal",
            task_id="task:propose-response",
            title="Propose candidate response",
            capability=AgentCapability.RESPONSE,
            priority=TaskPriority.CRITICAL if risk == RiskLevel.HIGH else TaskPriority.HIGH,
            condition=can_request_response and not has_response,
        )
        # 获取回复、审核和批评结果
        response = board.latest_artifact("response_proposal")
        review = board.latest_artifact("safety_review")
        critique = board.latest_artifact("critique")
        if response and (review is None or review.metadata.get("responseArtifactId") != response.id):
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
        # 为被否决的回复创建修订任务
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

# 如果 task:understand 已经创建过
# → 不重复创建

# 如果还没创建
# → 放进 board.tasks
# → 追加 TASK_CREATED 事件  因为task:understand是首要任务
    def _ensure_task(self, board: CollaborationBlackboard, task: AgentTask) -> CollaborationBlackboard:
        if task.id in board.tasks:
            return board
        return board.add_task(task).append_event(
            AgentEvent(type=AgentEventType.TASK_CREATED, actor=self.coordinator_agent.name, task_id=task.id, message=task.title)
        )
    # 遍历所有 open tasks，调用每个 Agent 的 decide()
    # candidate_decisions_for 会调用 agent.decide(task, board)。
    # 如果 Agent 返回 AgentDecision(True, ...)，就表示它愿意认领这个任务。
    # 从所有开放任务中，筛选出愿意认领任务的 Agent，并按照优先级、置信度选择本轮实际执行的任务。
    def _claim_candidates(self, board: CollaborationBlackboard, claim_counts: dict[str, int]):
        selected = [] # 最终选中、本轮真正执行的组合。
        task_candidates = [] # 所有符合条件的“任务-Agent”组合。
        # 遍历所有开放任务，获取愿意认领任务的 Agent
        for task in board.open_tasks():
            # 若agent返回decision.claim == True，代表认领
            for candidate in self.registry.candidate_decisions_for(task, board):
                # 如果它已经达到最大认领次数，就跳过该候选者。
                if claim_counts[candidate.agent.profile.name] >= self.max_claims_per_agent:
                    continue
                task_candidates.append((task, candidate)) # 符合条件的候选者加入列表 (任务, Agent候选者)
        # 按照(任务优先级, Agent 置信度, Agent 名称)排序
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
        # 选择本轮执行者
        for task, candidate in task_candidates:
            key = (task.id, candidate.agent.profile.name) # 类似("task:generate-response", "ResponseAgent")
            # 跳过两种情况：
            # 同一个任务和同一个 Agent 已经处理过；
            # 该 Agent 本轮已经被选中过。
            # 第二条意味着：同一个 Agent 在同一轮最多执行一个任务。
            if key in seen or candidate.agent.profile.name in selected_agents:
                continue
            selected.append((task, candidate))
            seen.add(key)
            selected_agents.add(candidate.agent.profile.name)
            if len(selected) >= self.max_claims_per_round:
                break
        # 最终返回
        # [
        #     (task1, candidate1),
        #     (task2, candidate2),
        # ]
        return selected

    def _try_accept_final(self, board: CollaborationBlackboard) -> CollaborationBlackboard:
        if board.final_artifact_id:
            return board
        response = board.latest_artifact("response_proposal")
        review = board.latest_artifact("safety_review")
        if response is None or review is None:
            return board
        if review.metadata.get("responseArtifactId") != response.id:
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
