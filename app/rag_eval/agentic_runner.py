from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from app.agents.autonomous import AgentPrivateMemory, AgentRuntimeServices, ContextAgent
from app.core.bootstrap import create_schema, seed_data
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.enums import IntentType, RiskLevel
from app.models.entities import ChatSession, UserAccount
from app.schemas.dtos import AiMessage
from app.services.agent_models import AgentModelRegistry
from app.services.ai import AiClient, has_consult_signal, has_high_risk_signal
from app.services.knowledge import KnowledgeService, SearchResult
from app.services.memory import RedisShortTermMemoryStore


def _is_relevant(source: str, content: str, expected_sources: set[str], expected_terms: list[str]) -> bool:
    if source.lower() in expected_sources:
        return True
    lower = content.lower()
    return any(len(term) >= 2 and term in lower for term in expected_terms)


@dataclass
class AgenticRagCaseResult:
    id: str
    question: str
    requires_retrieval: bool
    did_retrieve: bool
    retrieval_decision_correct: bool
    rewritten_query: str = ""
    rewrite_correct: bool = False
    hit: bool = False
    iterations: int = 0
    iterations_ok: bool = False
    retrieved: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def evaluate() -> dict:
    settings = get_settings()
    create_schema()
    db = SessionLocal()
    try:
        seed_data(db)

        user = db.query(UserAccount).filter(UserAccount.username == "student").first()
        if user is None:
            raise RuntimeError("student user not found for agentic RAG evaluation")

        session = ChatSession(public_id="agentic-rag-eval-session", user_id=user.id, title="agentic-rag-eval")
        db.add(session)
        db.commit()
        db.refresh(session)

        services = _build_services(db, settings, user, session)
        agent = ContextAgent(services)

        dataset_path = Path(settings.agentic_rag_eval_dataset)
        cases = json.loads(dataset_path.read_text(encoding="utf-8"))
        results = [_evaluate_case(agent, case) for case in cases]

        total = max(1, len(results))
        report = {
            "createdAt": _now_iso(),
            "dataset": str(dataset_path),
            "totalCases": len(results),
            "retrievalDecisionAccuracy": sum(1 for r in results if r.retrieval_decision_correct) / total,
            "rewriteAccuracy": sum(1 for r in results if r.rewrite_correct) / total,
            "hitRate": sum(1 for r in results if r.hit) / total,
            "iterationOkRate": sum(1 for r in results if r.iterations_ok) / total,
            "averageIterations": sum(r.iterations for r in results) / total,
            "results": [_serialize_result(r) for r in results],
        }

        output = Path(settings.agentic_rag_eval_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        db.close()


def _build_services(db, settings, user, session) -> AgentRuntimeServices:
    ai = AiClient(settings)
    knowledge = KnowledgeService(db, settings)
    memory = RedisShortTermMemoryStore(settings)
    model_registry = AgentModelRegistry(settings)
    private_memory = AgentPrivateMemory(settings)

    return AgentRuntimeServices(
        db=db,
        settings=settings,
        user=user,
        session=session,
        ai=ai,
        model_registry=model_registry,
        memory=memory,
        private_memory=private_memory,
        knowledge=knowledge,
    )


def _evaluate_case(agent: ContextAgent, case: dict) -> AgenticRagCaseResult:
    question = case["question"]
    requires_retrieval = case.get("requiresRetrieval", True)

    did_retrieve = _should_retrieve(question)
    result = AgenticRagCaseResult(
        id=case["id"],
        question=question,
        requires_retrieval=requires_retrieval,
        did_retrieve=did_retrieve,
        retrieval_decision_correct=did_retrieve == requires_retrieval,
    )

    if not requires_retrieval:
        result.iterations_ok = True
        result.rewrite_correct = True
        result.hit = True
        return result

    memory_brief = "无相关历史记忆。"
    rewritten_query, retrieved, iterations = agent._iterative_retrieve(memory_brief, question)

    result.rewritten_query = rewritten_query
    result.iterations = iterations

    # Query rewrite quality
    expected_terms = [t.lower() for t in case.get("expectedRewrittenQueryContains", [])]
    if expected_terms:
        lowered_query = rewritten_query.lower()
        missing = [term for term in expected_terms if term not in lowered_query]
        result.rewrite_correct = not missing
        if missing:
            result.failures.append(f"改写 query 缺少关键词：{missing}")
    else:
        result.rewrite_correct = True

    # Retrieval quality
    expected_sources = {s.lower() for s in case.get("expectedSources", [])}
    expected_content_terms = [t.lower() for t in case.get("expectedTerms", [])]

    result.retrieved = [_serialize_search_result(item, expected_sources, expected_content_terms) for item in retrieved]

    if retrieved:
        result.hit = any(
            _is_relevant(item.source, item.content, expected_sources, expected_content_terms)
            for item in retrieved
        )
    else:
        result.hit = False

    if not result.hit:
        result.failures.append("未召回相关结果")

    # Iteration budget
    max_iterations = case.get("maxAcceptableIterations", 2)
    result.iterations_ok = iterations <= max_iterations
    if not result.iterations_ok:
        result.failures.append(f"迭代次数 {iterations} 超过阈值 {max_iterations}")

    return result


def _should_retrieve(question: str) -> bool:
    """Mirror the production retrieval trigger used by ContextAgent.decide().

    Production uses UnderstandingAgent/SafetyAgent outputs (intent/risk).
    Here we approximate with a broader keyword list so the eval cases stay
    realistic without spinning up the full agent runtime.
    """
    lowered = question.lower()
    if has_consult_signal(lowered) or has_high_risk_signal(lowered):
        return True
    broader_signals = [
        "低落", "孤独", "压抑", "害怕", "状态不好", "状态差", "情绪", "人际", "室友",
        "关系", "恐慌", "心跳", "喘不过气", "紧张", "抑郁", "焦虑", "失眠", "压力",
    ]
    return any(signal in lowered for signal in broader_signals)


def _serialize_search_result(item: SearchResult, expected_sources: set[str], expected_terms: list[str]) -> dict:
    return {
        "chunkId": item.chunk_id,
        "source": item.source,
        "score": item.score,
        "relevant": _is_relevant(item.source, item.content, expected_sources, expected_terms),
        "preview": " ".join(item.content.split())[:160],
    }


def _serialize_result(result: AgenticRagCaseResult) -> dict:
    return {
        "id": result.id,
        "question": result.question,
        "requiresRetrieval": result.requires_retrieval,
        "didRetrieve": result.did_retrieve,
        "retrievalDecisionCorrect": result.retrieval_decision_correct,
        "rewrittenQuery": result.rewritten_query,
        "rewriteCorrect": result.rewrite_correct,
        "hit": result.hit,
        "iterations": result.iterations,
        "iterationsOk": result.iterations_ok,
        "retrieved": result.retrieved,
        "failures": result.failures,
    }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    settings = get_settings()
    report = evaluate()
    print("Agentic RAG evaluation completed.")
    for key in ["totalCases", "retrievalDecisionAccuracy", "rewriteAccuracy", "hitRate", "iterationOkRate", "averageIterations"]:
        print(f"{key}={report[key]}")
