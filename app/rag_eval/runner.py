import json
import math
from app.core.timezone import now_cn
from pathlib import Path

from app.core.bootstrap import create_schema, seed_data
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.services.knowledge import KnowledgeService


def evaluate() -> dict:
    # 初始化数据库和知识库服务
    settings = get_settings()
    create_schema()
    db = SessionLocal()
    try:
        # 读取配置、创建数据库表，并打开数据库连接。
        seed_data(db)
        # 初始化测试数据和真正的 KnowledgeService。
        # 这里的 service.retrieve() 就是线上实际使用的检索流程.
        service = KnowledgeService(db, settings)
        # 加载评测数据集 app/rag_eval/mindbridge-rag-eval.json
        dataset_path = Path(settings.rag_eval_dataset)
        cases = json.loads(dataset_path.read_text(encoding="utf-8"))
        # 对每个 case 执行检索
        results = [evaluate_case(service, case, settings.knowledge_top_k) for case in cases]
        total = max(1, len(results))
        hits = [item for item in results if item["hit"]]
        report = {
            "createdAt": now_cn().isoformat(),
            "dataset": settings.rag_eval_dataset,
            "topK": settings.knowledge_top_k,
            "totalCases": len(results),
            "recallAtK": sum(item["recallAtK"] for item in results) / total,
            "precisionAtK": sum(item["precisionAtK"] for item in results) / total,
            "mrr": sum(item["reciprocalRank"] for item in results) / total,
            "ndcgAtK": sum(item["ndcgAtK"] for item in results) / total,
            "hitRate": len(hits) / total,
            "averageFirstRelevantRank": sum(item["firstRelevantRank"] for item in hits) / max(1, len(hits)),
            "results": results,
        }
        output = Path(settings.rag_eval_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        db.close()


def evaluate_case(service: KnowledgeService, case: dict, top_k: int) -> dict:
    # 直接调用真实 RAG 检索
    retrieved = service.retrieve(case["question"], top_k)
    # 读取期望答案条件
    expected_sources = {source.lower() for source in case.get("expectedSources", [])}
    expected_terms = [term.lower() for term in case.get("expectedTerms", [])]
    items = []
    first_rank = 0
    relevant_count = 0
    for index, item in enumerate(retrieved, start=1):
        # 判断单条结果是否相关
        relevant = is_relevant(item.source, item.content, expected_sources, expected_terms)
        if relevant:
            relevant_count += 1
            if first_rank == 0:
                first_rank = index
        items.append({
            "rank": index,
            "chunkId": item.chunk_id,
            "source": item.source,
            "score": item.score,
            "relevant": relevant,
            "preview": " ".join(item.content.split())[:160],
        })
    hit = first_rank > 0
    return {
        "id": case["id"],
        "question": case["question"],
        "expectedSources": case.get("expectedSources", []),
        "expectedTerms": case.get("expectedTerms", []),
        "retrieved": items,
        "hit": hit,
        "firstRelevantRank": first_rank,
        "recallAtK": 1.0 if hit else 0.0,
        "precisionAtK": relevant_count / top_k if top_k > 0 else 0.0,
        "reciprocalRank": 1.0 / first_rank if hit else 0.0,
        "ndcgAtK": ndcg(items),
    }


# 匹配规则：来源文件完全匹配，或者正文中包含任意一个 expectedTerm
def is_relevant(source: str, content: str, expected_sources: set[str], expected_terms: list[str]) -> bool:
    if source.lower() in expected_sources:
        return True
    lower = content.lower()
    return any(len(term) >= 2 and term in lower for term in expected_terms)


def ndcg(items: list[dict]) -> float:
    dcg = 0.0
    relevant = 0
    for index, item in enumerate(items):
        if item["relevant"]:
            relevant += 1
            dcg += 1.0 / math.log(index + 2.0)
    if relevant == 0:
        return 0.0
    ideal = sum(1.0 / math.log(index + 2.0) for index in range(relevant))
    return dcg / ideal


if __name__ == "__main__":
    report = evaluate()
    print("RAG evaluation completed.")
    for key in ["totalCases", "topK", "recallAtK", "precisionAtK", "mrr", "ndcgAtK", "hitRate", "averageFirstRelevantRank"]:
        print(f"{key}={report[key]}")

