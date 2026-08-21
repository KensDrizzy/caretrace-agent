from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Hashable

from pypdf import PdfReader
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.entities import KnowledgeChunk
from app.services.vector_store import FALLBACK_RETRIEVAL_LABEL, PRIMARY_RETRIEVAL_LABEL, ChromaKnowledgeStore


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: int | None
    source: str
    content: str
    score: float


@dataclass
class RetrievalCandidate:
    result: SearchResult
    vector_score: float = 0.0
    bm25_score: float = 0.0


class KnowledgeService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.vector_store = ChromaKnowledgeStore(settings)

    def count(self) -> int:
        return self.db.query(KnowledgeChunk).count()

# 作用是：
# 文档没变化：不重新切块、不重新请求 Embedding。
# 文档改变任何一个字符：删除该来源的旧数据，重新完整入库。
# 它是“文档级增量判断”，不是“chunk 级差异更新”。
    def ensure_source(self, source: str, content: str) -> int:
        # 对整篇文档算 Hash，原文只要改动一个字，SHA-256都会变化
        content_hash = _hash_content(content)
        # Fast path: if the source hash matches, the document is unchanged.
        # 查询当前source在数据库里面对应的hash
        stored_hash = (
            self.db.query(KnowledgeChunk.content_hash)
            .filter(KnowledgeChunk.source == source)
            .limit(1)
            .scalar()
        )
        # 如果相同，就跳过ingest
        if stored_hash == content_hash:
            return (
                self.db.query(KnowledgeChunk)
                .filter(KnowledgeChunk.source == source)
                .count()
            )
        # 如果不相同，就重新ingest
        return self.ingest(source, content, content_hash)

    def status(self) -> dict:
        vector_chunks = None
        vector_error = getattr(self.vector_store, "error", "")
        if self.vector_store.can_embed:
            try:
                vector_chunks = self.vector_store.count()
            except Exception as exc:
                vector_error = f"{type(exc).__name__}: {exc}"
        return {
            "retrievalOrder": [
                PRIMARY_RETRIEVAL_LABEL,
                f"{FALLBACK_RETRIEVAL_LABEL} when OPENAI_API_KEY/chromadb/vector call is unavailable",
            ],
            "primaryRetrieval": PRIMARY_RETRIEVAL_LABEL,
            "fallbackRetrieval": FALLBACK_RETRIEVAL_LABEL,
            "databaseChunks": self.count(),
            "vectorEnabled": self.settings.knowledge_vector_enabled,
            "vectorAvailable": self.vector_store.ready,
            "vectorNeedsRebuild": self.vector_store.needs_rebuild,
            "vectorRequired": self.settings.knowledge_vector_required,
            "embeddingModel": self.settings.resolved_embedding_model,
            "embeddingBaseUrl": self.settings.resolved_embedding_base_url,
            "vectorChunks": vector_chunks,
            "chromaPersistDir": self.settings.chroma_persist_dir,
            "chromaCollectionName": self.settings.chroma_collection_name,
            "chromaSnapshotDir": self.settings.chroma_snapshot_dir,
            "candidateK": self.settings.knowledge_candidate_k,
            "hybridVectorWeight": self.settings.knowledge_hybrid_vector_weight,
            "hybridBm25Weight": self.settings.knowledge_hybrid_bm25_weight,
            "rerankEnabled": self.settings.knowledge_rerank_enabled,
            "vectorError": vector_error,
        }

    def rebuild_vector_index(self) -> int:
        if not self.vector_store.can_embed:
            raise RuntimeError(getattr(self.vector_store, "error", "") or "Chroma 向量库不可用")
        rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.source.asc(), KnowledgeChunk.source_index.asc()).all()
        # 模型切换后必须丢弃旧 collection，并强制重新生成 embedding；
        # 仅检查 embedding_json 是否存在会错误复用旧模型向量。
        self.vector_store.reset_collection()
        self._sync_vector_chunks(rows, force=True)
        self.db.commit()
        return len(rows)

    def backup_vector_index(self) -> str:
        if not self.vector_store.can_embed:
            raise RuntimeError(getattr(self.vector_store, "error", "") or "Chroma 向量库不可用")
        snapshot = self.vector_store.snapshot()
        if snapshot is None:
            raise RuntimeError("Chroma 持久化目录不存在，无法生成快照")
        return snapshot

    def ingest(self, source: str, content: str, content_hash: str | None = None) -> int:
        content_hash = content_hash or _hash_content(content)
        # 重新切块
        # 返回chunks列表，此时这些 chunk 只是存在于内存中，还没有保存到数据库或 Chroma。
        chunks = chunk_text(content, self.settings.knowledge_chunk_size, self.settings.knowledge_chunk_overlap)
        # 删除 Chroma 中这个文档以前的向量。
        self._delete_vector_source(source)
        # 删除 MySQL 里的旧 chunks。
        # 1.SQLAlchemy 会把：self.db.query(KnowledgeChunk)
        # 转换成类似 SQL：SELECT * FROM knowledge_chunks; 再用filter去做一些条件查询限定
        # self.db 是 SQLAlchemy 的数据库会话，也就是当前数据库连接和事务的管理对象。
        self.db.query(KnowledgeChunk).filter(KnowledgeChunk.source == source).delete()
        rows = []
        # 然后创建新的数据库对象
        for index, chunk in enumerate(chunks):
            # 第一行只是创建 Python 对象，此时它还没有真正写入数据库。
            row = KnowledgeChunk(source=source, source_index=index, content=chunk, content_hash=content_hash)
            # 把这个对象放入 SQLAlchemy 当前 Session，状态类似：待插入数据库。但这时通常还没有执行真正的 SQL。
            self.db.add(row)
            rows.append(row)
        # flush() 会把 INSERT 发送给数据库，但还没有最终提交。它的重要作用是让数据库生成每个 chunk 的 id。
        # 具体：会让 SQLAlchemy 把当前 Session 中待插入的对象发送给数据库。数据库执行插入时，会为每一行生成主键id，然后 SQLAlchemy 会把这些主键同步回 Python 对象。
        # 把 SQL 发给数据库，并拿到自动生成的 ID，但事务还没有最终确认。如果后面出错，事务仍然可以回滚。
        self.db.flush()
        # 重新生成/写入向量索引：使用这些已经有 id 的 chunk 生成 embedding，并写入 Chroma。
        self._index_vector_chunks(rows)
        # 正式提交事务，其他数据库连接可以看到这些数据。
        self.db.commit()
        return len(chunks)

    def ingest_file(self, filename: str, data: bytes) -> int:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            text = extract_pdf(data)
        else:
            text = data.decode("utf-8", errors="ignore")
        return self.ingest(filename, text)

# 向量检索 + BM25 关键词检索
#        ↓
# 结果融合
#         ↓
# 本地重排序
#         ↓
# 扩展最佳结果上下文
    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        # 向量检索和 BM25 先各找最多 16 条候选
        # 最后融合、排序后只保留 4 条
        top_k = top_k or self.settings.knowledge_top_k
        candidate_k = self._candidate_k(top_k)

        # Primary retrieval uses hybrid recall: semantic vector candidates plus
        # BM25 keyword candidates, followed by deterministic local rerank.
        # BM25 now works on a keyword-pre-filtered candidate set instead of the
        # full table, keeping latency flat as the knowledge base grows.

        # 向量检索
        vector_results = self._retrieve_vector(query, candidate_k)
        # BM25 关键词检索
        bm25_results = self._retrieve_bm25(query, candidate_k)
        # 融合两种检索结果，返回topk个候选
        ranked = self._fuse_and_rerank(query, vector_results, bm25_results, top_k)
        if ranked:
            # 比如最后rerank返回topk=4，
            # 然后取top1的chunk，从同一来源中找它前后相邻的知识块，拼接成一个SearchResult，最后还是返回topk=4，只是对top1进行expansion
            return self._expand_best(ranked, top_k)
        return []

    def _retrieve_bm25(self, query: str, top_k: int) -> list[SearchResult]:
        chunks = self._fetch_bm25_candidates(query)
        scores = bm25_scores(query, chunks)
        ranked = [
            SearchResult(chunk.id, chunk.source, chunk.content, scores.get(chunk.id, 0.0))
            for chunk in chunks
            if chunk.id is not None and scores.get(chunk.id, 0.0) > 0
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    def _fetch_bm25_candidates(self, query: str) -> list[KnowledgeChunk]:
        """Return a bounded set of chunks that contain at least one query term.

        Falls back to the newest chunks when no textual terms are present.
        """
        max_docs = max(50, self.settings.knowledge_bm25_max_docs)
        terms = _bm25_query_terms(query)
        q = self.db.query(KnowledgeChunk)
        if terms:
            q = q.filter(or_(*(KnowledgeChunk.content.ilike(f"%{term}%") for term in terms)))
        else:
            # No textual terms to pre-filter; fall back to the most recent chunks.
            q = q.order_by(KnowledgeChunk.id.desc())
        return q.limit(max_docs).all()

    def _fuse_and_rerank(
        self,
        query: str,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        candidates: dict[Hashable, RetrievalCandidate] = {}
        vector_scores = {result_key(item): item.score for item in vector_results if item.score > 0}
        bm25_scores_by_key = {result_key(item): item.score for item in bm25_results if item.score > 0}
        # 分数归一化
        normalized_vector = normalize_scores(vector_scores)
        normalized_bm25 = normalize_scores(bm25_scores_by_key)

        for item in [*vector_results, *bm25_results]:
            key = result_key(item)
            candidate = candidates.get(key)
            if candidate is None:
                candidate = RetrievalCandidate(result=item)
                candidates[key] = candidate
            candidate.vector_score = max(candidate.vector_score, normalized_vector.get(key, 0.0))
            candidate.bm25_score = max(candidate.bm25_score, normalized_bm25.get(key, 0.0))

        if not candidates:
            return []
        # 给两种检索方式加权 vector_weight = 0.65, bm25_weight = 0.35
        vector_weight = max(0.0, self.settings.knowledge_hybrid_vector_weight) if vector_results else 0.0
        bm25_weight = max(0.0, self.settings.knowledge_hybrid_bm25_weight)
        if vector_weight == 0.0 and bm25_weight == 0.0:
            bm25_weight = 1.0
        total_weight = vector_weight + bm25_weight

        fused = []
        for candidate in candidates.values():
            score = (
                candidate.vector_score * vector_weight
                + candidate.bm25_score * bm25_weight
            ) / total_weight
            fused.append(replace_score(candidate.result, score))

        fused.sort(key=lambda item: item.score, reverse=True)
        fused = fused[:self._candidate_k(top_k)]
        # 本地重排序
        # return (
        #     base_score * 0.55 向量和 BM25 的融合分数
        #     + lexical * 0.25 词汇相似度
        #     + coverage * 0.15 query 关键词覆盖率
        #     + phrase * 0.05 完整短语匹配程度
        # )
        return self._rerank(query, fused, top_k)

    def _rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not self.settings.knowledge_rerank_enabled:
            return candidates[:top_k]
        reranked = [
            replace_score(item, rerank_score(query, item.content, item.score))
            for item in candidates
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    def _candidate_k(self, top_k: int) -> int:
        return max(top_k, self.settings.knowledge_candidate_k)

    def _retrieve_vector(self, query: str, top_k: int) -> list[SearchResult]:
        # 判断向量检索是否可用
        if not self.vector_store.ready:
            return []
        try:
            # 确保向量索引是最新的，这个方法会检查 MySQL 中的 KnowledgeChunk 是否已经同步到 Chroma。
            self._ensure_vector_index()
            # 把查询的query文本转换成向量
            query_embedding = self.vector_store.embed_texts([query])[0]
            # 在 Chroma 中搜索相似向量
            hits = self.vector_store.query(query_embedding, top_k)
        # 向量检索失败时的处理
        except Exception as exc:
            self._handle_vector_error("retrieve", exc)
            return []
        results = []
        # 根据命中的 chunk_id 回查 MySQL
        # Chroma 返回的是向量库中的命中记录，但 MySQL 是知识块的主数据来源。
        # 需要根据chroma里面查出来的chunk_id去MySQL里面获取来源、正文、chunk信息
        for hit in hits:
            chunk = self.db.get(KnowledgeChunk, hit.chunk_id) if hit.chunk_id is not None else None
            # 组装统一的 SearchResult
            results.append(
                SearchResult(
                    chunk.id if chunk is not None else hit.chunk_id,
                    chunk.source if chunk is not None else hit.source,
                    chunk.content if chunk is not None else hit.content,
                    hit.score,
                )
            )
        return results

    def _ensure_vector_index(self) -> None:
        if not self.vector_store.ready:
            return
        rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.source.asc(), KnowledgeChunk.source_index.asc()).all()
        if not rows:
            return
        # Fast path: if counts match and every DB row already has a cached
        # embedding, assume the vector collection is in sync. The expensive
        # exact-ID check is skipped here; use rebuild_vector_index() to force
        # a full consistency repair when needed.
        if self.vector_store.count() == len(rows) and all(row.embedding_json for row in rows):
            return
        self._sync_vector_chunks(rows)
        self.db.commit()

    def _delete_vector_source(self, source: str) -> None:
        if not self.vector_store.ready:
            return
        try:
            self.vector_store.delete_source(source)
        except Exception as exc:
            self._handle_vector_error("delete_source", exc)

    def _index_vector_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        # not self.vector_store.ready表示当前向量检索功能不可用。直接返回
        if not chunks or not self.vector_store.ready:
            return
        try:
            # 得到所有chunk的向量
            embeddings = self._embeddings_for_chunks(chunks)
            # 把向量转成json字符串
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_json = json.dumps(embedding, separators=(",", ":"))
            # 然后把chunk和对应的向量写入Chroma向量库。  具体upsert到chroma，会传chunk.id, chunk.content, metadatas[db_id,source,souce_index], embeddings
            self.vector_store.upsert_chunks(chunks, embeddings)
        except Exception as exc:
            self._handle_vector_error("index", exc)

    def _sync_vector_chunks(self, chunks: list[KnowledgeChunk], force: bool = False) -> None:
        if not chunks or not self.vector_store.ready:
            return
        try:
            embeddings = self._embeddings_for_chunks(chunks, force=force)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_json = json.dumps(embedding, separators=(",", ":"))
            self.vector_store.sync_chunks(chunks, embeddings)
        except Exception as exc:
            if force:
                self.vector_store.needs_rebuild = True
                self.vector_store.error = f"向量索引重建失败：{type(exc).__name__}: {exc}"
            self._handle_vector_error("sync", exc)
            if force:
                raise

    def _embeddings_for_chunks(self, chunks: list[KnowledgeChunk], force: bool = False) -> list[list[float]]:
        embeddings: list[list[float] | None] = []
        missing_indexes = []
        missing_texts = []
        for index, chunk in enumerate(chunks):
            # 尝试读取缓存，找到已经向量化过的chunk的向量
            embedding = None if force else parse_embedding(chunk.embedding_json)
            embeddings.append(embedding)
            # 如果当前chunk缺失向量，就记录下来
            if embedding is None:
                missing_indexes.append(index)
                missing_texts.append(chunk.content)
        # 只对缺失的chunk进行向量化
        if missing_texts:
            # 批量请求向量化
            new_embeddings = self.vector_store.embed_texts(missing_texts)
            # 把新向量放到对应的原来的位置
            for index, embedding in zip(missing_indexes, new_embeddings):
                embeddings[index] = embedding
        resolved = [embedding for embedding in embeddings if embedding is not None]
        # 检查有向量的chunk数量是否一致
        if len(resolved) != len(chunks):
            raise ValueError("Embedding response count did not match knowledge chunks.")
        return resolved

    def _handle_vector_error(self, action: str, exc: Exception) -> None:
        if self.settings.knowledge_vector_required:
            raise exc
        logger.warning(
            "%s %s failed; falling back to %s: %s",
            PRIMARY_RETRIEVAL_LABEL,
            action,
            FALLBACK_RETRIEVAL_LABEL,
            exc,
        )

    def _expand_best(self, ranked: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not ranked:
            return []
        best = ranked[0]
        expanded = self._expand(best)
        results = [expanded]
        for item in ranked[1:]:
            if item.chunk_id != expanded.chunk_id and len(results) < top_k:
                results.append(item)
        return results

    def _expand(self, result: SearchResult) -> SearchResult:
        if result.chunk_id is None:
            return result
        chunk = self.db.get(KnowledgeChunk, result.chunk_id)
        if chunk is None:
            return result
        neighbors = (
            self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.source == chunk.source)
            .filter(KnowledgeChunk.source_index >= max(0, chunk.source_index - 1))
            .filter(KnowledgeChunk.source_index <= chunk.source_index + 1)
            .order_by(KnowledgeChunk.source_index.asc())
            .all()
        )
        return SearchResult(chunk.id, chunk.source, "\n\n".join(item.content for item in neighbors), result.score)


def chunk_text(content: str, size: int, overlap: int) -> list[str]:
    #  作用是把连续的空白字符统一成一个空格。
    text = re.sub(r"\s+", " ", content or "").strip()
    #  如果文本为空，直接返回
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def hybrid_score(query: str, content: str) -> float:
    return token_cosine(query, content) * 0.75 + keyword_score(query, content) * 0.25


def bm25_scores(query: str, chunks: list[KnowledgeChunk]) -> dict[int, float]:
    query_terms = counts(tokenize(query))
    if not query_terms or not chunks:
        return {}

    documents = []
    doc_freqs: dict[str, int] = {}
    for chunk in chunks:
        if chunk.id is None:
            continue
        token_counts = counts(tokenize(chunk.content))
        documents.append((chunk.id, token_counts, sum(token_counts.values())))
        for term in token_counts:
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    total_docs = len(documents)
    if total_docs == 0:
        return {}
    average_length = sum(length for _, _, length in documents) / total_docs or 1.0
    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = {}

    for chunk_id, token_counts, doc_length in documents:
        score = 0.0
        length_norm = k1 * (1.0 - b + b * doc_length / average_length)
        for term, query_frequency in query_terms.items():
            term_frequency = token_counts.get(term, 0)
            if term_frequency == 0:
                continue
            doc_frequency = doc_freqs.get(term, 0)
            idf = math.log(1.0 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            query_boost = 1.0 + math.log(query_frequency)
            score += idf * query_boost * (term_frequency * (k1 + 1.0)) / (term_frequency + length_norm)
        if score > 0:
            scores[chunk_id] = score
    return scores

# 对已经检索出来的候选知识片段重新打分，综合语义相似度、关键词匹配和短语匹配，最后重新排序。
def rerank_score(query: str, content: str, base_score: float) -> float:
    #     base_score * 0.55 向量和 BM25 的融合分数
    #     + lexical * 0.25 词汇相似度
    #     + coverage * 0.15 query 关键词覆盖率
    #     + phrase * 0.05 完整短语匹配程度
    lexical = hybrid_score(query, content)
    coverage = query_token_coverage(query, content)
    phrase = phrase_score(query, content)
    return base_score * 0.55 + lexical * 0.25 + coverage * 0.15 + phrase * 0.05

# 计算 query 覆盖率    计算query_token和content_token的交集
# query 中有多少不同的词，
# 在 content 中也出现了
def query_token_coverage(query: str, content: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(content))
    return len(query_tokens & content_tokens) / len(query_tokens)

# 计算完整短语匹配度
def phrase_score(query: str, content: str) -> float:
    normalized_query = compact_text(query)
    if not normalized_query:
        return 0.0
    normalized_content = compact_text(content) # 转成小写，删除空白字符。
    # 如果整个 query 去掉空格后，完整出现在 content 中，就给满分 1。
    if normalized_query in normalized_content:
        return 1.0
    return keyword_score(query, content)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def normalize_scores(scores: dict[Hashable, float]) -> dict[Hashable, float]:
    positives = [score for score in scores.values() if score > 0]
    if not positives:
        return {key: 0.0 for key in scores}
    lowest = min(positives)
    highest = max(positives)
    if math.isclose(lowest, highest):
        return {key: 1.0 if score > 0 else 0.0 for key, score in scores.items()}
    return {
        key: (score - lowest) / (highest - lowest) if score > 0 else 0.0
        for key, score in scores.items()
    }


def result_key(result: SearchResult) -> Hashable:
    return result.chunk_id if result.chunk_id is not None else (result.source, result.content)


def replace_score(result: SearchResult, score: float) -> SearchResult:
    return SearchResult(result.chunk_id, result.source, result.content, score)


def parse_embedding(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(item, (int, float)) for item in data):
        return None
    return [float(item) for item in data]


# tokenize() 会提取：
# 英文单词；
# 数字；
# 中文单字；
# 连续中文二元词组。
def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    grams = words[:]
    compact = "".join(ch for ch in text.lower() if "\u4e00" <= ch <= "\u9fff")
    grams.extend(compact[i:i + 2] for i in range(max(0, len(compact) - 1)))
    return [item for item in grams if item.strip()]


# 它把 query 和 content 分词，再计算词频向量的余弦相似度。
def token_cosine(left: str, right: str) -> float:
    # 分词并统计词频，counts() 会统计每个 token 出现了几次。
    left_counts = counts(tokenize(left))
    right_counts = counts(tokenize(right))
    # 如果任意一边没有词，返回 0
    if not left_counts or not right_counts:
        return 0.0
    # 计算向量点积
    dot = sum(value * right_counts.get(key, 0) for key, value in left_counts.items())
    # 计算左向量长度
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    # 计算右向量长度
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    # 计算余弦相似度
    return 0.0 if left_norm == 0 or right_norm == 0 else dot / (left_norm * right_norm)


# 它把 query 按空格和标点拆分成若干长度至少为 2 的关键词，然后统计有多少关键词直接出现在 content 中。
def keyword_score(query: str, content: str) -> float:
    terms = [term for term in re.split(r"[\s，。！？、；：,.!?;:]+", query.lower()) if len(term) >= 2]
    if not terms:
        return 0.0
    lower = content.lower()
    matched = sum(1 for term in terms if term in lower)
    return min(1.0, matched / len(terms))


def _bm25_query_terms(query: str) -> list[str]:
    """Extract searchable terms for the BM25 candidate pre-filter (tokenized)."""
    terms = [term for term in tokenize(query) if len(term) >= 2]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return unique


def _hash_content(content: str) -> str:
    """Return a stable hash for the full source document."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def extract_pdf(data: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
