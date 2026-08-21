from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime

from app.core.timezone import now_cn
from pathlib import Path

import httpx

from app.core.config import Settings
from app.core.http_client import get_sync_client
from app.models.entities import KnowledgeChunk


PRIMARY_RETRIEVAL_LABEL = "Chroma vector + BM25 hybrid + local reranker"
FALLBACK_RETRIEVAL_LABEL = "local BM25 + hybrid_score reranker"


class VectorStoreUnavailable(RuntimeError):
    pass


@dataclass
class VectorSearchHit:
    chunk_id: int | None
    source: str
    source_index: int
    content: str
    score: float


class ChromaKnowledgeStore:
    """Primary RAG path: configurable OpenAI-compatible embeddings in Chroma."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.can_embed = False
        self.needs_rebuild = False
        self.error = ""
        self.embedding_model = settings.resolved_embedding_model
        self.embedding_base_url = settings.resolved_embedding_base_url
        self.embedding_api_key = settings.resolved_embedding_api_key
        if not settings.knowledge_vector_enabled:
            self.error = "Chroma 向量库未启用"
            return
        if not self.embedding_api_key:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 Embedding API Key，无法启用 Chroma 向量检索")
            self.error = f"缺少 Embedding API Key，Chroma 向量检索不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return
        try:
            import chromadb
        except ImportError as exc:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 chromadb 依赖，无法启用 Chroma 向量检索") from exc
            self.error = f"缺少 chromadb 依赖，Chroma 向量检索不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return

        persist_dir = self._resolve_path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
        )
        self.can_embed = settings.knowledge_vector_enabled
        stored_model = str((self.collection.metadata or {}).get("embedding_model", "")).strip()
        if stored_model and stored_model != self.embedding_model:
            self.needs_rebuild = True
            self.error = (
                f"Chroma 索引使用 {stored_model}，当前配置为 {self.embedding_model}，"
                "请重建向量索引后再启用向量检索"
            )

    @property
    def ready(self) -> bool:
        return self.can_embed and not self.needs_rebuild

    def reset_collection(self) -> None:
        """删除当前 collection；数据库中的 KnowledgeChunk 仍保留，可重新索引。"""
        if not self.can_embed:
            raise VectorStoreUnavailable(self.error or "Chroma 向量库不可用")
        self.client.delete_collection(name=self.settings.chroma_collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
        )
        self.needs_rebuild = False
        self.error = ""

    def upsert_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        rows = [chunk for chunk in chunks if chunk.id is not None and chunk.content.strip()]
        if not rows:
            return 0
        ids = [self._id(chunk.id) for chunk in rows] # knowledge-chunk-{chunk_id}
        documents = [chunk.content for chunk in rows]
        metadatas = [
            {"db_id": int(chunk.id), "source": chunk.source, "source_index": int(chunk.source_index)}
            for chunk in rows
        ]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        self.snapshot()
        return len(rows)

    def sync_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}
        current_ids = set(self.collection.get().get("ids", []))
        stale_ids = sorted(current_ids - valid_ids)
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return self.upsert_chunks(chunks, embeddings)

    def has_exact_chunk_ids(self, chunks: list[KnowledgeChunk]) -> bool:
        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}
        current_ids = set(self.collection.get().get("ids", []))
        return current_ids == valid_ids

    def delete_source(self, source: str) -> None:
        if not self.can_embed:
            return
        self.collection.delete(where={"source": source})

    def query(self, query_embedding: list[float], top_k: int) -> list[VectorSearchHit]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorSearchHit(
                    chunk_id=int(metadata["db_id"]) if metadata.get("db_id") is not None else None,
                    source=str(metadata.get("source", "")),
                    source_index=int(metadata.get("source_index", 0)),
                    content=document or "",
                    score=1.0 / (1.0 + max(0.0, distance)),
                )
            )
        return hits


    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.can_embed:
            raise VectorStoreUnavailable(self.error or "Chroma 向量检索不可用")
        return self._embed(texts)

    def snapshot(self) -> str | None:
        if not self.can_embed:
            return None
        if not self.persist_dir.exists():
            return None
        snapshot_root = self._resolve_path(self.settings.chroma_snapshot_dir)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        destination = snapshot_root / now_cn().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copytree(self.persist_dir, destination)
        self._prune_snapshots(snapshot_root)
        return str(destination)

    def count(self) -> int:
        if not self.can_embed:
            return 0
        return int(self.collection.count())

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # 构造请求体
        payload = {
            "model": self.embedding_model,
            "input": [text if text.strip() else " " for text in texts],
        }
        headers = {"Authorization": f"Bearer {self.embedding_api_key}"}
        # 发起 HTTP POST请求
        response = get_sync_client(self.settings).post(
            f"{self.embedding_base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=self.settings.embedding_timeout_seconds,
        )
        # 检查 HTTP 状态码
        response.raise_for_status()
        # 读取 API 返回结果
        # {
        #   "object": "list",
        #   "data": [
        #     {
        #       "object": "embedding",
        #       "index": 0,
        #       "embedding": [0.012, -0.083, 0.441]
        #     },
        #     {
        #       "object": "embedding",
        #       "index": 1,
        #       "embedding": [0.102, 0.031, -0.228]
        #     }
        #   ],
        #   "model": "text-embedding-3-small"
        # }
        # 按照index排序
        rows = sorted(response.json().get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [row.get("embedding") for row in rows]
        # 检查返回数量和内容是否对得上
        if len(embeddings) != len(texts) or any(not embedding for embedding in embeddings):
            raise VectorStoreUnavailable("OpenAI embeddings 接口返回向量数量不匹配")
        return [[float(value) for value in embedding] for embedding in embeddings]

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.settings.project_root / path

    def _prune_snapshots(self, snapshot_root: Path) -> None:
        keep = max(1, self.settings.chroma_snapshot_keep)
        snapshots = sorted([path for path in snapshot_root.iterdir() if path.is_dir()], reverse=True)
        for stale in snapshots[keep:]:
            shutil.rmtree(stale, ignore_errors=True)

    def _id(self, chunk_id: int) -> str:
        return f"knowledge-chunk-{chunk_id}"


ChromaKnowledgeVectorStore = ChromaKnowledgeStore
