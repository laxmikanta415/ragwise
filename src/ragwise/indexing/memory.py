"""InMemoryStore — numpy dense + rank-bm25 sparse. Zero external services."""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from ragwise.indexing.base import EmbeddedDoc, SearchResult, VectorStore


class InMemoryStore(VectorStore):
    """Volatile in-memory store. Ideal for tests and single-session demos."""

    def __init__(self) -> None:
        self._docs: list[EmbeddedDoc] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_matrix(self) -> np.ndarray:
        if not self._docs:
            return np.empty((0, 0), dtype=np.float32)
        return np.array([d.embedding for d in self._docs], dtype=np.float32)

    def _rebuild_bm25(self) -> BM25Okapi:
        corpus = [d.text.lower().split() for d in self._docs]
        return BM25Okapi(corpus)

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    async def upsert(self, docs: list[EmbeddedDoc]) -> None:
        incoming_ids = {d.id for d in docs}
        # Remove existing docs whose id appears in the upsert batch
        self._docs = [d for d in self._docs if d.id not in incoming_ids]
        self._docs.extend(docs)

    async def dense_search(self, query_vec: list[float], top_k: int) -> list[SearchResult]:
        if not self._docs:
            return []

        matrix = self._rebuild_matrix()
        qv = np.array(query_vec, dtype=np.float32)

        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(qv)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1e-10, norms)
        scores = np.dot(matrix, qv) / safe_norms

        top_indices = np.argsort(-scores)[: top_k]
        return [
            SearchResult(
                id=self._docs[i].id,
                text=self._docs[i].text,
                source=self._docs[i].source,
                score=float(scores[i]),
                metadata=self._docs[i].metadata,
                embedding=list(self._docs[i].embedding),
            )
            for i in top_indices
        ]

    async def sparse_search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self._docs:
            return []

        bm25 = self._rebuild_bm25()
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)

        top_indices = np.argsort(-scores)[: top_k]
        return [
            SearchResult(
                id=self._docs[i].id,
                text=self._docs[i].text,
                source=self._docs[i].source,
                score=float(scores[i]),
                metadata=self._docs[i].metadata,
            )
            for i in top_indices
        ]

    async def delete(self, source: str) -> None:
        self._docs = [d for d in self._docs if d.source != source]

    async def get_indexed_sources(self) -> dict[str, str]:
        seen: dict[str, str] = {}
        for doc in self._docs:
            if doc.source not in seen:
                seen[doc.source] = doc.metadata.get("content_hash", "")
        return seen

    async def list_sources(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for doc in self._docs:
            if doc.source not in seen:
                seen.add(doc.source)
                result.append(doc.source)
        return result
