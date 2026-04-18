"""HybridSearcher — dense + sparse retrieval fused via RRF."""
from __future__ import annotations

from datetime import datetime
from fnmatch import fnmatch
from typing import Any

from ragwise.embedding.base import EmbedderProtocol
from ragwise.indexing.base import SearchResult, VectorStore
from ragwise.utils.rrf import rrf


def _parse_as_of(as_of: str | datetime | None) -> datetime | None:
    if as_of is None:
        return None
    if isinstance(as_of, datetime):
        return as_of
    if as_of == "now":
        return datetime.utcnow()
    return datetime.fromisoformat(as_of)


def _in_temporal_range(metadata: dict[str, Any], as_of_dt: datetime) -> bool:
    """Return True if the chunk's valid_from/until range includes as_of_dt.

    Chunks without valid_from/until metadata always pass (backward compat).
    """
    valid_from = metadata.get("valid_from")
    valid_until = metadata.get("valid_until")
    if valid_from is None and valid_until is None:
        return True
    if valid_from is not None:
        try:
            if as_of_dt < datetime.fromisoformat(str(valid_from)):
                return False
        except ValueError:
            pass
    if valid_until is not None:
        try:
            if as_of_dt > datetime.fromisoformat(str(valid_until)):
                return False
        except ValueError:
            pass
    return True


class HybridSearcher:
    """Fuses dense (vector) and sparse (BM25/FTS) retrieval via Reciprocal Rank Fusion."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbedderProtocol,
        k: int = 60,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self.k = k

    async def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,  # reserved for future weighted fusion — not used in RRF
        tenant_id: str | None = None,
        allowed_sources: list[str] | None = None,
        as_of: str | datetime | None = None,
        version: str | None = None,
    ) -> list[SearchResult]:
        # Embed once; reuse for dense search
        vecs = await self._embedder.embed([query])
        query_vec = vecs[0]

        dense_results = await self._store.dense_search(query_vec, top_k * 2)
        sparse_results = await self._store.sparse_search(query, top_k * 2)

        if not dense_results and not sparse_results:
            return []

        # Build lookup: id → SearchResult (dense takes precedence for tie-breaking)
        lookup: dict[str, SearchResult] = {}
        for r in sparse_results:
            lookup[r.id] = r
        for r in dense_results:
            lookup[r.id] = r

        # Preserve individual scores for trace/observability
        dense_score_map: dict[str, float] = {r.id: r.score for r in dense_results}
        sparse_score_map: dict[str, float] = {r.id: r.score for r in sparse_results}

        dense_ids = [r.id for r in dense_results]
        sparse_ids = [r.id for r in sparse_results]

        fused = rrf([dense_ids, sparse_ids], k=self.k)

        results = []
        for doc_id, rrf_score in fused[:top_k]:
            if doc_id in lookup:
                orig = lookup[doc_id]
                results.append(
                    SearchResult(
                        id=orig.id,
                        text=orig.text,
                        source=orig.source,
                        score=rrf_score,
                        metadata=orig.metadata,
                        embedding=orig.embedding,
                        bm25_score=sparse_score_map.get(doc_id, 0.0),
                        dense_score=dense_score_map.get(doc_id, 0.0),
                    )
                )

        # Post-hoc filtering — applied after RRF so all stores benefit without schema changes
        if tenant_id is not None:
            results = [r for r in results if r.metadata.get("tenant_id") == tenant_id]
        if allowed_sources:
            results = [r for r in results if any(fnmatch(r.source, pat) for pat in allowed_sources)]
        if as_of is not None:
            as_of_dt = _parse_as_of(as_of)
            if as_of_dt is not None:
                results = [r for r in results if _in_temporal_range(r.metadata, as_of_dt)]
        if version is not None:
            results = [
                r for r in results
                if r.metadata.get("version") == version or "version" not in r.metadata
            ]

        return results
