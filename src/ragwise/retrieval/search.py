"""HybridSearcher — dense + sparse retrieval fused via RRF."""
from __future__ import annotations

from fnmatch import fnmatch

from ragwise.embedding.base import EmbedderProtocol
from ragwise.indexing.base import SearchResult, VectorStore
from ragwise.utils.rrf import rrf


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

        return results
