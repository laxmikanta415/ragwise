"""FlashRankReranker — local reranking, no GPU required."""
from __future__ import annotations

import logging
from typing import Any

from ragwise.indexing.base import SearchResult

logger = logging.getLogger(__name__)


class FlashRankReranker:
    """Reranks search results using FlashRank (local, CPU-only)."""

    def __init__(self, model: str = "ms-marco-MiniLM-L-12-v2") -> None:
        self.model = model
        try:
            from flashrank import Ranker
            logger.info("FlashRank: loading model %s (downloads on first use)", model)
            self._ranker: Any = Ranker(model_name=model)
        except ImportError as e:
            raise ImportError("pip install ragwise[rerank]") from e

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not results:
            return []

        import anyio
        from flashrank import RerankRequest

        passages = [{"id": i, "text": r.text} for i, r in enumerate(results)]
        request = RerankRequest(query=query, passages=passages)
        ranker = self._ranker

        def _call() -> list[Any]:
            return ranker.rerank(request)  # type: ignore[no-any-return]

        reranked: list[Any] = await anyio.to_thread.run_sync(_call)

        output: list[SearchResult] = []
        for item in reranked[:top_k]:
            orig = results[item["id"]]
            output.append(
                SearchResult(
                    id=orig.id,
                    text=orig.text,
                    source=orig.source,
                    score=float(item["score"]),
                    metadata=orig.metadata,
                    embedding=orig.embedding,
                    bm25_score=orig.bm25_score,
                    dense_score=orig.dense_score,
                )
            )
        return output
