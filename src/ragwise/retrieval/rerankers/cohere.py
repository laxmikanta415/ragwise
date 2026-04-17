"""CohereReranker — Cohere Rerank API (cloud, best quality)."""
from __future__ import annotations

import os
from typing import Any

from ragwise.indexing.base import SearchResult


class CohereReranker:
    """Reranks search results using Cohere's Rerank API."""

    def __init__(self, model: str = "rerank-v3.5") -> None:
        self.model = model
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY env var not set — required for CohereReranker")
        try:
            import cohere
            self._client: Any = cohere.ClientV2(api_key=api_key)
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

        documents = [r.text for r in results]
        model = self.model
        client = self._client

        def _call() -> list[Any]:
            response = client.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(results)),
            )
            return response.results  # type: ignore[no-any-return]

        rerank_results: list[Any] = await anyio.to_thread.run_sync(_call)

        output: list[SearchResult] = []
        for item in rerank_results:
            orig = results[item.index]
            output.append(
                SearchResult(
                    id=orig.id,
                    text=orig.text,
                    source=orig.source,
                    score=float(item.relevance_score),
                    metadata=orig.metadata,
                    embedding=orig.embedding,
                    bm25_score=orig.bm25_score,
                    dense_score=orig.dense_score,
                )
            )
        return output
