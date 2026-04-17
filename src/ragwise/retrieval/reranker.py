"""CrossEncoderReranker — optional opt-in reranking via sentence-transformers."""
from __future__ import annotations

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
except ImportError as _e:
    raise ImportError("pip install ragwise[local-emb]") from _e

from ragwise.indexing.base import SearchResult


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model = _CrossEncoder(model)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not results:
            return []

        import anyio

        pairs = [(query, r.text) for r in results]

        def _predict() -> list[float]:
            return self._model.predict(pairs).tolist()  # type: ignore[no-any-return]

        scores: list[float] = await anyio.to_thread.run_sync(_predict)

        reranked = sorted(
            zip(results, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            SearchResult(
                id=r.id,
                text=r.text,
                source=r.source,
                score=s,
                metadata=r.metadata,
            )
            for r, s in reranked[:top_k]
        ]
