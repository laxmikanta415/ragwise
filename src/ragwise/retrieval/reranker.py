"""Reranker implementations and factory — cross-encoder, Cohere, FlashRank."""
from __future__ import annotations

from typing import Any

from ragwise.indexing.base import SearchResult


def resolve_reranker(spec: str | Any | None) -> Any:
    """Return a reranker from a string spec, an existing object, or None."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec
    if spec.startswith("cross-encoder/") or spec.startswith("BAAI/"):
        return CrossEncoderReranker(model=spec)
    if spec.startswith("cohere/"):
        model = spec.split("/", 1)[1]
        from ragwise.retrieval.rerankers.cohere import CohereReranker
        return CohereReranker(model=model)
    if spec == "flashrank":
        from ragwise.retrieval.rerankers.flashrank import FlashRankReranker
        return FlashRankReranker()
    if spec.startswith("flashrank/"):
        model = spec.split("/", 1)[1]
        from ragwise.retrieval.rerankers.flashrank import FlashRankReranker
        return FlashRankReranker(model=model)
    raise ValueError(f"Unknown reranker spec: {spec!r}")


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model (sentence-transformers)."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from sentence_transformers import CrossEncoder as _CrossEncoder
            self._model = _CrossEncoder(model)
        except ImportError as e:
            raise ImportError("pip install ragwise[local-emb]") from e

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
                embedding=r.embedding,
                bm25_score=r.bm25_score,
                dense_score=r.dense_score,
            )
            for r, s in reranked[:top_k]
        ]
