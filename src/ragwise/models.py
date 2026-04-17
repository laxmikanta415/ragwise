"""Shared domain models: RetrievedChunk, Citation, QueryTrace."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved document chunk with full scoring breakdown."""

    text: str
    source: str
    chunk_id: str
    final_score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float | None = None
    char_start: int = 0
    char_end: int = 0
    page: int | None = None

    def explain(self) -> str:
        """Human-readable ranking explanation for debugging retrieval."""
        lines = [f"Chunk {self.chunk_id} from {self.source}"]
        lines.append(
            f"  BM25: {self.bm25_score:.3f}  Dense: {self.dense_score:.3f}"
            f"  Final: {self.final_score:.3f}"
        )
        if self.rerank_score is not None:
            lines.append(f"  Rerank: {self.rerank_score:.3f}")
        return "\n".join(lines)


# Citation is the public-facing alias for RetrievedChunk
Citation = RetrievedChunk


@dataclass
class QueryTrace:
    """Full observability trace for a single RAG query."""

    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    dropped_chunks: list[RetrievedChunk] = field(default_factory=list)
    query_embedding_ms: int = 0
    retrieval_ms: int = 0
    rerank_ms: int = 0
    generation_ms: int = 0
    context_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
