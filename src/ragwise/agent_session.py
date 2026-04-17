"""AgentSession — stateful multi-turn RAG agent with dedup, loop detection, context budget."""
from __future__ import annotations

import difflib
import logging
from typing import Any

from ragwise.indexing.base import SearchResult
from ragwise.models import RetrievedChunk
from ragwise.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class AgentSession:
    """Tracks retrieved chunks across multiple search calls for a single agent turn.

    Features:
    - Chunk deduplication by chunk_id across calls
    - Loop detection via query similarity (difflib ratio > 0.85)
    - Context budget tracking (token count)

    Example::

        session = AgentSession(rag, max_iterations=5)
        chunks = await session.search("refund policy")
        budget = session.context_budget_status()
    """

    def __init__(
        self,
        rag: Any,
        max_iterations: int = 5,
        context_budget_tokens: int = 8000,
    ) -> None:
        self.rag = rag
        self.max_iterations = max_iterations
        self.context_budget_tokens = context_budget_tokens
        self._retrieved_chunks: dict[str, RetrievedChunk] = {}
        self._query_history: list[str] = []
        self._iteration: int = 0
        self._tokens_used: int = 0

    @property
    def retrieved_chunks(self) -> dict[str, RetrievedChunk]:
        return self._retrieved_chunks

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    def _check_loop(self, query: str) -> None:
        for i, prev in enumerate(self._query_history):
            ratio = difflib.SequenceMatcher(None, query.lower(), prev.lower()).ratio()
            if ratio > 0.85:
                logger.warning(
                    "[AgentSession] Possible retrieval loop: query similar to prior query #%d "
                    "(similarity=%.2f): %r vs %r",
                    i,
                    ratio,
                    query[:60],
                    prev[:60],
                )

    async def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search the RAG index, deduplicate against already-retrieved chunks."""
        self._check_loop(query)
        self._query_history.append(query)
        self._iteration += 1

        results: list[SearchResult] = await self.rag.search(query, top_k=top_k)

        new_chunks: list[RetrievedChunk] = []
        for r in results:
            if r.id not in self._retrieved_chunks:
                chunk = RetrievedChunk(
                    text=r.text,
                    source=r.source,
                    chunk_id=r.id,
                    final_score=r.score,
                    bm25_score=r.bm25_score,
                    dense_score=r.dense_score,
                )
                self._retrieved_chunks[r.id] = chunk
                new_chunks.append(chunk)
                self._tokens_used += count_tokens(r.text)

        summary = (
            f"\n\n[Session: {self._iteration}/{self.max_iterations} iterations used, "
            f"{len(self._retrieved_chunks)} unique chunks retrieved, "
            f"{self._tokens_used}/{self.context_budget_tokens} tokens]"
        )
        logger.debug(summary)
        return new_chunks

    async def get_context(self, chunk_id: str, window: int = 2) -> list[RetrievedChunk]:
        """Get adjacent chunks (window before and after) for a given chunk_id."""
        assert self.rag._store is not None, "RAG must be used as a context manager"
        results = await self.rag._store.get_adjacent_chunks(chunk_id, window)
        return [
            RetrievedChunk(
                text=r.text,
                source=r.source,
                chunk_id=r.id,
                final_score=r.score,
                bm25_score=r.bm25_score,
                dense_score=r.dense_score,
            )
            for r in results
        ]

    def context_budget_status(self) -> dict[str, int]:
        """Return current context budget state."""
        return {
            "chunks_retrieved": len(self._retrieved_chunks),
            "tokens_used": self._tokens_used,
            "tokens_remaining": max(0, self.context_budget_tokens - self._tokens_used),
            "iteration": self._iteration,
        }
