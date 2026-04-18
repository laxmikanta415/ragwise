"""VectorStore ABC, EmbeddedDoc, and SearchResult."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmbeddedDoc:
    id: str
    text: str
    source: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    # populated by dense_search when embeddings are available; empty otherwise
    embedding: list[float] = field(default_factory=list)
    # individual pre-fusion scores; populated by HybridSearcher
    bm25_score: float = 0.0
    dense_score: float = 0.0


class VectorStore(ABC):
    """Abstract base class for all ragwise vector stores."""

    @abstractmethod
    async def upsert(self, docs: list[EmbeddedDoc]) -> None:
        """Insert or replace documents by id."""

    @abstractmethod
    async def dense_search(self, query_vec: list[float], top_k: int) -> list[SearchResult]:
        """Return top-k results by cosine/vector similarity."""

    @abstractmethod
    async def sparse_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return top-k results by keyword/BM25/FTS similarity."""

    @abstractmethod
    async def delete(self, source: str) -> None:
        """Remove all documents whose source matches *source*."""

    @abstractmethod
    async def get_indexed_sources(self) -> dict[str, str]:
        """Return mapping of source path → content hash for incremental indexing."""

    @abstractmethod
    async def list_sources(self) -> list[str]:
        """Return all distinct source paths currently indexed."""

    async def list_sources_with_metadata(self) -> list[dict[str, Any]]:
        """Return list of dicts with at minimum 'source' and all chunk metadata keys.

        Used by StalenessChecker to inspect valid_from/valid_until without loading embeddings.
        Default implementation returns source + empty metadata; override for richer output.
        """
        sources = await self.list_sources()
        return [{"source": s} for s in sources]

    async def get_adjacent_chunks(self, chunk_id: str, window: int = 2) -> list[SearchResult]:
        """Return up to window chunks before and after chunk_id (same source).

        Default implementation returns an empty list. Override in stores that
        persist chunk_index metadata.
        """
        return []
