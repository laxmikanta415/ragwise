"""Public configuration and response types for the ragx API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGConfig:
    """Top-level configuration for a RAG pipeline instance."""

    embedder: str | Any = "openai/text-embedding-3-small"
    store: str | Any = "memory"
    llm: str | Any = "openai/gpt-4o-mini"
    chunker: str | Any = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    reranker: str | None = None
    cache: bool | str = True
    batch_size: int = 32


@dataclass
class QueryConfig:
    """Per-query configuration overrides."""

    top_k: int = 10
    alpha: float = 0.5
    max_context_tokens: int = 8000
    check_sufficiency: bool = False
    sufficiency_threshold: float = 0.6
    include_citations: bool = True
    stream: bool = False
    tenant_id: str | None = None
    allowed_sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Answer:
    """Immutable response returned by RAG.query()."""

    text: str
    citations: list[str]
    chunks_used: int
    sufficient: bool = True


@dataclass
class IngestResult:
    """Result of a RAG.ingest() call."""

    succeeded: int
    failed: int
    errors: list[str] = field(default_factory=list)
