"""Public configuration and response types for the ragwise API."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ragwise.models import Citation, QueryTrace


class RAGConfig(BaseModel):
    """Top-level configuration for a RAG pipeline instance.

    Validated at construction — invalid values raise immediately::

        RAGConfig(chunk_size=-1)   # ValidationError: chunk_size must be > 0
        RAGConfig.from_env()       # reads RAGWISE_* env vars
        RAGConfig.from_yaml("ragwise.yaml")
    """

    model_config = {"arbitrary_types_allowed": True}

    embedder: Any = "openai/text-embedding-3-small"
    store: Any = "memory"
    llm: Any = "openai/gpt-4o-mini"
    chunker: Any = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    reranker: str | None = None
    cache: bool | str = True
    batch_size: int = 32
    confidence_threshold: float = 0.0
    insufficient_response: str = (
        "I could not find reliable information in the indexed documents to answer this question."
    )

    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"chunk_size must be > 0, got {v}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _chunk_overlap_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def _batch_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"batch_size must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _overlap_less_than_size(self) -> RAGConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        return self

    @classmethod
    def from_env(cls) -> RAGConfig:
        """Construct RAGConfig from RAGWISE_* environment variables."""
        kwargs: dict[str, Any] = {}
        _str_fields = {
            "RAGWISE_LLM": "llm",
            "RAGWISE_STORE": "store",
            "RAGWISE_EMBEDDER": "embedder",
            "RAGWISE_RERANKER": "reranker",
            "RAGWISE_CHUNKER": "chunker",
        }
        _int_fields = {
            "RAGWISE_CHUNK_SIZE": "chunk_size",
            "RAGWISE_CHUNK_OVERLAP": "chunk_overlap",
            "RAGWISE_BATCH_SIZE": "batch_size",
        }
        for env_key, field_name in _str_fields.items():
            if val := os.getenv(env_key):
                kwargs[field_name] = val
        for env_key, field_name in _int_fields.items():
            if val := os.getenv(env_key):
                kwargs[field_name] = int(val)
        if cache_val := os.getenv("RAGWISE_CACHE"):
            kwargs["cache"] = False if cache_val.lower() in ("0", "false", "no") else cache_val
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGConfig:
        """Load RAGConfig from a YAML file."""
        import yaml  # type: ignore[import-untyped]

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**(data or {}))


class QueryConfig(BaseModel):
    """Per-query configuration overrides."""

    model_config = {"arbitrary_types_allowed": True}

    top_k: int = 10
    alpha: float = 0.5
    max_context_tokens: int = 8000
    check_sufficiency: bool = False
    sufficiency_threshold: float = 0.6
    include_citations: bool = True
    stream: bool = False
    tenant_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    citation_mode: str = "passage"  # "passage" | "source"

    @field_validator("alpha")
    @classmethod
    def _alpha_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"top_k must be > 0, got {v}")
        return v

    @field_validator("sufficiency_threshold")
    @classmethod
    def _threshold_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"sufficiency_threshold must be in [0.0, 1.0], got {v}")
        return v


@dataclass(frozen=True)
class Answer:
    """Immutable response returned by RAG.query()."""

    text: str
    citations: list[Citation]
    chunks_used: int
    sufficient: bool = True
    has_sufficient_context: bool = True
    trace: QueryTrace | None = None
    top_retrieved: list[Citation] | None = None

    @property
    def citation_sources(self) -> list[str]:
        """Convenience shorthand: list of source file paths from citations."""
        return [c.source for c in self.citations]


@dataclass
class IngestResult:
    """Result of a RAG.ingest() call."""

    succeeded: int
    failed: int
    errors: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
