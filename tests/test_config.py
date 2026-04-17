"""Tests for RAGConfig, QueryConfig, Answer, IngestResult — S1-T3."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ragwise import Answer, Citation, IngestResult, QueryConfig, RAGConfig


def test_ragconfig_defaults() -> None:
    cfg = RAGConfig()
    assert cfg.embedder == "openai/text-embedding-3-small"
    assert cfg.store == "memory"
    assert cfg.llm == "openai/gpt-4o-mini"
    assert cfg.chunker == "recursive"
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64
    assert cfg.reranker is None
    assert cfg.cache is True
    assert cfg.batch_size == 32


def test_ragconfig_custom_llm() -> None:
    cfg = RAGConfig(llm="anthropic/claude-haiku-4-5")
    assert cfg.llm == "anthropic/claude-haiku-4-5"


def test_queryconfig_defaults() -> None:
    qc = QueryConfig()
    assert qc.top_k == 10
    assert qc.alpha == 0.5
    assert qc.max_context_tokens == 8000
    assert qc.check_sufficiency is False
    assert qc.sufficiency_threshold == 0.6
    assert qc.include_citations is True
    assert qc.stream is False


def test_answer_fields() -> None:
    cit = Citation(text="passage", source="a.txt", chunk_id="c1", final_score=0.9)
    ans = Answer(text="hello", citations=[cit], chunks_used=3)
    assert ans.text == "hello"
    assert ans.citations[0].source == "a.txt"
    assert ans.citation_sources == ["a.txt"]
    assert ans.chunks_used == 3
    assert ans.sufficient is True
    assert ans.has_sufficient_context is True


def test_answer_is_frozen() -> None:
    ans = Answer(text="x", citations=[], chunks_used=0)
    with pytest.raises(FrozenInstanceError):
        ans.text = "mutated"  # type: ignore[misc]


def test_ingest_result_fields() -> None:
    result = IngestResult(succeeded=5, failed=1, errors=["x.pdf: bad"])
    assert result.succeeded == 5
    assert result.failed == 1
    assert result.errors == ["x.pdf: bad"]


def test_ingest_result_defaults() -> None:
    result = IngestResult(succeeded=0, failed=0)
    assert result.errors == []


def test_all_four_importable_from_ragwise() -> None:
    import ragwise
    for name in ("RAGConfig", "QueryConfig", "Answer", "IngestResult"):
        assert name in ragwise.__all__
