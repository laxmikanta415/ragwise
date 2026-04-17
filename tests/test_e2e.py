"""End-to-end integration tests for ragwise v0.1 — S5-T3."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragwise import (
    RAG,
    Answer,
    EvalSchema,
    IngestResult,
    QueryConfig,
    RAGConfig,
)
from ragwise.ingestion.document import Document
from ragwise.indexing.memory import InMemoryStore

_FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def _make_embedder(dim: int = 8) -> MagicMock:
    emb = MagicMock()
    emb.embed = AsyncMock(side_effect=lambda texts: [[0.1] * dim for _ in texts])
    return emb


def _make_llm(answer: str = "This is a test answer based on the provided context.") -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=answer)
    return llm


# ---------------------------------------------------------------------------
# S5-T3 — top-level imports
# ---------------------------------------------------------------------------

def test_imports() -> None:
    """All 7 public types are importable from ragwise."""
    assert RAG
    assert RAGConfig
    assert QueryConfig
    assert Answer
    assert IngestResult
    assert Document
    assert EvalSchema


# ---------------------------------------------------------------------------
# S5-T3 — end-to-end pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_ingest_query() -> None:
    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        result = await rag.ingest(str(_FIXTURES))
        assert isinstance(result, IngestResult)
        assert result.succeeded >= 2  # sample.txt + guide.md

        answer = await rag.query("What is ragwise?")
        assert isinstance(answer, Answer)
        assert answer.text == "This is a test answer based on the provided context."


@pytest.mark.asyncio
async def test_incremental_ingest() -> None:
    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        r1 = await rag.ingest(str(_FIXTURES))
        r2 = await rag.ingest(str(_FIXTURES))  # unchanged

    assert r1.succeeded >= 2
    assert r2.succeeded == 0  # all skipped


@pytest.mark.asyncio
async def test_empty_store_query() -> None:
    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        answer = await rag.query("anything")

    assert isinstance(answer, Answer)
    assert answer.citations == []
    assert answer.chunks_used == 0


@pytest.mark.asyncio
async def test_answer_frozen() -> None:
    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        answer = await rag.query("q")

    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.text = "modified"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_query_config_top_k(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"doc{i}.txt").write_text(f"Document number {i}. " * 20)

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("document", config=QueryConfig(top_k=2))

    assert answer.chunks_used <= 2


@pytest.mark.asyncio
async def test_query_config_include_citations_false(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("some content")

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("content", config=QueryConfig(include_citations=False))

    assert answer.citations == []


@pytest.mark.asyncio
async def test_ingest_result_errors_on_bad_file(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("good")
    (tmp_path / "bad.xyz").write_text("unsupported")

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        result = await rag.ingest(str(tmp_path))

    assert result.failed == 1
    assert len(result.errors) == 1
    assert result.succeeded == 1
