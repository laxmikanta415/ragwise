"""S2-T2: Passage-level citations — Citation objects replacing list[str]."""
from __future__ import annotations

import pytest

from ragwise import Citation, RAG
from ragwise.config import Answer


def _make_embedder():
    from unittest.mock import AsyncMock, MagicMock
    emb = MagicMock()
    emb.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 384 for _ in texts])
    return emb


def _make_llm():
    from unittest.mock import AsyncMock, MagicMock
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="test answer")
    return llm


@pytest.mark.asyncio
async def test_citations_are_citation_objects(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert len(answer.citations) > 0
    assert isinstance(answer.citations[0], Citation)


@pytest.mark.asyncio
async def test_citation_text_not_empty(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert answer.citations[0].text != ""


@pytest.mark.asyncio
async def test_citation_score_populated(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert answer.citations[0].final_score >= 0


@pytest.mark.asyncio
async def test_citation_source_populated(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert str(tmp_path / "doc.txt") in answer.citations[0].source


@pytest.mark.asyncio
async def test_citation_sources_property(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("Document about refunds.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    sources = answer.citation_sources
    assert isinstance(sources, list)
    assert all(isinstance(s, str) for s in sources)


def test_answer_citation_sources_empty_citations() -> None:
    ans = Answer(text="x", citations=[], chunks_used=0)
    assert ans.citation_sources == []


def test_citation_dataclass_fields() -> None:
    c = Citation(
        text="passage text",
        source="docs/policy.md",
        chunk_id="chunk-001",
        final_score=0.85,
        bm25_score=0.4,
        dense_score=0.7,
        page=3,
    )
    assert c.text == "passage text"
    assert c.source == "docs/policy.md"
    assert c.chunk_id == "chunk-001"
    assert c.final_score == 0.85
    assert c.page == 3


def test_citation_page_none_for_non_pdf() -> None:
    c = Citation(text="t", source="doc.txt", chunk_id="x", final_score=0.5)
    assert c.page is None
