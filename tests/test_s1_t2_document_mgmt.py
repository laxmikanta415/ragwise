"""S1-T2: Document management API — delete, list_sources, update."""
from __future__ import annotations

import pytest

from ragwise.indexing.memory import InMemoryStore
from ragwise.indexing.base import EmbeddedDoc


def _make_doc(id: str, source: str, text: str = "hello") -> EmbeddedDoc:
    return EmbeddedDoc(id=id, text=text, source=source, embedding=[0.1] * 8)


@pytest.mark.asyncio
async def test_delete_removes_all_chunks() -> None:
    store = InMemoryStore()
    await store.upsert([
        _make_doc("a1", "docs/a.md"),
        _make_doc("a2", "docs/a.md"),
        _make_doc("b1", "docs/b.md"),
    ])
    await store.delete("docs/a.md")
    sources = await store.list_sources()
    assert "docs/a.md" not in sources
    assert "docs/b.md" in sources


@pytest.mark.asyncio
async def test_delete_nonexistent_source_is_noop() -> None:
    store = InMemoryStore()
    await store.upsert([_make_doc("x1", "docs/x.md")])
    await store.delete("docs/nonexistent.md")  # should not raise
    sources = await store.list_sources()
    assert sources == ["docs/x.md"]


@pytest.mark.asyncio
async def test_list_sources_returns_indexed() -> None:
    store = InMemoryStore()
    await store.upsert([
        _make_doc("a1", "docs/a.md"),
        _make_doc("a2", "docs/a.md"),
        _make_doc("b1", "docs/b.md"),
        _make_doc("c1", "docs/c.txt"),
    ])
    sources = await store.list_sources()
    assert set(sources) == {"docs/a.md", "docs/b.md", "docs/c.txt"}
    # Each source appears exactly once
    assert len(sources) == 3


@pytest.mark.asyncio
async def test_list_sources_empty_store() -> None:
    store = InMemoryStore()
    sources = await store.list_sources()
    assert sources == []


@pytest.mark.asyncio
async def test_rag_delete_public_method(tmp_path, mock_embedder, mock_llm) -> None:
    from ragwise import RAG
    from unittest.mock import MagicMock

    (tmp_path / "a.txt").write_text("Document about refund policies.")
    (tmp_path / "b.txt").write_text("Document about shipping times.")

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        await rag.ingest(str(tmp_path))
        sources_before = await rag.list_sources()
        assert len(sources_before) == 2

        a_path = str(tmp_path / "a.txt")
        await rag.delete(a_path)

        sources_after = await rag.list_sources()
        assert len(sources_after) == 1
        assert a_path not in sources_after


@pytest.mark.asyncio
async def test_rag_list_sources(tmp_path, mock_embedder, mock_llm) -> None:
    from ragwise import RAG

    (tmp_path / "doc1.txt").write_text("First document.")
    (tmp_path / "doc2.txt").write_text("Second document.")

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        await rag.ingest(str(tmp_path))
        sources = await rag.list_sources()
        assert len(sources) == 2


@pytest.mark.asyncio
async def test_rag_update_replaces_content(tmp_path, mock_embedder, mock_llm) -> None:
    from ragwise import RAG

    doc = tmp_path / "doc.txt"
    doc.write_text("Original content about refunds.")

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        result1 = await rag.ingest(str(doc))
        assert result1.succeeded == 1

        doc.write_text("Updated content about new policies.")
        result2 = await rag.update(str(doc))
        assert result2.succeeded == 1

        sources = await rag.list_sources()
        # Still exactly one source after update
        assert sources.count(str(doc)) == 1
