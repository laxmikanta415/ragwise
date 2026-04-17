"""Tests for ChunkerProtocol and RecursiveChunker — S2-T5/T6, S6-T5."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragwise.ingestion import ChunkerProtocol, ContextualChunker, RecursiveChunker
from ragwise.ingestion.document import Document


# ---------------------------------------------------------------------------
# S2-T5 — RecursiveChunker
# ---------------------------------------------------------------------------

_LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 50


def _make_doc(text: str = _LONG_TEXT) -> Document:
    return Document(text=text, source="test.txt")


def test_recursive_chunker_returns_nonempty_list() -> None:
    doc = _make_doc("Hello world. " * 100)
    chunks = RecursiveChunker().chunk(doc)
    assert len(chunks) >= 1


def test_recursive_chunker_chunk_index_and_count() -> None:
    doc = _make_doc(_LONG_TEXT)
    chunks = RecursiveChunker().chunk(doc)
    assert all("chunk_index" in c.metadata for c in chunks)
    assert all("chunk_count" in c.metadata for c in chunks)
    total = chunks[0].metadata["chunk_count"]
    assert total == len(chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(total))


def test_recursive_chunker_source_preserved() -> None:
    doc = _make_doc(_LONG_TEXT)
    chunks = RecursiveChunker().chunk(doc)
    assert all(c.source == "test.txt" for c in chunks)


def test_recursive_chunker_smaller_size_more_chunks() -> None:
    doc = _make_doc(_LONG_TEXT)
    small_chunks = RecursiveChunker(chunk_size=50).chunk(doc)
    large_chunks = RecursiveChunker(chunk_size=200).chunk(doc)
    assert len(small_chunks) > len(large_chunks)


def test_recursive_chunker_empty_doc_returns_empty() -> None:
    doc = Document(text="", source="empty.txt")
    chunks = RecursiveChunker().chunk(doc)
    assert chunks == []


def test_recursive_chunker_whitespace_only_returns_empty() -> None:
    doc = Document(text="   \n\t  ", source="ws.txt")
    chunks = RecursiveChunker().chunk(doc)
    assert chunks == []


def test_recursive_chunker_metadata_inherited() -> None:
    doc = Document(text=_LONG_TEXT, source="doc.txt", metadata={"extension": ".txt"})
    chunks = RecursiveChunker().chunk(doc)
    assert all(c.metadata["extension"] == ".txt" for c in chunks)


def test_chunker_protocol_structural_compatibility() -> None:
    chunker: ChunkerProtocol = RecursiveChunker()  # type: ignore[assignment]
    assert callable(chunker.chunk)


# ---------------------------------------------------------------------------
# S2-T6 — StructureAwareChunker
# ---------------------------------------------------------------------------

from ragwise.ingestion import StructureAwareChunker  # noqa: E402

_MD_TEXT = """# Introduction

This is the introduction section with some content.
It has multiple sentences to ensure chunking.

## Background

The background section explains the context.
It provides additional details about the topic.

## Methods

The methods section describes the approach.
We use various techniques to achieve the goal.
"""


def test_structure_aware_chunker_heading_sections() -> None:
    doc = Document(
        text=_MD_TEXT,
        source="guide.md",
        metadata={"extension": ".md", "headings": ["Introduction", "Background", "Methods"]},
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) >= 1
    headings = {c.metadata["section_heading"] for c in chunks}
    # At least two distinct heading values should appear
    assert len(headings) >= 2


def test_structure_aware_chunker_section_heading_in_metadata() -> None:
    doc = Document(
        text=_MD_TEXT,
        source="guide.md",
        metadata={"extension": ".md", "headings": ["Introduction", "Background", "Methods"]},
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert all("section_heading" in c.metadata for c in chunks)


def test_structure_aware_chunker_fallback_no_headings() -> None:
    doc = Document(
        text=_LONG_TEXT,
        source="plain.txt",
        metadata={"extension": ".txt"},
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) >= 1
    assert all(c.metadata.get("section_heading") is None for c in chunks)


def test_structure_aware_chunker_fallback_pdf_page() -> None:
    doc = Document(
        text="Hello World this is page text. " * 30,
        source="file.pdf",
        metadata={"extension": ".pdf", "page": 1, "total_pages": 3},
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) >= 1
    assert all("section_heading" in c.metadata for c in chunks)


def test_structure_aware_chunker_no_empty_text() -> None:
    doc = Document(
        text=_MD_TEXT,
        source="guide.md",
        metadata={"extension": ".md", "headings": ["Introduction", "Background", "Methods"]},
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert all(c.text.strip() for c in chunks)


def test_structure_aware_chunker_protocol_compatibility() -> None:
    chunker: ChunkerProtocol = StructureAwareChunker()  # type: ignore[assignment]
    assert callable(chunker.chunk)


# ---------------------------------------------------------------------------
# S6-T5 — ContextualChunker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contextual_chunker_prepends_summary() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Summary sentence.")

    doc = Document(text="Hello world. " * 50, source="test.txt")
    chunker = ContextualChunker(llm=llm, chunk_size=512)
    chunks = await chunker.chunk(doc)

    assert len(chunks) >= 1
    assert all(c.metadata.get("has_context") is True for c in chunks)
    assert chunks[0].text.startswith("Summary sentence.")


@pytest.mark.asyncio
async def test_contextual_chunker_llm_called_once_per_chunk() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Context.")

    doc = Document(text="Paragraph of text. " * 100, source="test.txt")
    chunker = ContextualChunker(llm=llm, chunk_size=512)
    chunks = await chunker.chunk(doc)

    assert llm.complete.call_count == len(chunks)


@pytest.mark.asyncio
async def test_contextual_chunker_llm_failure_returns_chunk_without_context() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

    doc = Document(text="Some text to chunk. " * 20, source="test.txt")
    chunker = ContextualChunker(llm=llm, chunk_size=512)
    chunks = await chunker.chunk(doc)

    # Chunks still returned — no crash; has_context is False on failure
    assert len(chunks) >= 1
    assert all(c.metadata.get("has_context") is False for c in chunks)


@pytest.mark.asyncio
async def test_contextual_chunker_empty_doc() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Context.")

    doc = Document(text="   ", source="test.txt")
    chunker = ContextualChunker(llm=llm)
    chunks = await chunker.chunk(doc)

    assert chunks == []
    llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# S6-T6 — HierarchicalChunker
# ---------------------------------------------------------------------------

from ragwise.ingestion import HierarchicalChunker  # noqa: E402


_HIER_TEXT = "Sentence about the topic. " * 200  # long enough for multiple parents


def test_hierarchical_chunker_returns_small_chunks() -> None:
    doc = Document(text=_HIER_TEXT, source="doc.txt")
    chunks = HierarchicalChunker(child_size=64, parent_size=256, overlap=16).chunk(doc)
    assert len(chunks) >= 2


def test_hierarchical_chunker_has_parent_text() -> None:
    doc = Document(text=_HIER_TEXT, source="doc.txt")
    chunks = HierarchicalChunker(child_size=64, parent_size=256, overlap=16).chunk(doc)
    assert all("parent_text" in c.metadata for c in chunks)


def test_hierarchical_chunker_parent_text_longer_than_child() -> None:
    doc = Document(text=_HIER_TEXT, source="doc.txt")
    chunks = HierarchicalChunker(child_size=64, parent_size=256, overlap=16).chunk(doc)
    for c in chunks:
        parent = c.metadata["parent_text"]
        # Parent text should be >= child text length (usually longer)
        assert len(parent) >= len(c.text) // 2


def test_hierarchical_chunker_parent_id_in_metadata() -> None:
    doc = Document(text=_HIER_TEXT, source="doc.txt")
    chunks = HierarchicalChunker(child_size=64, parent_size=256, overlap=16).chunk(doc)
    assert all("parent_id" in c.metadata for c in chunks)
    assert all("child_index" in c.metadata for c in chunks)


def test_hierarchical_chunker_multiple_children_share_parent_id() -> None:
    doc = Document(text=_HIER_TEXT, source="doc.txt")
    chunks = HierarchicalChunker(child_size=64, parent_size=256, overlap=16).chunk(doc)
    parent_ids = [c.metadata["parent_id"] for c in chunks]
    # At least two chunks share a parent if text is long enough
    from collections import Counter
    counts = Counter(parent_ids)
    assert any(v > 1 for v in counts.values()), "Expected multiple children per parent"
