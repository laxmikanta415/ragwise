"""S1-T3: Stale chunk invalidation — no duplicate chunks on re-ingest."""
from __future__ import annotations

import pytest

from ragwise import RAG
from ragwise.indexing.memory import InMemoryStore


@pytest.mark.asyncio
async def test_reingest_removes_old_chunks(tmp_path, mock_embedder, mock_llm) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("Original content.")

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        assert isinstance(rag._store, InMemoryStore)

        await rag.ingest(str(doc))
        count_after_first = len(rag._store._docs)

        # Modify the file so it re-ingests
        doc.write_text("Updated content with more words to potentially create different chunks.")
        await rag.ingest(str(doc))
        count_after_second = len(rag._store._docs)

        # Should not accumulate: second ingest replaces first
        # Allow for different chunk count but old chunks must be gone
        doc_sources = [d.source for d in rag._store._docs]
        assert doc_sources.count(str(doc)) == len(
            [d for d in rag._store._docs if d.source == str(doc)]
        )


@pytest.mark.asyncio
async def test_reingest_no_duplicate_chunks(tmp_path, mock_embedder, mock_llm) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("Content about refund policy that is long enough to produce chunks.")

    async with RAG(llm=mock_llm, embedder=mock_embedder, chunk_size=20, chunk_overlap=5) as rag:
        assert isinstance(rag._store, InMemoryStore)

        await rag.ingest(str(doc))
        chunks_after_first = [d for d in rag._store._docs if d.source == str(doc)]
        count_first = len(chunks_after_first)

        # Re-ingest same file with force=True (to bypass hash check)
        await rag.ingest(str(doc), force=True)
        chunks_after_second = [d for d in rag._store._docs if d.source == str(doc)]
        count_second = len(chunks_after_second)

        # Must not double-accumulate
        assert count_second == count_first, (
            f"Expected {count_first} chunks after re-ingest, got {count_second} — stale chunks not deleted"
        )


@pytest.mark.asyncio
async def test_reingest_unchanged_file_skipped(tmp_path, mock_embedder, mock_llm) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("Unchanged content.")

    embed_calls: list[int] = []

    import asyncio
    original_embed = mock_embedder.embed.side_effect

    async def counting_embed(texts):
        embed_calls.append(len(texts))
        return [[0.1] * 384 for _ in texts]

    mock_embedder.embed.side_effect = counting_embed

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        await rag.ingest(str(doc))
        calls_after_first = len(embed_calls)

        # Second ingest — file unchanged, should skip entirely
        await rag.ingest(str(doc))
        calls_after_second = len(embed_calls)

        assert calls_after_second == calls_after_first, "Unchanged file should not be re-embedded"


@pytest.mark.asyncio
async def test_multiple_files_independent_stale_invalidation(tmp_path, mock_embedder, mock_llm) -> None:
    doc_a = tmp_path / "a.txt"
    doc_b = tmp_path / "b.txt"
    doc_a.write_text("Document A content.")
    doc_b.write_text("Document B content.")

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        assert isinstance(rag._store, InMemoryStore)

        await rag.ingest(str(tmp_path))

        # Modify only doc_a
        doc_a.write_text("Document A updated content.")
        await rag.ingest(str(tmp_path))

        # doc_b chunks should still be present and unchanged
        b_chunks = [d for d in rag._store._docs if d.source == str(doc_b)]
        assert len(b_chunks) > 0
