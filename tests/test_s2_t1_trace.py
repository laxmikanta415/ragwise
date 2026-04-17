"""S2-T1: answer.trace — always-on retrieval observability."""
from __future__ import annotations

import pytest

from ragwise import RAG, QueryTrace
from ragwise.indexing.memory import InMemoryStore


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
async def test_trace_populated(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert answer.trace is not None
    assert answer.trace.query_embedding_ms >= 0
    assert answer.trace.retrieval_ms >= 0
    assert answer.trace.generation_ms >= 0


@pytest.mark.asyncio
async def test_trace_retrieved_chunks_populated(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert answer.trace is not None
    assert len(answer.trace.retrieved_chunks) > 0
    chunk = answer.trace.retrieved_chunks[0]
    assert chunk.chunk_id != ""
    assert chunk.source != ""
    assert chunk.final_score >= 0


@pytest.mark.asyncio
async def test_trace_component_scores(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert answer.trace is not None
    chunk = answer.trace.retrieved_chunks[0]
    # At least one of bm25 or dense should be non-zero
    assert chunk.bm25_score >= 0
    assert chunk.dense_score >= 0


@pytest.mark.asyncio
async def test_trace_rerank_score_none_without_reranker(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), reranker=None, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert answer.trace is not None
    for chunk in answer.trace.retrieved_chunks:
        assert chunk.rerank_score is None


@pytest.mark.asyncio
async def test_trace_cost_zero_for_fake_llm(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert answer.trace is not None
    # Fake LLM spec is not in pricing table → cost = 0
    assert answer.trace.cost_usd == 0.0


@pytest.mark.asyncio
async def test_trace_context_tokens_populated(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days." * 10)
    async with RAG(embedder=_make_embedder(), llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert answer.trace is not None
    assert answer.trace.context_tokens > 0


def test_chunk_explain_output() -> None:
    from ragwise.models import RetrievedChunk
    chunk = RetrievedChunk(
        text="some text",
        source="docs/policy.md",
        chunk_id="abc123",
        final_score=0.72,
        bm25_score=0.45,
        dense_score=0.61,
    )
    explanation = chunk.explain()
    assert "abc123" in explanation
    assert "docs/policy.md" in explanation
    assert "0.45" in explanation
    assert "0.61" in explanation


def test_chunk_explain_includes_rerank_score() -> None:
    from ragwise.models import RetrievedChunk
    chunk = RetrievedChunk(
        text="t", source="s", chunk_id="x", final_score=0.9,
        rerank_score=0.88,
    )
    assert "0.88" in chunk.explain()


def test_query_trace_defaults() -> None:
    trace = QueryTrace()
    assert trace.cache_hit is False
    assert trace.cost_usd == 0.0
    assert trace.retrieved_chunks == []
