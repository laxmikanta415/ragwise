"""S4-T3: Query expansion (RAG-Fusion) — multi-query with RRF."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from ragwise import RAG, QueryConfig
from ragwise.testing import FakeEmbedder, FakeLLM


@pytest.mark.asyncio
async def test_query_expansion_n1_no_change(tmp_path) -> None:
    """n_queries=1 (default) behaves exactly as single-query retrieval."""
    (tmp_path / "doc.txt").write_text("Refund policy: 30 days.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund", config=QueryConfig(n_queries=1))

    assert answer.trace is not None
    assert answer.trace.expanded_queries is None


@pytest.mark.asyncio
async def test_query_expansion_custom_variants(tmp_path) -> None:
    """query_variants bypasses LLM variant generation."""
    (tmp_path / "doc.txt").write_text("Policy about returns and refunds.")
    llm = FakeLLM()
    llm_call_count = [0]
    original = llm.complete

    async def counting(prompt):
        llm_call_count[0] += 1
        return await original(prompt)

    llm.complete = counting

    async with RAG(embedder=FakeEmbedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query(
            "refund policy",
            config=QueryConfig(query_variants=["return policy", "money back"]),
        )

    assert answer.trace is not None
    assert answer.trace.expanded_queries == ["return policy", "money back"]
    # LLM should only have been called for generation, not variant generation
    # (1 call for the final answer, 0 for variant generation)
    assert llm_call_count[0] <= 1


@pytest.mark.asyncio
async def test_query_expansion_generates_variants(tmp_path) -> None:
    """n_queries=3 → trace.expanded_queries has variants."""
    (tmp_path / "doc.txt").write_text("The refund policy allows 30 day returns.")
    llm = MagicMock()
    # First call returns JSON variants, second call returns the answer
    llm.complete = AsyncMock(side_effect=[
        '["refund policy", "money back guarantee", "return items"]',
        "You can return items within 30 days.",
    ])

    async with RAG(embedder=FakeEmbedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund", config=QueryConfig(n_queries=3))

    assert answer.trace is not None
    assert answer.trace.expanded_queries is not None
    assert len(answer.trace.expanded_queries) == 3


@pytest.mark.asyncio
async def test_query_expansion_trace_expansion_ms(tmp_path) -> None:
    """trace.expansion_ms is populated when expansion is used."""
    (tmp_path / "doc.txt").write_text("Some content.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query(
            "content",
            config=QueryConfig(query_variants=["content info", "details"]),
        )

    assert answer.trace is not None
    assert answer.trace.expansion_ms >= 0


@pytest.mark.asyncio
async def test_query_expansion_deduplication(tmp_path) -> None:
    """Variant results deduplicated — each chunk appears once."""
    (tmp_path / "doc.txt").write_text("Returns and refunds are accepted within 30 days.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query(
            "refund",
            config=QueryConfig(
                query_variants=["refund policy", "return policy", "money back"],
                top_k=5,
            ),
        )

    # No duplicate chunk_ids in citations
    ids = [c.chunk_id for c in answer.citations]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_query_expansion_llm_parse_failure(tmp_path) -> None:
    """Invalid JSON from LLM → falls back to single query gracefully."""
    (tmp_path / "doc.txt").write_text("Returns policy content.")
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=[
        "not valid json at all !!!",  # expansion call fails
        "Fallback answer",             # generation call
    ])

    async with RAG(embedder=FakeEmbedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        # Should not raise
        answer = await rag.query("policy", config=QueryConfig(n_queries=3))

    assert answer.text == "Fallback answer"
