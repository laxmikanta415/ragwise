"""S4-T2: Semantic query cache — embedding similarity replacing SHA-256."""
from __future__ import annotations

import pytest

from ragwise import RAG, QueryConfig
from ragwise.generation.semantic_cache import MemorySemanticCache
from ragwise.testing import FakeEmbedder, FakeLLM


# ---------------------------------------------------------------------------
# MemorySemanticCache unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_cache_miss_empty() -> None:
    cache = MemorySemanticCache(threshold=0.92)
    result, sim = cache.lookup([0.1, 0.2, 0.3])
    assert result is None
    assert sim is None


@pytest.mark.asyncio
async def test_semantic_cache_exact_hit() -> None:
    from ragwise.config import Answer

    cache = MemorySemanticCache(threshold=0.5)
    vec = [1.0, 0.0, 0.0]
    fake_answer = Answer(text="cached", citations=[], chunks_used=1)
    cache.store(vec, fake_answer)
    result, sim = cache.lookup(vec)
    assert result is not None
    assert result.text == "cached"
    assert sim is not None and sim > 0.99


@pytest.mark.asyncio
async def test_semantic_cache_threshold_override() -> None:
    from ragwise.config import Answer

    cache = MemorySemanticCache(threshold=0.5)
    vec = [1.0, 0.0, 0.0]
    cache.store(vec, Answer(text="x", citations=[], chunks_used=0))
    # With very high threshold, exact match still hits
    result, _ = cache.lookup(vec, threshold=0.99)
    assert result is not None
    # With threshold=1.1 (impossible), miss
    result2, _ = cache.lookup(vec, threshold=1.1)
    assert result2 is None


def test_semantic_cache_stats() -> None:
    from ragwise.config import Answer

    cache = MemorySemanticCache(threshold=0.5)
    vec = [1.0, 0.0, 0.0]
    ans = Answer(text="a", citations=[], chunks_used=0)
    cache.store(vec, ans)

    # 3 hits
    cache.lookup(vec)
    cache.lookup(vec)
    cache.lookup(vec)
    # 2 misses
    cache.lookup([0.0, 1.0, 0.0])
    cache.lookup([0.0, 0.0, 1.0])

    stats = cache.stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 2
    assert abs(stats["hit_rate"] - 0.6) < 0.01
    assert stats["entries"] == 1


def test_semantic_cache_clear() -> None:
    from ragwise.config import Answer

    cache = MemorySemanticCache(threshold=0.5)
    cache.store([1.0, 0.0], Answer(text="x", citations=[], chunks_used=0))
    cache.clear()
    assert cache.stats()["entries"] == 0
    assert cache.stats()["hits"] == 0


# ---------------------------------------------------------------------------
# Integration tests via RAG
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_cache_hit_populated(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    llm = FakeLLM(response="30 days")
    emb = FakeEmbedder()

    async with RAG(embedder=emb, llm=llm, cache=True) as rag:
        await rag.ingest(str(tmp_path))
        answer1 = await rag.query("refund policy")
        # Second identical query → cache hit
        answer2 = await rag.query("refund policy")

    assert answer2.trace is not None
    assert answer2.trace.cache_hit is True
    assert answer2.trace.cache_similarity is not None
    assert answer2.trace.cache_similarity > 0.99

    # First query should be a miss
    assert answer1.trace is not None
    assert answer1.trace.cache_hit is False


@pytest.mark.asyncio
async def test_cache_disabled_no_hit(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Some content.")
    llm = FakeLLM()

    async with RAG(embedder=FakeEmbedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        await rag.query("content")
        answer = await rag.query("content")

    assert answer.trace is not None
    assert answer.trace.cache_hit is False


@pytest.mark.asyncio
async def test_no_cache_for_temporal(tmp_path) -> None:
    """Queries with as_of set should never hit the semantic cache."""
    (tmp_path / "doc.txt").write_text("Policy content.")
    llm = FakeLLM()
    llm_call_count = [0]
    original_complete = llm.complete

    async def counting_complete(prompt):
        llm_call_count[0] += 1
        return await original_complete(prompt)

    llm.complete = counting_complete

    async with RAG(embedder=FakeEmbedder(), llm=llm, cache=True) as rag:
        await rag.ingest(str(tmp_path))
        cfg = QueryConfig(as_of="2024-01-01")
        answer1 = await rag.query("policy", config=cfg)
        answer2 = await rag.query("policy", config=cfg)

    # Both calls should have hit the LLM (no caching for temporal)
    assert answer1.trace is not None
    assert answer1.trace.cache_hit is False
    assert answer2.trace is not None
    assert answer2.trace.cache_hit is False


@pytest.mark.asyncio
async def test_cache_stats_via_rag(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Shipping policy: 5 days.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=True) as rag:
        await rag.ingest(str(tmp_path))
        await rag.query("shipping")  # miss
        await rag.query("shipping")  # hit

        assert rag.cache is not None
        stats = rag.cache.stats()

    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
