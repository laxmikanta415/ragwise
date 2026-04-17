"""Tests for HybridSearcher — S3-T8."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ragwise.indexing import EmbeddedDoc, InMemoryStore, SearchResult
from ragwise.retrieval import HybridSearcher


def _make_embedder(vec: list[float] | None = None) -> AsyncMock:
    """Return a mock embedder that always returns *vec* (or [0.1, 0.9])."""
    if vec is None:
        vec = [0.1, 0.9]
    mock = AsyncMock()
    mock.embed = AsyncMock(return_value=[vec])
    return mock


async def _populated_store() -> InMemoryStore:
    store = InMemoryStore()
    await store.upsert([
        EmbeddedDoc(id="a", text="apple fruit healthy", source="a.txt",
                    embedding=[1.0, 0.0]),
        EmbeddedDoc(id="b", text="banana yellow fruit", source="b.txt",
                    embedding=[0.9, 0.1]),
        EmbeddedDoc(id="c", text="carrot vegetable orange", source="c.txt",
                    embedding=[0.0, 1.0]),
    ])
    return store


@pytest.mark.asyncio
async def test_hybrid_searcher_returns_search_results() -> None:
    store = await _populated_store()
    embedder = _make_embedder([1.0, 0.0])
    searcher = HybridSearcher(store=store, embedder=embedder)
    results = await searcher.search("apple", top_k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_hybrid_searcher_top_k_respected() -> None:
    store = await _populated_store()
    embedder = _make_embedder([1.0, 0.0])
    searcher = HybridSearcher(store=store, embedder=embedder)
    results = await searcher.search("fruit", top_k=2)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_hybrid_searcher_empty_store_returns_empty() -> None:
    store = InMemoryStore()
    embedder = _make_embedder([1.0, 0.0])
    searcher = HybridSearcher(store=store, embedder=embedder)
    results = await searcher.search("anything")
    assert results == []


@pytest.mark.asyncio
async def test_hybrid_searcher_doc_in_both_ranks_higher() -> None:
    """Doc appearing in both dense AND sparse results should rank first via RRF."""
    store = InMemoryStore()
    # doc "overlap" has high dense similarity AND contains the query word
    await store.upsert([
        EmbeddedDoc(id="overlap", text="fruit apple fresh",
                    source="o.txt", embedding=[1.0, 0.0]),
        EmbeddedDoc(id="dense_only", text="zzzz aaaa bbbb",
                    source="d.txt", embedding=[0.99, 0.01]),
        EmbeddedDoc(id="sparse_only", text="fruit apple fresh",
                    source="s.txt", embedding=[0.0, 1.0]),
    ])
    embedder = _make_embedder([1.0, 0.0])
    searcher = HybridSearcher(store=store, embedder=embedder)
    results = await searcher.search("fruit apple", top_k=3)
    ids = [r.id for r in results]
    # "overlap" must rank first (present in both dense + sparse)
    assert ids[0] == "overlap"


@pytest.mark.asyncio
async def test_hybrid_searcher_results_sorted_descending() -> None:
    store = await _populated_store()
    embedder = _make_embedder([1.0, 0.0])
    searcher = HybridSearcher(store=store, embedder=embedder)
    results = await searcher.search("apple", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_hybrid_searcher_embeds_query_once() -> None:
    store = await _populated_store()
    embedder = _make_embedder([1.0, 0.0])
    searcher = HybridSearcher(store=store, embedder=embedder)
    await searcher.search("apple", top_k=2)
    # embed should be called exactly once per search
    assert embedder.embed.call_count == 1
