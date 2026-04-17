"""Tests for VectorStore ABC, EmbeddedDoc, SearchResult, InMemoryStore — S3-T4/T5."""
from __future__ import annotations

import dataclasses

import pytest

from ragwise.indexing import EmbeddedDoc, InMemoryStore, SearchResult, VectorStore


# ---------------------------------------------------------------------------
# S3-T4 — base types
# ---------------------------------------------------------------------------

def test_embedded_doc_instantiates() -> None:
    doc = EmbeddedDoc(id="x", text="hello", source="a.txt", embedding=[0.1, 0.2])
    assert doc.id == "x"
    assert doc.embedding == [0.1, 0.2]


def test_search_result_is_frozen() -> None:
    sr = SearchResult(id="x", text="t", source="s", score=0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sr.score = 0.0  # type: ignore[misc]


def test_vector_store_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


def test_concrete_subclass_can_instantiate() -> None:
    class ConcreteStore(VectorStore):
        async def upsert(self, docs: list[EmbeddedDoc]) -> None: ...
        async def dense_search(self, query_vec: list[float], top_k: int) -> list[SearchResult]: return []
        async def sparse_search(self, query: str, top_k: int) -> list[SearchResult]: return []
        async def delete(self, source: str) -> None: ...
        async def get_indexed_sources(self) -> dict[str, str]: return {}

    store = ConcreteStore()
    assert isinstance(store, VectorStore)


# ---------------------------------------------------------------------------
# S3-T5 — InMemoryStore
# ---------------------------------------------------------------------------

def _make_doc(id: str = "d1", text: str = "hello world", source: str = "src.txt") -> EmbeddedDoc:
    return EmbeddedDoc(id=id, text=text, source=source, embedding=[0.1, 0.9, 0.0])


@pytest.mark.asyncio
async def test_inmemory_upsert_and_dense_search() -> None:
    store = InMemoryStore()
    doc1 = EmbeddedDoc(id="a", text="alpha", source="s1.txt", embedding=[1.0, 0.0])
    doc2 = EmbeddedDoc(id="b", text="beta", source="s2.txt", embedding=[0.0, 1.0])
    await store.upsert([doc1, doc2])

    results = await store.dense_search([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "a"  # closest to [1, 0]


@pytest.mark.asyncio
async def test_inmemory_dense_search_sorted_desc() -> None:
    store = InMemoryStore()
    docs = [
        EmbeddedDoc(id=str(i), text=f"text{i}", source="s", embedding=[float(i), 0.0])
        for i in range(1, 4)
    ]
    await store.upsert(docs)
    results = await store.dense_search([3.0, 0.0], top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_inmemory_sparse_search_keyword() -> None:
    store = InMemoryStore()
    doc1 = EmbeddedDoc(id="a", text="the quick brown fox", source="s", embedding=[0.1])
    doc2 = EmbeddedDoc(id="b", text="lorem ipsum dolor", source="s", embedding=[0.2])
    await store.upsert([doc1, doc2])

    results = await store.sparse_search("fox", top_k=2)
    assert len(results) >= 1
    # doc1 should rank higher since it contains "fox"
    assert results[0].id == "a"


@pytest.mark.asyncio
async def test_inmemory_delete_by_source() -> None:
    store = InMemoryStore()
    await store.upsert([
        EmbeddedDoc(id="a", text="t", source="file_a.txt", embedding=[1.0]),
        EmbeddedDoc(id="b", text="t", source="file_b.txt", embedding=[0.5]),
    ])
    await store.delete("file_a.txt")
    results = await store.dense_search([1.0], top_k=5)
    ids = [r.id for r in results]
    assert "a" not in ids
    assert "b" in ids


@pytest.mark.asyncio
async def test_inmemory_get_indexed_sources() -> None:
    store = InMemoryStore()
    await store.upsert([
        EmbeddedDoc(id="a", text="t", source="a.txt", embedding=[1.0],
                    metadata={"content_hash": "abc123"}),
        EmbeddedDoc(id="b", text="t", source="b.txt", embedding=[0.5],
                    metadata={"content_hash": "def456"}),
    ])
    sources = await store.get_indexed_sources()
    assert sources == {"a.txt": "abc123", "b.txt": "def456"}


@pytest.mark.asyncio
async def test_inmemory_upsert_replaces_by_id() -> None:
    store = InMemoryStore()
    await store.upsert([EmbeddedDoc(id="x", text="old", source="s", embedding=[1.0])])
    await store.upsert([EmbeddedDoc(id="x", text="new", source="s", embedding=[1.0])])
    results = await store.dense_search([1.0], top_k=5)
    assert len(results) == 1
    assert results[0].text == "new"


@pytest.mark.asyncio
async def test_inmemory_empty_store_returns_empty() -> None:
    store = InMemoryStore()
    assert await store.dense_search([1.0], top_k=5) == []
    assert await store.sparse_search("query", top_k=5) == []
    assert await store.get_indexed_sources() == {}


@pytest.mark.asyncio
async def test_inmemory_delete_then_get_sources() -> None:
    store = InMemoryStore()
    await store.upsert([EmbeddedDoc(id="a", text="t", source="a.txt", embedding=[1.0])])
    await store.delete("a.txt")
    sources = await store.get_indexed_sources()
    assert "a.txt" not in sources
