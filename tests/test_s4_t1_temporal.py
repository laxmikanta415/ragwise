"""S4-T1: Temporal metadata filtering — valid_from/until at ingest, as_of at query."""
from __future__ import annotations

import pytest

from ragwise import RAG, QueryConfig
from ragwise.testing import FakeEmbedder, FakeLLM


def _rag(**kw):
    return RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False, **kw)


@pytest.mark.asyncio
async def test_temporal_filter_no_as_of(tmp_path) -> None:
    """Without as_of, all docs are returned regardless of valid_from/until."""
    (tmp_path / "old.txt").write_text("Old policy document.")
    async with _rag() as rag:
        await rag.ingest(str(tmp_path), metadata={"valid_until": "2020-01-01"})
        results = await rag.search("policy", top_k=5)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_temporal_filter_past_date(tmp_path) -> None:
    """Docs with valid_until before as_of should be excluded."""
    (tmp_path / "old.txt").write_text("Expired policy from 2022.")
    async with _rag() as rag:
        await rag.ingest(str(tmp_path), metadata={"valid_until": "2022-12-31"})
        results = await rag.search(
            "expired policy",
            top_k=5,
            config=QueryConfig(as_of="2024-01-01"),
        )
    assert len(results) == 0


@pytest.mark.asyncio
async def test_temporal_filter_matching_date(tmp_path) -> None:
    """Docs with valid_from <= as_of should be returned."""
    (tmp_path / "current.txt").write_text("Active policy starting January 2024.")
    async with _rag() as rag:
        await rag.ingest(str(tmp_path), metadata={"valid_from": "2024-01-01"})
        results = await rag.search(
            "active policy",
            top_k=5,
            config=QueryConfig(as_of="2024-06-15"),
        )
    assert len(results) > 0


@pytest.mark.asyncio
async def test_temporal_filter_future_valid_from(tmp_path) -> None:
    """Docs with valid_from after as_of should be excluded."""
    (tmp_path / "future.txt").write_text("Policy starting next year.")
    async with _rag() as rag:
        await rag.ingest(str(tmp_path), metadata={"valid_from": "2026-01-01"})
        results = await rag.search(
            "future policy",
            top_k=5,
            config=QueryConfig(as_of="2024-01-01"),
        )
    assert len(results) == 0


@pytest.mark.asyncio
async def test_temporal_filter_no_metadata(tmp_path) -> None:
    """Docs without valid_from/until always returned when as_of is set."""
    (tmp_path / "doc.txt").write_text("A document without temporal metadata.")
    async with _rag() as rag:
        await rag.ingest(str(tmp_path))
        results = await rag.search(
            "document",
            top_k=5,
            config=QueryConfig(as_of="2024-01-01"),
        )
    assert len(results) > 0


@pytest.mark.asyncio
async def test_temporal_filter_version(tmp_path) -> None:
    """Version filter returns only matching version docs."""
    (tmp_path / "v2.txt").write_text("Version 2 API documentation.")
    (tmp_path / "v3.txt").write_text("Version 3 API documentation.")

    async with _rag() as rag:
        await rag.ingest(str(tmp_path / "v2.txt"), metadata={"version": "v2"})
        await rag.ingest(str(tmp_path / "v3.txt"), metadata={"version": "v3"})

        results_v3 = await rag.search(
            "API documentation",
            top_k=5,
            config=QueryConfig(version="v3"),
        )

    sources = {r.source for r in results_v3}
    assert all("v3" in s for s in sources)


@pytest.mark.asyncio
async def test_temporal_filter_trace_populated(tmp_path) -> None:
    """trace.temporal_filter_applied is True when as_of is set."""
    (tmp_path / "doc.txt").write_text("Some content.")
    async with _rag() as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("content", config=QueryConfig(as_of="2024-01-01"))

    assert answer.trace is not None
    assert answer.trace.temporal_filter_applied is True
    assert answer.trace.as_of_used == "2024-01-01"


@pytest.mark.asyncio
async def test_temporal_filter_as_of_now(tmp_path) -> None:
    """as_of='now' uses current timestamp."""
    (tmp_path / "doc.txt").write_text("Current content.")
    async with _rag() as rag:
        await rag.ingest(str(tmp_path))
        results = await rag.search("current", top_k=5, config=QueryConfig(as_of="now"))
    # Docs without valid_until always pass
    assert len(results) > 0
