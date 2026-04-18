"""S4-T4: Document TTL + staleness — list_stale, purge_stale, staleness_report."""
from __future__ import annotations

import pytest

from ragwise import RAG
from ragwise.lifecycle import PurgeResult, StaleSource
from ragwise.testing import FakeEmbedder, FakeLLM


@pytest.mark.asyncio
async def test_list_stale_returns_expired(tmp_path) -> None:
    (tmp_path / "old.txt").write_text("Old policy from 2022.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path), metadata={"valid_until": "2022-01-01"})
        expired, _ = await rag.list_stale(as_of="2024-01-01")

    assert len(expired) >= 1
    sources = [s.source for s in expired]
    assert any("old.txt" in s for s in sources)


def test_stale_source_fields() -> None:
    ss = StaleSource(source="docs/policy.md", valid_until="2022-01-01", chunk_count=3, expired_days_ago=365)
    assert ss.source == "docs/policy.md"
    assert ss.expired_days_ago == 365


@pytest.mark.asyncio
async def test_list_stale_ignores_no_expiry(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("No expiry metadata.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        expired, _ = await rag.list_stale(as_of="2024-01-01")

    assert len(expired) == 0


@pytest.mark.asyncio
async def test_list_stale_future_expiry(tmp_path) -> None:
    (tmp_path / "future.txt").write_text("Future policy document.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path), metadata={"valid_until": "2099-01-01"})
        expired, _ = await rag.list_stale(as_of="2024-01-01")

    assert len(expired) == 0


@pytest.mark.asyncio
async def test_purge_stale_removes_chunks(tmp_path) -> None:
    (tmp_path / "old.txt").write_text("Expired content.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path), metadata={"valid_until": "2020-01-01"})
        result = await rag.purge_stale(as_of="2024-01-01")
        remaining = await rag.search("expired content", top_k=5)

    assert isinstance(result, PurgeResult)
    assert len(result.purged_sources) >= 1
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_purge_stale_returns_result(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("First expired doc.")
    (tmp_path / "b.txt").write_text("Second expired doc.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path / "a.txt"), metadata={"valid_until": "2020-01-01"})
        await rag.ingest(str(tmp_path / "b.txt"), metadata={"valid_until": "2021-01-01"})
        result = await rag.purge_stale(as_of="2024-01-01")

    assert len(result.purged_sources) == 2
    assert result.purged_chunks >= 2


@pytest.mark.asyncio
async def test_staleness_report_format(tmp_path) -> None:
    (tmp_path / "expired.txt").write_text("Old expired doc.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path / "expired.txt"), metadata={"valid_until": "2020-01-01"})
        report = await rag.staleness_report(as_of="2024-01-01")

    assert "Expired" in report
    assert "Expiring soon" in report
    assert "Total sources" in report


@pytest.mark.asyncio
async def test_expiring_soon_days(tmp_path) -> None:
    """Docs expiring within N days appear in expiring_soon list."""
    from datetime import datetime, timedelta

    soon = (datetime.utcnow() + timedelta(days=10)).strftime("%Y-%m-%d")
    (tmp_path / "soon.txt").write_text("Doc expiring soon.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path), metadata={"valid_until": soon})
        _, expiring_soon = await rag.list_stale(expiring_soon_days=30)

    assert len(expiring_soon) >= 1
    sources = [s.source for s in expiring_soon]
    assert any("soon.txt" in s for s in sources)


@pytest.mark.asyncio
async def test_list_stale_now(tmp_path) -> None:
    """as_of='now' (default) uses current time correctly."""
    (tmp_path / "past.txt").write_text("Already expired.")
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path), metadata={"valid_until": "2020-06-01"})
        expired, _ = await rag.list_stale()  # defaults to "now"

    assert len(expired) >= 1
