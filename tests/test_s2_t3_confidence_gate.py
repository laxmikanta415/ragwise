"""S2-T3: Confidence-gated answers — skip LLM when retrieval is insufficient."""
from __future__ import annotations

import pytest

from ragwise import RAG, RAGConfig


def _make_embedder(seed_offset: int = 0):
    from unittest.mock import AsyncMock, MagicMock
    import numpy as np
    emb = MagicMock()

    def _embed(texts):
        # Return vectors pointing in direction based on text hash so we get real similarity
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32) + seed_offset)
            v = rng.standard_normal(64).astype(np.float32)
            vecs.append((v / np.linalg.norm(v)).tolist())
        return vecs

    emb.embed = AsyncMock(side_effect=_embed)
    return emb


def _make_llm():
    from unittest.mock import AsyncMock, MagicMock
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="LLM was called")
    return llm


@pytest.mark.asyncio
async def test_confidence_gate_disabled_by_default(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Refund policy information.")
    llm = _make_llm()

    async with RAG(embedder=_make_embedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("zzzzz unrelated query zzzzz")

    # Default threshold=0.0 means gate never fires
    assert answer.has_sufficient_context is True
    llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_confidence_gate_passes_on_relevant_query(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Refund policy: 30 day returns accepted.")
    llm = _make_llm()

    # Threshold below minimum cosine similarity (-1.0) — gate always passes
    cfg = RAGConfig(confidence_threshold=-1.0)
    async with RAG(config=cfg, embedder=_make_embedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund policy")

    assert answer.has_sufficient_context is True


@pytest.mark.asyncio
async def test_confidence_gate_fires_llm_not_called(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Information about cats and dogs.")
    llm = _make_llm()

    # Set an impossibly high threshold — always fires
    cfg = RAGConfig(confidence_threshold=0.9999)
    async with RAG(config=cfg, embedder=_make_embedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("quantum physics research")

    assert answer.has_sufficient_context is False
    # LLM must NOT have been called
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_insufficient_response_text(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Information about cats.")
    cfg = RAGConfig(confidence_threshold=0.9999)
    llm = _make_llm()

    async with RAG(config=cfg, embedder=_make_embedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("something unrelated")

    assert answer.text == cfg.insufficient_response
    assert "could not find" in answer.text.lower()


@pytest.mark.asyncio
async def test_custom_insufficient_response(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Some content.")
    custom_msg = "Sorry, no relevant documents found."
    cfg = RAGConfig(confidence_threshold=0.9999, insufficient_response=custom_msg)
    llm = _make_llm()

    async with RAG(config=cfg, embedder=_make_embedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("unrelated question")

    assert answer.text == custom_msg


@pytest.mark.asyncio
async def test_top_retrieved_populated_when_insufficient(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Some content about various topics.")
    cfg = RAGConfig(confidence_threshold=0.9999)
    llm = _make_llm()

    async with RAG(config=cfg, embedder=_make_embedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("something else")

    assert answer.top_retrieved is not None
    assert len(answer.top_retrieved) <= 3


@pytest.mark.asyncio
async def test_stream_query_insufficient_yields_message(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Some content.")
    cfg = RAGConfig(confidence_threshold=0.9999)
    llm = _make_llm()

    tokens: list[str] = []
    async with RAG(config=cfg, embedder=_make_embedder(), llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        async for token in rag.stream_query("unrelated"):
            tokens.append(token)

    full_text = "".join(tokens)
    assert full_text == cfg.insufficient_response
    llm.complete.assert_not_called()
