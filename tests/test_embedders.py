"""Tests for EmbedderProtocol, resolve_embedder, OpenAIEmbedder — S3-T1/T2/T3."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragwise.embedding import EmbedderProtocol, OpenAIEmbedder, resolve_embedder


# ---------------------------------------------------------------------------
# S3-T1 — resolve_embedder
# ---------------------------------------------------------------------------

def test_resolve_embedder_openai_returns_openai_embedder() -> None:
    embedder = resolve_embedder("openai/text-embedding-3-small")
    assert isinstance(embedder, OpenAIEmbedder)


def test_resolve_embedder_local_returns_sentence_transformer() -> None:
    pytest.importorskip("sentence_transformers")
    from ragwise.embedding.local import SentenceTransformerEmbedder

    embedder = resolve_embedder("local/all-MiniLM-L6-v2")
    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_resolve_embedder_unknown_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="Unknown embedder"):
        resolve_embedder("unknown/model")


def test_resolve_embedder_passthrough_duck_type() -> None:
    mock = MagicMock()
    mock.embed = AsyncMock()
    result = resolve_embedder(mock)
    assert result is mock


def test_resolve_embedder_non_embedder_object_raises() -> None:
    with pytest.raises(ValueError):
        resolve_embedder(42)


def test_embedder_protocol_structural() -> None:
    embedder: EmbedderProtocol = OpenAIEmbedder()  # type: ignore[assignment]
    assert callable(embedder.embed)


# ---------------------------------------------------------------------------
# S3-T2 — OpenAIEmbedder
# ---------------------------------------------------------------------------

def test_openai_embedder_instantiates_without_api_key() -> None:
    # Should not raise even without OPENAI_API_KEY in env
    embedder = OpenAIEmbedder()
    assert embedder.model == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_openai_embedder_empty_returns_empty() -> None:
    embedder = OpenAIEmbedder()
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_openai_embedder_returns_two_embeddings() -> None:
    fake_emb = [0.1, 0.2, 0.3]
    item1 = MagicMock()
    item1.embedding = fake_emb
    item2 = MagicMock()
    item2.embedding = fake_emb

    mock_response = MagicMock()
    mock_response.data = [item1, item2]

    embedder = OpenAIEmbedder()
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    embedder._client = mock_client  # inject before lazy creation

    result = await embedder.embed(["hello", "world"])
    assert len(result) == 2
    assert result[0] == fake_emb


@pytest.mark.asyncio
async def test_openai_embedder_batching_call_count() -> None:
    """100 texts with batch_size=32 should produce 4 API calls."""
    fake_item = MagicMock()
    fake_item.embedding = [0.0]

    async def side_effect(**kwargs: object) -> MagicMock:
        batch = kwargs.get("input", [])
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.0]) for _ in batch]  # type: ignore[arg-type]
        return resp

    embedder = OpenAIEmbedder(batch_size=32)
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=side_effect)
    embedder._client = mock_client

    result = await embedder.embed(["x"] * 100)
    assert len(result) == 100
    assert mock_client.embeddings.create.call_count == 4
