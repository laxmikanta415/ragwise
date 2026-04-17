"""S2-T5: Cohere Rerank + FlashRank + reranker factory."""
from __future__ import annotations

import pytest

from ragwise.indexing.base import SearchResult
from ragwise.retrieval.reranker import resolve_reranker


def _sr(id: str, text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(id=id, text=text, source="test.txt", score=score)


def test_resolve_reranker_none() -> None:
    assert resolve_reranker(None) is None


def test_resolve_reranker_passthrough_object() -> None:
    class FakeReranker:
        async def rerank(self, query, results, top_k=10):
            return results

    obj = FakeReranker()
    assert resolve_reranker(obj) is obj


def test_resolve_reranker_unknown_spec() -> None:
    with pytest.raises(ValueError, match="Unknown reranker spec"):
        resolve_reranker("unknown/model")


def test_resolve_reranker_cohere_string() -> None:
    import os
    os.environ.setdefault("COHERE_API_KEY", "test-key")
    try:
        import cohere  # noqa: F401
        reranker = resolve_reranker("cohere/rerank-v3.5")
        assert reranker.__class__.__name__ == "CohereReranker"
    except ImportError:
        pytest.skip("cohere not installed")


def test_resolve_reranker_flashrank_string() -> None:
    try:
        import flashrank  # noqa: F401
        reranker = resolve_reranker("flashrank")
        assert reranker.__class__.__name__ == "FlashRankReranker"
    except ImportError:
        pytest.skip("flashrank not installed")


@pytest.mark.asyncio
async def test_cohere_reranker_mock() -> None:
    """Mock the Cohere client to verify reranking reorders results."""
    import os
    from unittest.mock import MagicMock, patch

    try:
        import cohere  # noqa: F401
    except ImportError:
        pytest.skip("cohere not installed")

    os.environ["COHERE_API_KEY"] = "test-key"

    from ragwise.retrieval.rerankers.cohere import CohereReranker

    mock_result_0 = MagicMock()
    mock_result_0.index = 0
    mock_result_0.relevance_score = 0.9

    mock_result_1 = MagicMock()
    mock_result_1.index = 1
    mock_result_1.relevance_score = 0.3

    mock_response = MagicMock()
    mock_response.results = [mock_result_0, mock_result_1]

    mock_client = MagicMock()
    mock_client.rerank.return_value = mock_response

    with patch("cohere.ClientV2", return_value=mock_client):
        reranker = CohereReranker(model="rerank-v3.5")
        reranker._client = mock_client

        results = [_sr("a", "doc a", 0.5), _sr("b", "doc b", 0.8)]
        reranked = await reranker.rerank("query", results)

    assert reranked[0].id == "a"  # index 0 → highest score → first
    assert reranked[1].id == "b"
    assert reranked[0].score == pytest.approx(0.9)


def test_reranker_none_no_reranking() -> None:
    reranker = resolve_reranker(None)
    assert reranker is None
