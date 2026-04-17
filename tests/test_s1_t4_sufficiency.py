"""S1-T4: SufficiencyChecker fix — centroid-distance replaces normalized-mean."""
from __future__ import annotations

import numpy as np
import pytest

from ragwise.indexing.base import SearchResult
from ragwise.retrieval.sufficiency import SufficiencyChecker


def _make_result(score: float, embedding: list[float] | None = None) -> SearchResult:
    return SearchResult(
        id="x", text="t", source="s", score=score,
        embedding=embedding or [],
    )


def _unit_vec(n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ---------- centroid-distance path (embeddings available) ----------

@pytest.mark.asyncio
async def test_sufficiency_aligned_embeddings_returns_true() -> None:
    checker = SufficiencyChecker()
    # Query vec and chunk embeddings pointing in the same direction
    query_vec = _unit_vec(8, seed=0)
    results = [_make_result(0.9, embedding=query_vec) for _ in range(3)]
    assert await checker.check(query_vec, results, threshold=0.5) is True


@pytest.mark.asyncio
async def test_sufficiency_orthogonal_embeddings_returns_false() -> None:
    checker = SufficiencyChecker()
    # Orthogonal embeddings → cosine sim ≈ 0
    query_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    chunk_vec = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results = [_make_result(0.9, embedding=chunk_vec) for _ in range(3)]
    assert await checker.check(query_vec, results, threshold=0.5) is False


@pytest.mark.asyncio
async def test_sufficiency_no_longer_always_true_for_consistent_scores() -> None:
    """Regression: old normalized-mean always returned True when scores were consistent."""
    checker = SufficiencyChecker()
    query_vec = [1.0, 0.0, 0.0, 0.0]
    # All chunks have score=0.8 but point in the WRONG direction
    chunk_vec = [0.0, 0.0, 0.0, 1.0]
    results = [_make_result(0.8, embedding=chunk_vec) for _ in range(5)]
    # Must return False — the old implementation returned True here
    assert await checker.check(query_vec, results, threshold=0.5) is False


# ---------- absolute max-score fallback (no embeddings) ----------

@pytest.mark.asyncio
async def test_sufficiency_high_scores_returns_true() -> None:
    checker = SufficiencyChecker()
    query_vec = [0.1] * 8
    results = [_make_result(0.9), _make_result(0.85), _make_result(0.8)]
    assert await checker.check(query_vec, results, threshold=0.6) is True


@pytest.mark.asyncio
async def test_sufficiency_low_scores_returns_false() -> None:
    checker = SufficiencyChecker()
    query_vec = [0.1] * 8
    results = [_make_result(0.3), _make_result(0.2), _make_result(0.1)]
    assert await checker.check(query_vec, results, threshold=0.6) is False


@pytest.mark.asyncio
async def test_sufficiency_empty_results_returns_true() -> None:
    checker = SufficiencyChecker()
    assert await checker.check([0.1] * 8, [], threshold=0.6) is True


@pytest.mark.asyncio
async def test_sufficiency_threshold_boundary() -> None:
    checker = SufficiencyChecker()
    query_vec = [0.1] * 8
    # Exactly at threshold
    results = [_make_result(0.6)]
    assert await checker.check(query_vec, results, threshold=0.6) is True

    # Just below threshold
    results_below = [_make_result(0.59)]
    assert await checker.check(query_vec, results_below, threshold=0.6) is False
