"""SufficiencyChecker — cosine-proxy check on retrieved results."""
from __future__ import annotations

from ragwise.indexing.base import SearchResult


class SufficiencyChecker:
    """Returns True if retrieved results are likely sufficient to answer the query."""

    async def check(
        self,
        query_vec: list[float],
        results: list[SearchResult],
        threshold: float = 0.6,
    ) -> bool:
        if not results:
            return True  # no docs → return True (model will say "I don't know")

        scores = [r.score for r in results]
        max_score = max(scores)

        if max_score <= 0:
            return False

        # Normalize to [0, 1] and take mean
        normalised = [s / max_score for s in scores]
        mean_score = sum(normalised) / len(normalised)
        return mean_score >= threshold
