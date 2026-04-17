"""SufficiencyChecker — centroid-distance check on retrieved results."""
from __future__ import annotations

import numpy as np

from ragwise.indexing.base import SearchResult


class SufficiencyChecker:
    """Returns True if retrieved results are likely sufficient to answer the query.

    Uses centroid-distance when result embeddings are available (memory backend),
    otherwise falls back to absolute max-score threshold. The previous normalized-mean
    approach always returned True for corpora with consistent scores — this fixes that.
    """

    async def check(
        self,
        query_vec: list[float],
        results: list[SearchResult],
        threshold: float = 0.6,
    ) -> bool:
        if not results:
            return True  # no docs → model will say "I don't know"

        embeddings = [r.embedding for r in results if r.embedding]
        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            qv = np.array(query_vec, dtype=np.float32)
            norm_c = float(np.linalg.norm(centroid))
            norm_q = float(np.linalg.norm(qv))
            if norm_c == 0 or norm_q == 0:
                return False
            cosine_sim = float(np.dot(centroid, qv) / (norm_c * norm_q))
            return cosine_sim >= threshold

        # Fallback: absolute max score (avoids normalized-mean always-true bug)
        max_score = max(r.score for r in results)
        return max_score >= threshold
