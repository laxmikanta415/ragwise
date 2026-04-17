"""Reciprocal Rank Fusion — fuses N ranked lists of doc IDs into a single ranking."""
from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse N ranked lists of doc IDs using Reciprocal Rank Fusion.

    Args:
        rankings: Each inner list is a ranked list of doc IDs (best first).
        k: Smoothing constant (default 60, matching RAGFlow convention).
           Larger k → scores closer together (less aggressive rank differences).

    Returns:
        List of (doc_id, score) tuples sorted descending by score.
        Ties broken alphabetically by doc_id for determinism.
    """
    scores: dict[str, float] = {}

    for ranked_list in rankings:
        for rank_i, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_i)

    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
