"""Chunk quality scoring — eval/chunks.py."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np

from ragwise.ingestion.document import Document
from ragwise.utils.tokens import count_tokens


@dataclass
class ChunkEvalSchema:
    """Objective quality metrics for a chunked document set."""

    avg_length_tokens: float = 0.0
    length_std: float = 0.0
    coherence_score: float = 0.0
    overlap_ratio: float = 0.0
    total_chunks: int = 0


async def score_chunks(
    docs: list[Document],
    embedder: Any,
    chunk_size: int = 512,
) -> ChunkEvalSchema:
    """Score a list of chunked documents for quality metrics.

    Args:
        docs: The chunks to evaluate.
        embedder: An embedder with an ``embed(texts)`` method.
        chunk_size: The configured chunk size (used to detect over-length chunks).

    Returns:
        ChunkEvalSchema with all fields populated.
    """
    total = len(docs)
    if total == 0:
        return ChunkEvalSchema()

    token_counts = [count_tokens(d.text) for d in docs]
    avg_length = statistics.mean(token_counts)
    std_length = statistics.stdev(token_counts) if total > 1 else 0.0

    over_length_threshold = chunk_size * 1.1
    over_count = sum(1 for tc in token_counts if tc > over_length_threshold)
    overlap_ratio = over_count / total

    if total < 2:
        coherence = 1.0
    else:
        texts = [d.text for d in docs]
        vecs = await embedder.embed(texts)
        matrix = np.array(vecs, dtype=np.float32)

        # Normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = matrix / norms

        # Cosine similarity between adjacent pairs
        sims = [float(np.dot(normed[i], normed[i + 1])) for i in range(len(normed) - 1)]
        coherence = float(statistics.mean(sims))
        coherence = max(0.0, min(1.0, coherence))

    return ChunkEvalSchema(
        avg_length_tokens=avg_length,
        length_std=std_length,
        coherence_score=coherence,
        overlap_ratio=overlap_ratio,
        total_chunks=total,
    )
