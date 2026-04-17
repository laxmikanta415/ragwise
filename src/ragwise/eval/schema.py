"""EvalSchema — unified evaluation result dataclass."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalSchema:
    """Unified eval result. All float fields are in [0, 1] except latency_ms."""

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    latency_ms: float = 0.0
    chunks_used: int = 0
    sufficient: bool = True
    query: str = ""

    def is_passing(self, min_faithfulness: float = 0.7, min_relevance: float = 0.7) -> bool:
        """Return True if both faithfulness and answer_relevance meet thresholds."""
        return self.faithfulness >= min_faithfulness and self.answer_relevance >= min_relevance
