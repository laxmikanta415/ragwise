"""RAGAS-backed eval + pytest helper for CI quality gates."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragwise.eval.schema import EvalSchema

if TYPE_CHECKING:
    pass


def _import_ragas() -> Any:
    try:
        from ragas import evaluate  # noqa: F401

        return evaluate
    except ImportError as exc:
        raise ImportError("pip install ragwise[eval]") from exc


async def evaluate_dataset(dataset: list[dict[str, Any]]) -> EvalSchema:
    """Run RAGAS evaluation on a dataset and return a populated EvalSchema.

    Args:
        dataset: List of dicts with keys: question, answer, contexts, ground_truth.

    Returns:
        EvalSchema with faithfulness, answer_relevance, context_recall, context_precision.
    """
    ragas_evaluate = _import_ragas()

    try:
        from datasets import Dataset  # noqa: F401
        from ragas.metrics import (  # noqa: F401
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError("pip install ragwise[eval]") from exc

    hf_dataset = Dataset.from_list(dataset)
    result = ragas_evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores = result.to_pandas()
    avg = scores.mean(numeric_only=True)

    return EvalSchema(
        faithfulness=float(avg.get("faithfulness", 0.0)),
        answer_relevance=float(avg.get("answer_relevancy", 0.0)),
        context_recall=float(avg.get("context_recall", 0.0)),
        context_precision=float(avg.get("context_precision", 0.0)),
    )


def assert_eval_passes(
    eval_result: EvalSchema,
    *,
    min_faithfulness: float = 0.7,
    min_relevance: float = 0.7,
) -> None:
    """Raise AssertionError if the eval result does not meet quality thresholds.

    Designed for use in pytest — call after ``rag.eval(dataset)``::

        scores = await rag.eval(dataset)
        assert_eval_passes(scores, min_faithfulness=0.8, min_relevance=0.7)
    """
    if not eval_result.is_passing(min_faithfulness=min_faithfulness, min_relevance=min_relevance):
        raise AssertionError(
            f"Eval did not pass thresholds: "
            f"faithfulness={eval_result.faithfulness:.3f} (min={min_faithfulness}), "
            f"answer_relevance={eval_result.answer_relevance:.3f} (min={min_relevance})"
        )
