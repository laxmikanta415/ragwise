"""LangfuseTracer — production tracing for ragwise queries."""
from __future__ import annotations

import logging
from typing import Any

from ragwise.config import Answer
from ragwise.eval.schema import EvalSchema

logger = logging.getLogger(__name__)


def _import_langfuse() -> Any:
    try:
        from langfuse import Langfuse  # noqa: F401

        return Langfuse
    except ImportError as exc:
        raise ImportError("pip install ragwise[eval]") from exc


class LangfuseTracer:
    """Sends ragwise query traces to Langfuse for production observability.

    Example::

        tracer = LangfuseTracer(public_key="pk-...", secret_key="sk-...")
        async with RAG(llm="openai/gpt-4o-mini") as rag:
            rag.set_tracer(tracer)
            answer = await rag.query("What is ragwise?")  # trace sent automatically
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
    ) -> None:
        Langfuse = _import_langfuse()
        self._client: Any = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

    async def trace(self, query: str, answer: Answer, schema: EvalSchema) -> None:
        """Send a trace to Langfuse. Failures are logged and swallowed."""
        try:
            self._client.trace(
                name="ragwise-query",
                input={"query": query},
                output={"answer": answer.text, "citations": answer.citations},
                metadata={
                    "chunks_used": answer.chunks_used,
                    "sufficient": answer.sufficient,
                    "faithfulness": schema.faithfulness,
                    "answer_relevance": schema.answer_relevance,
                    "context_recall": schema.context_recall,
                    "context_precision": schema.context_precision,
                    "latency_ms": schema.latency_ms,
                },
            )
        except Exception:
            logger.warning("LangfuseTracer: trace failed", exc_info=True)
