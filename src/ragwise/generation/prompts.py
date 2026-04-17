"""RAG prompt template and context Assembler."""
from __future__ import annotations

from ragwise.indexing.base import SearchResult
from ragwise.models import Citation
from ragwise.utils.tokens import count_tokens

RAG_PROMPT = """\
You are a helpful assistant. Answer the question using only the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

_TEMPLATE_OVERHEAD = count_tokens(RAG_PROMPT.format(context="", question=""))


class Assembler:
    """Builds RAG prompts from search results within a token budget."""

    def __init__(self, max_context_tokens: int = 8000, model: str = "gpt-4o") -> None:
        self.max_context_tokens = max_context_tokens
        self.model = model

    def assemble(
        self, query: str, results: list[SearchResult]
    ) -> tuple[str, list[Citation], list[SearchResult]]:
        """Return (prompt_text, citations, dropped_results).

        Citations contain the actual passage text and metadata.
        dropped_results are chunks that exceeded the token budget.
        """
        question_tokens = count_tokens(query, self.model)
        budget = self.max_context_tokens - _TEMPLATE_OVERHEAD - question_tokens

        context_parts: list[str] = []
        citations: list[Citation] = []
        seen_sources: set[str] = set()
        used_tokens = 0
        dropped: list[SearchResult] = []

        for result in results:
            context_text = result.metadata.get("parent_text", result.text)
            chunk = f"[Source: {result.source}]\n{context_text}"
            chunk_tokens = count_tokens(chunk, self.model)
            if used_tokens + chunk_tokens > budget:
                dropped.append(result)
                continue
            context_parts.append(chunk)
            used_tokens += chunk_tokens
            if result.source not in seen_sources:
                seen_sources.add(result.source)
            citations.append(
                Citation(
                    text=context_text,
                    source=result.source,
                    chunk_id=result.id,
                    final_score=result.score,
                    bm25_score=result.bm25_score,
                    dense_score=result.dense_score,
                    page=result.metadata.get("page"),
                    char_start=result.metadata.get("char_start", 0),
                    char_end=result.metadata.get("char_end", 0),
                )
            )

        context = "\n\n".join(context_parts)
        prompt = RAG_PROMPT.format(context=context, question=query)
        return prompt, citations, dropped
