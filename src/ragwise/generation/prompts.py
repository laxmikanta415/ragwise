"""RAG prompt template and context Assembler."""
from __future__ import annotations

from ragwise.indexing.base import SearchResult
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
    ) -> tuple[str, list[str]]:
        """Return (prompt_text, citation_sources).

        Context chunks are added until the token budget is exhausted.
        """
        question_tokens = count_tokens(query, self.model)
        budget = self.max_context_tokens - _TEMPLATE_OVERHEAD - question_tokens

        context_parts: list[str] = []
        citations: list[str] = []
        seen_sources: set[str] = set()
        used_tokens = 0

        for result in results:
            context_text = result.metadata.get("parent_text", result.text)  # use parent for generation
            chunk = f"[Source: {result.source}]\n{context_text}"
            chunk_tokens = count_tokens(chunk, self.model)
            if used_tokens + chunk_tokens > budget:
                break
            context_parts.append(chunk)
            used_tokens += chunk_tokens
            if result.source not in seen_sources:
                seen_sources.add(result.source)
                citations.append(result.source)

        context = "\n\n".join(context_parts)
        prompt = RAG_PROMPT.format(context=context, question=query)
        return prompt, citations
