"""Document chunkers — ChunkerProtocol, RecursiveChunker, StructureAwareChunker, ContextualChunker."""
from __future__ import annotations

import logging
import re
from typing import Any, Protocol

import anyio

from ragwise.ingestion.document import Document
from ragwise.utils.tokens import count_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)

# Mapping from tiktoken model names to encoding names accepted by Chonkie
_MODEL_TO_ENCODING: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}


def _resolve_tokenizer(model: str) -> str:
    """Return a tiktoken encoding name that Chonkie accepts for the given model."""
    return _MODEL_TO_ENCODING.get(model, "cl100k_base")


class ChunkerProtocol(Protocol):
    def chunk(self, doc: Document) -> list[Document]:
        ...


class RecursiveChunker:
    """Wraps Chonkie's RecursiveChunker; returns one Document per chunk."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        model: str = "gpt-4o",
    ) -> None:
        from chonkie import RecursiveChunker as _ChonkieRecursive

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        tokenizer = _resolve_tokenizer(model)
        self._chunker = _ChonkieRecursive(tokenizer=tokenizer, chunk_size=chunk_size)

    def chunk(self, doc: Document) -> list[Document]:
        if not doc.text.strip():
            return []
        # _chunker(str) always returns list[Chunk]; union type is for Sequence[str] overload
        raw_chunks = self._chunker(doc.text)
        chunks = [c for c in raw_chunks if c.text.strip()]  # type: ignore[union-attr]
        total = len(chunks)
        return [
            Document(
                text=c.text,  # type: ignore[union-attr]
                source=doc.source,
                metadata={**doc.metadata, "chunk_index": i, "chunk_count": total},
            )
            for i, c in enumerate(chunks)
        ]


_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


def _split_by_headings(text: str) -> list[tuple[str | None, str]]:
    """Split *text* at Markdown heading lines.

    Returns a list of (heading_title, section_text) pairs.
    The heading_title is the heading line stripped of '#' chars, or None for
    any leading content before the first heading.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    # Content before the first heading (if any)
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        first_line = section_text.splitlines()[0]
        heading_title = first_line.lstrip("#").strip()
        sections.append((heading_title, section_text))

    return sections


class StructureAwareChunker:
    """Splits at heading boundaries then recursively chunks each section."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        model: str = "gpt-4o",
    ) -> None:
        self._recursive = RecursiveChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model
        )

    def chunk(self, doc: Document) -> list[Document]:
        has_headings = "headings" in doc.metadata and doc.metadata["headings"]
        if not has_headings:
            # Fallback: plain recursive chunking, add section_heading=None
            chunks = self._recursive.chunk(doc)
            return [
                Document(
                    text=c.text,
                    source=c.source,
                    metadata={**c.metadata, "section_heading": None},
                )
                for c in chunks
            ]

        sections = _split_by_headings(doc.text)
        all_chunks: list[Document] = []
        for heading, section_text in sections:
            if not section_text.strip():
                continue
            section_doc = Document(
                text=section_text,
                source=doc.source,
                metadata={k: v for k, v in doc.metadata.items() if k != "chunk_index"},
            )
            section_chunks = self._recursive.chunk(section_doc)
            for c in section_chunks:
                all_chunks.append(
                    Document(
                        text=c.text,
                        source=c.source,
                        metadata={**c.metadata, "section_heading": heading},
                    )
                )

        # Re-index chunk_index and chunk_count across all sections
        total = len(all_chunks)
        return [
            Document(
                text=c.text,
                source=c.source,
                metadata={**c.metadata, "chunk_index": i, "chunk_count": total},
            )
            for i, c in enumerate(all_chunks)
        ]


class HierarchicalChunker:
    """Hierarchical chunking — small chunks for precise retrieval, parent text for generation.

    Small child chunks are indexed and retrieved; the larger parent chunk they came from
    is stored in ``metadata["parent_text"]`` and used by ``Assembler`` when building the prompt.

    Usage::

        chunks = HierarchicalChunker(child_size=128, parent_size=512).chunk(doc)
    """

    def __init__(
        self,
        child_size: int = 128,
        parent_size: int = 512,
        overlap: int = 32,
        model: str = "gpt-4o",
    ) -> None:
        self._parent_chunker = RecursiveChunker(
            chunk_size=parent_size, chunk_overlap=overlap, model=model
        )
        self._child_chunker = RecursiveChunker(
            chunk_size=child_size, chunk_overlap=overlap, model=model
        )

    def chunk(self, doc: Document) -> list[Document]:
        parent_chunks = self._parent_chunker.chunk(doc)
        result: list[Document] = []

        for parent_idx, parent in enumerate(parent_chunks):
            parent_id = f"{doc.id}_{parent_idx}"
            child_chunks = self._child_chunker.chunk(parent)

            for child_idx, child in enumerate(child_chunks):
                result.append(
                    Document(
                        text=child.text,
                        source=child.source,
                        metadata={
                            **child.metadata,
                            "parent_text": parent.text,
                            "parent_id": parent_id,
                            "child_index": child_idx,
                        },
                    )
                )

        return result


_CONTEXT_PROMPT = (
    "Given this document:\n{doc_text}\n\n"
    "Write 1-2 sentences describing what this chunk discusses in context of the whole document:\n"
    "{chunk_text}"
)
_MAX_DOC_TOKENS = 2000


class ContextualChunker:
    """Contextual chunking — prepends an LLM-generated summary to each chunk.

    Based on Anthropic's contextual retrieval research. Reduces retrieval failures by ~49%
    at the cost of LLM calls at ingest time.

    Usage::

        chunker = ContextualChunker(llm=my_llm, chunk_size=512)
        chunks = await chunker.chunk(doc)
    """

    def __init__(
        self,
        llm: Any,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        model: str = "gpt-4o",
    ) -> None:
        self._llm = llm
        self._recursive = RecursiveChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model
        )

    async def chunk(self, doc: Document) -> list[Document]:  # noqa: PLR6301
        """Chunk document and prepend LLM context summaries concurrently."""
        base_chunks = self._recursive.chunk(doc)
        if not base_chunks:
            return []

        # Truncate parent doc for the LLM prompt
        doc_tokens = count_tokens(doc.text)
        doc_text = (
            truncate_to_tokens(doc.text, _MAX_DOC_TOKENS)
            if doc_tokens > _MAX_DOC_TOKENS
            else doc.text
        )

        results: list[Document] = [Document(text="", source=doc.source) for _ in base_chunks]
        sem = anyio.Semaphore(10)

        async def _enhance(idx: int, chunk: Document) -> None:
            prompt = _CONTEXT_PROMPT.format(doc_text=doc_text, chunk_text=chunk.text)
            summary = ""
            async with sem:
                try:
                    summary = await self._llm.complete(prompt)
                except Exception:
                    logger.warning("ContextualChunker: LLM call failed for chunk %d", idx)

            enhanced_text = f"{summary}\n\n{chunk.text}" if summary else chunk.text
            results[idx] = Document(
                id=chunk.id,
                text=enhanced_text,
                source=chunk.source,
                metadata={**chunk.metadata, "has_context": bool(summary)},
            )

        async with anyio.create_task_group() as tg:
            for i, c in enumerate(base_chunks):
                tg.start_soon(_enhance, i, c)

        return results
