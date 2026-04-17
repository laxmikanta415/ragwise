"""RAG — the main user-facing async context manager."""
from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from ragwise.config import Answer, IngestResult, QueryConfig, RAGConfig
from ragwise.eval.chunks import ChunkEvalSchema
from ragwise.eval.schema import EvalSchema
from ragwise.generation.cache import CachedLLM, LLMCache
from ragwise.generation.llm import LLMProtocol, StreamingLLMProtocol, resolve_llm
from ragwise.generation.prompts import Assembler
from ragwise.indexing.base import EmbeddedDoc, SearchResult, VectorStore
from ragwise.ingestion.document import Document, hash_source
from ragwise.ingestion.loader import AutoLoader
from ragwise.retrieval.search import HybridSearcher
from ragwise.retrieval.sufficiency import SufficiencyChecker


def _resolve_store(spec: str | Any) -> VectorStore:
    if isinstance(spec, str):
        if spec == "memory":
            from ragwise.indexing.memory import InMemoryStore

            return InMemoryStore()
        if spec.startswith("lance://"):
            from ragwise.indexing.lance import LanceDBStore

            return LanceDBStore(uri=spec[len("lance://") :])
        if spec.startswith("postgresql://") or spec.startswith("postgres://"):
            from ragwise.indexing.pgvector import PgVectorStore

            return PgVectorStore(dsn=spec)
        raise ValueError(f"Unknown store spec: {spec!r}")
    return spec  # type: ignore[no-any-return]  # assume it's already a VectorStore


def _resolve_chunker(
    spec: str | Any,
    chunk_size: int,
    chunk_overlap: int,
    model: str,
    llm: Any = None,
) -> Any:
    if isinstance(spec, str):
        if spec == "recursive":
            from ragwise.ingestion.chunker import RecursiveChunker

            return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model)
        if spec == "structure":
            from ragwise.ingestion.chunker import StructureAwareChunker

            return StructureAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model)
        if spec == "contextual":
            from ragwise.ingestion.chunker import ContextualChunker

            if llm is None:
                raise ValueError("contextual chunker requires an LLM — pass llm= to RAG()")
            return ContextualChunker(llm=llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model)
        if spec == "hierarchical":
            from ragwise.ingestion.chunker import HierarchicalChunker

            return HierarchicalChunker(child_size=chunk_size // 4, parent_size=chunk_size, overlap=chunk_overlap, model=model)
        raise ValueError(f"Unknown chunker spec: {spec!r}")
    return spec


class RAG:
    """Async context manager for the full RAG pipeline.

    Basic usage::

        async with RAG(llm="openai/gpt-4o-mini") as rag:
            await rag.ingest("./docs/")
            answer = await rag.query("What is the refund policy?")

    Streaming::

        async for token in rag.stream_query("What changed in v2?"):
            print(token, end="", flush=True)

    Agent tool (no generation)::

        results = await rag.search("error code 0x80004005", top_k=5)
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        *,
        embedder: str | Any = "openai/text-embedding-3-small",
        store: str | Any = "memory",
        llm: str | Any = "openai/gpt-4o-mini",
        chunker: str | Any = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        reranker: str | None = None,
        cache: bool | str = True,
        batch_size: int = 32,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = RAGConfig(
                embedder=embedder,
                store=store,
                llm=llm,
                chunker=chunker,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                reranker=reranker,
                cache=cache,
                batch_size=batch_size,
            )

        self._store: VectorStore | None = None
        self._embedder: Any = None
        self._llm: LLMProtocol | None = None
        self._chunker: Any = None
        self._searcher: HybridSearcher | None = None
        self._assembler: Assembler | None = None
        self._sufficiency: SufficiencyChecker = SufficiencyChecker()
        self._tracer: Any = None  # LangfuseTracer | None

    async def __aenter__(self) -> RAG:
        from ragwise.embedding.base import resolve_embedder

        cfg = self._config
        self._store = _resolve_store(cfg.store)
        self._embedder = resolve_embedder(cfg.embedder)

        base_llm = resolve_llm(cfg.llm)
        if cfg.cache:
            backend = cfg.cache if isinstance(cfg.cache, str) else "memory"
            cache_obj = LLMCache(backend=backend)
            self._llm = CachedLLM(llm=base_llm, cache=cache_obj)
        else:
            self._llm = base_llm

        self._chunker = _resolve_chunker(
            cfg.chunker, cfg.chunk_size, cfg.chunk_overlap,
            model="gpt-4o",
            llm=base_llm,
        )

        self._searcher = HybridSearcher(
            store=self._store, embedder=self._embedder
        )
        self._assembler = Assembler()
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass  # InMemoryStore needs no teardown; connection cleanup deferred to v0.2

    async def ingest(
        self,
        path: str,
        *,
        glob: str = "**/*",
        force: bool = False,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        show_progress: bool = False,
    ) -> IngestResult:
        assert self._store is not None, "RAG must be used as a context manager"
        assert self._embedder is not None
        assert self._chunker is not None

        loader = AutoLoader()
        p = Path(path)

        indexed: dict[str, str] = {}
        if not force:
            indexed = await self._store.get_indexed_sources()

        files = [p] if p.is_file() else [f for f in p.glob(glob) if f.is_file()]
        total_files = len(files)

        _progress_bar: Any = None
        if show_progress:
            with contextlib.suppress(ImportError):
                from tqdm import tqdm
                _progress_bar = tqdm(total=total_files, desc="Indexing", unit="file")

        succeeded = 0
        failed = 0
        errors: list[str] = []
        failed_files: list[str] = []
        cfg = self._config

        for files_done, file_path in enumerate(files, start=1):
            try:
                current_hash = hash_source(file_path)
                if not force and indexed.get(str(file_path)) == current_hash:
                    if on_progress:
                        on_progress(str(file_path), files_done, total_files)
                    if _progress_bar is not None:
                        _progress_bar.update(1)
                    continue

                docs: list[Document] = loader.load(file_path)

                chunks: list[Document] = []
                for doc in docs:
                    chunks.extend(self._chunker.chunk(doc))

                if not chunks:
                    succeeded += 1
                    if on_progress:
                        on_progress(str(file_path), files_done, total_files)
                    if _progress_bar is not None:
                        _progress_bar.update(1)
                    continue

                texts = [c.text for c in chunks]
                all_embeddings: list[list[float]] = []
                for i in range(0, len(texts), cfg.batch_size):
                    batch = texts[i : i + cfg.batch_size]
                    batch_vecs = await self._embedder.embed(batch)
                    all_embeddings.extend(batch_vecs)

                # Delete stale chunks before upserting — prevents duplicates on re-ingest
                await self._store.delete(str(file_path))

                chunk_meta = metadata or {}
                embedded_docs = [
                    EmbeddedDoc(
                        id=chunk.id,
                        text=chunk.text,
                        source=chunk.source,
                        embedding=vec,
                        metadata={
                            **chunk.metadata,
                            **chunk_meta,
                            "content_hash": current_hash,
                            "tenant_id": tenant_id or "",
                        },
                    )
                    for chunk, vec in zip(chunks, all_embeddings, strict=True)
                ]
                await self._store.upsert(embedded_docs)
                succeeded += 1

            except Exception as exc:
                failed += 1
                errors.append(f"{file_path}: {exc}")
                failed_files.append(str(file_path))

            if on_progress:
                on_progress(str(file_path), files_done, total_files)
            if _progress_bar is not None:
                _progress_bar.update(1)

        if _progress_bar is not None:
            _progress_bar.close()

        return IngestResult(
            succeeded=succeeded,
            failed=failed,
            errors=errors,
            failed_files=failed_files,
        )

    async def query(
        self,
        question: str,
        *,
        config: QueryConfig | None = None,
    ) -> Answer:
        assert self._searcher is not None, "RAG must be used as a context manager"
        assert self._assembler is not None
        assert self._llm is not None

        qc = config or QueryConfig()

        results = await self._searcher.search(
            question,
            top_k=qc.top_k,
            alpha=qc.alpha,
            tenant_id=qc.tenant_id,
            allowed_sources=qc.allowed_sources if qc.allowed_sources else None,
        )

        sufficient = True
        if qc.check_sufficiency and results:
            query_vec = (await self._embedder.embed([question]))[0]
            sufficient = await self._sufficiency.check(query_vec, results, qc.sufficiency_threshold)

        if qc.max_context_tokens:
            self._assembler = Assembler(max_context_tokens=qc.max_context_tokens)

        prompt, citations = self._assembler.assemble(question, results)
        text = await self._llm.complete(prompt)

        answer = Answer(
            text=text,
            citations=citations if qc.include_citations else [],
            chunks_used=len(results),
            sufficient=sufficient,
        )

        if self._tracer is not None:
            with contextlib.suppress(Exception):
                await self._tracer.trace(question, answer, EvalSchema(query=question))

        return answer

    async def stream_query(
        self,
        question: str,
        *,
        config: QueryConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens as they arrive from the LLM.

        Usage::

            async for token in rag.stream_query("What changed in v2?"):
                print(token, end="", flush=True)

        Falls back to yielding the complete response as a single token if the
        underlying LLM does not implement ``StreamingLLMProtocol``.
        """
        assert self._searcher is not None, "RAG must be used as a context manager"
        assert self._assembler is not None
        assert self._llm is not None

        qc = config or QueryConfig()

        results = await self._searcher.search(
            question,
            top_k=qc.top_k,
            alpha=qc.alpha,
            tenant_id=qc.tenant_id,
            allowed_sources=qc.allowed_sources if qc.allowed_sources else None,
        )

        if qc.max_context_tokens:
            assembler = Assembler(max_context_tokens=qc.max_context_tokens)
        else:
            assembler = self._assembler

        prompt, _ = assembler.assemble(question, results)

        if isinstance(self._llm, StreamingLLMProtocol):
            async for token in self._llm.stream_complete(prompt):
                yield token
        else:
            yield await self._llm.complete(prompt)

    async def delete(self, source: str) -> None:
        """Remove all indexed chunks whose source matches *source*.

        Use for GDPR right-to-erasure or removing a specific document::

            await rag.delete("docs/old-policy.md")
        """
        assert self._store is not None, "RAG must be used as a context manager"
        await self._store.delete(source)

    async def list_sources(self) -> list[str]:
        """Return all distinct source paths currently indexed.

        Useful for building admin UIs or audit reports::

            sources = await rag.list_sources()
        """
        assert self._store is not None, "RAG must be used as a context manager"
        return await self._store.list_sources()

    async def update(self, source: str, path: str | None = None) -> IngestResult:
        """Re-ingest a source, replacing all its existing chunks.

        If *path* is omitted, *source* is used as the file path::

            await rag.update("docs/policy.md")
            await rag.update("docs/policy.md", path="/new/location/policy.md")
        """
        assert self._store is not None, "RAG must be used as a context manager"
        await self._store.delete(source)
        return await self.ingest(path or source, force=True)

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        config: QueryConfig | None = None,
    ) -> list[SearchResult]:
        """Return raw hybrid search results — no generation.

        Use this to build agent tools, custom pipelines, or debug retrieval::

            results = await rag.search("error code 0x80004005", top_k=5)
            for r in results:
                print(r.score, r.source, r.text[:120])
        """
        assert self._searcher is not None, "RAG must be used as a context manager"
        qc = config or QueryConfig()
        return await self._searcher.search(
            query,
            top_k=top_k,
            alpha=qc.alpha,
            tenant_id=qc.tenant_id,
            allowed_sources=qc.allowed_sources if qc.allowed_sources else None,
        )

    def set_tracer(self, tracer: Any) -> RAG:
        """Set a tracer for production observability (e.g. LangfuseTracer).

        Returns self for fluent chaining::

            rag.set_tracer(LangfuseTracer(...))
        """
        self._tracer = tracer
        return self

    async def eval_chunks(self, docs: list[Document]) -> ChunkEvalSchema:
        """Score a list of Document chunks for quality metrics."""
        assert self._embedder is not None, "RAG must be used as a context manager"
        from ragwise.eval.chunks import score_chunks

        return await score_chunks(docs, self._embedder, chunk_size=self._config.chunk_size)

    async def eval(self, dataset: list[dict[str, Any]]) -> EvalSchema:
        """Evaluate the RAG pipeline on a labeled dataset using RAGAS."""
        from ragwise.eval.metrics import evaluate_dataset

        return await evaluate_dataset(dataset)
