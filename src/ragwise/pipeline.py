"""RAG — the main user-facing async context manager."""
from __future__ import annotations

import contextlib
import time
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
from ragwise.models import Citation, QueryTrace, RetrievedChunk
from ragwise.retrieval.reranker import resolve_reranker
from ragwise.retrieval.search import HybridSearcher
from ragwise.retrieval.sufficiency import SufficiencyChecker
from ragwise.utils.tokens import count_tokens

_UNSET: Any = object()  # sentinel for detecting "not passed to __init__"

# LLM pricing per token (USD). 0.0 for local/unknown models.
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini":      {"input": 0.15 / 1e6,  "output": 0.60 / 1e6},
    "gpt-4o":           {"input": 2.50 / 1e6,  "output": 10.00 / 1e6},
    "gpt-4.1-mini":     {"input": 0.40 / 1e6,  "output": 1.60 / 1e6},
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1e6, "output": 4.00 / 1e6},
    "claude-sonnet-4-6": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-opus-4-7":   {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
}


def _estimate_cost(model_spec: str, prompt_tokens: int, completion_tokens: int) -> float:
    model = model_spec.split("/", 1)[-1] if "/" in model_spec else model_spec
    pricing = _PRICING.get(model)
    if pricing is None:
        return 0.0
    return pricing["input"] * prompt_tokens + pricing["output"] * completion_tokens


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
        embedder: str | Any = _UNSET,
        store: str | Any = _UNSET,
        llm: str | Any = _UNSET,
        chunker: str | Any = _UNSET,
        chunk_size: Any = _UNSET,
        chunk_overlap: Any = _UNSET,
        reranker: Any = _UNSET,
        cache: Any = _UNSET,
        batch_size: Any = _UNSET,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = RAGConfig(
                embedder="openai/text-embedding-3-small" if embedder is _UNSET else embedder,
                store="memory" if store is _UNSET else store,
                llm="openai/gpt-4o-mini" if llm is _UNSET else llm,
                chunker="recursive" if chunker is _UNSET else chunker,
                chunk_size=512 if chunk_size is _UNSET else chunk_size,
                chunk_overlap=64 if chunk_overlap is _UNSET else chunk_overlap,
                reranker=None if reranker is _UNSET else reranker,
                cache=True if cache is _UNSET else cache,
                batch_size=32 if batch_size is _UNSET else batch_size,
            )

        # Explicit object overrides — used in __aenter__ when config= and kwargs are mixed
        self._embedder_kw: Any = None if embedder is _UNSET else embedder
        self._llm_kw: Any = None if llm is _UNSET else llm
        self._store_kw: Any = None if store is _UNSET else store
        self._reranker_kw: Any = _UNSET if reranker is _UNSET else reranker
        self._cache_kw: Any = _UNSET if cache is _UNSET else cache

        self._store: VectorStore | None = None
        self._embedder: Any = None
        self._llm: LLMProtocol | None = None
        self._reranker: Any = None
        self._chunker: Any = None
        self._searcher: HybridSearcher | None = None
        self._assembler: Assembler | None = None
        self._sufficiency: SufficiencyChecker = SufficiencyChecker()
        self._tracer: Any = None  # LangfuseTracer | None
        self._semantic_cache: Any = None  # MemorySemanticCache | None

    async def __aenter__(self) -> RAG:
        from ragwise.embedding.base import resolve_embedder

        cfg = self._config

        # Use explicit kwarg object if passed, otherwise resolve from config spec
        if self._embedder_kw is not None and not isinstance(self._embedder_kw, str):
            self._embedder = self._embedder_kw
        else:
            self._embedder = resolve_embedder(cfg.embedder)

        if self._store_kw is not None and not isinstance(self._store_kw, str):
            self._store = self._store_kw
        else:
            self._store = _resolve_store(cfg.store)

        if self._llm_kw is not None and not isinstance(self._llm_kw, str):
            base_llm = self._llm_kw
        else:
            base_llm = resolve_llm(cfg.llm)

        # cache: explicit False overrides config's True
        use_cache = cfg.cache if self._cache_kw is _UNSET else self._cache_kw
        if use_cache:
            from ragwise.generation.semantic_cache import MemorySemanticCache
            backend = use_cache if isinstance(use_cache, str) else "memory"
            # Semantic cache at query level (replaces SHA-256 prompt cache)
            if isinstance(backend, str) and backend.startswith("redis"):
                from ragwise.generation.semantic_cache import RedisSemanticCache
                self._semantic_cache = RedisSemanticCache(url=backend)
            else:
                self._semantic_cache = MemorySemanticCache()
            # Keep LLM-level cache for streaming paths
            cache_obj = LLMCache(backend=backend)
            self._llm = CachedLLM(llm=base_llm, cache=cache_obj)
        else:
            self._llm = base_llm

        reranker_spec = cfg.reranker if self._reranker_kw is _UNSET else self._reranker_kw
        self._reranker = resolve_reranker(reranker_spec)

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
        cfg = self._config
        trace = QueryTrace()

        # --- Embedding ---
        t0 = time.perf_counter()
        query_vec = (await self._embedder.embed([question]))[0]
        trace.query_embedding_ms = int((time.perf_counter() - t0) * 1000)

        # --- Semantic cache lookup (skip for temporal/versioned queries) ---
        _temporal_active = bool(qc.as_of or qc.version)
        if self._semantic_cache is not None and not _temporal_active:
            cached_answer, sim = self._semantic_cache.lookup(query_vec, qc.cache_threshold)
            if cached_answer is not None:
                trace.cache_hit = True
                trace.cache_similarity = sim
                return Answer(
                    text=cached_answer.text,
                    citations=cached_answer.citations,
                    chunks_used=cached_answer.chunks_used,
                    sufficient=cached_answer.sufficient,
                    has_sufficient_context=cached_answer.has_sufficient_context,
                    trace=trace,
                )

        # --- Retrieval (with optional expansion) ---
        if (qc.n_queries > 1 or qc.query_variants) and not _temporal_active:
            t0 = time.perf_counter()
            results, expanded = await self._expand_and_search(question, qc, query_vec)
            trace.expanded_queries = expanded
            trace.expansion_ms = int((time.perf_counter() - t0) * 1000)
            trace.retrieval_ms = trace.expansion_ms
        else:
            t0 = time.perf_counter()
            results = await self._searcher.search(
                question,
                top_k=qc.top_k,
                alpha=qc.alpha,
                tenant_id=qc.tenant_id,
                allowed_sources=qc.allowed_sources if qc.allowed_sources else None,
                as_of=qc.as_of,
                version=qc.version,
            )
            trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)
            if _temporal_active:
                trace.temporal_filter_applied = True
                trace.as_of_used = qc.as_of

        # --- Reranking ---
        rerank_score_map: dict[str, float] = {}
        if self._reranker is not None and results:
            t0 = time.perf_counter()
            results = await self._reranker.rerank(question, results, top_k=qc.top_k)
            trace.rerank_ms = int((time.perf_counter() - t0) * 1000)
            rerank_score_map = {r.id: r.score for r in results}

        # --- Sufficiency / confidence gate ---
        has_sufficient_context = True
        if cfg.confidence_threshold > 0.0 and results:
            has_sufficient_context = await self._sufficiency.check(
                query_vec, results, cfg.confidence_threshold
            )

        # Also honour legacy per-query sufficiency check
        sufficient = True
        if qc.check_sufficiency and results:
            sufficient = await self._sufficiency.check(
                query_vec, results, qc.sufficiency_threshold
            )

        # Build RetrievedChunks for trace
        def _to_chunk(r: SearchResult) -> RetrievedChunk:
            rs = rerank_score_map.get(r.id)
            return RetrievedChunk(
                text=r.text,
                source=r.source,
                chunk_id=r.id,
                final_score=r.score,
                bm25_score=r.bm25_score,
                dense_score=r.dense_score,
                rerank_score=rs,
                page=r.metadata.get("page"),
            )

        # --- Context assembly ---
        assembler = (
            Assembler(max_context_tokens=qc.max_context_tokens)
            if qc.max_context_tokens
            else self._assembler
        )
        prompt, citations, dropped_results = assembler.assemble(question, results)

        trace.retrieved_chunks = [_to_chunk(r) for r in results]
        trace.dropped_chunks = [_to_chunk(r) for r in dropped_results]
        trace.context_tokens = count_tokens(prompt)

        # --- Confidence gate: skip LLM if insufficient ---
        if not has_sufficient_context:
            top3 = [
                Citation(
                    text=r.text,
                    source=r.source,
                    chunk_id=r.id,
                    final_score=r.score,
                    bm25_score=r.bm25_score,
                    dense_score=r.dense_score,
                    page=r.metadata.get("page"),
                )
                for r in results[:3]
            ]
            return Answer(
                text=cfg.insufficient_response,
                citations=[],
                chunks_used=len(results),
                sufficient=False,
                has_sufficient_context=False,
                trace=trace,
                top_retrieved=top3,
            )

        # --- Generation ---
        t0 = time.perf_counter()
        text = await self._llm.complete(prompt)
        trace.generation_ms = int((time.perf_counter() - t0) * 1000)

        # Estimate token counts + cost
        trace.prompt_tokens = count_tokens(prompt)
        trace.completion_tokens = count_tokens(text)
        llm_spec = cfg.llm if isinstance(cfg.llm, str) else ""
        trace.cost_usd = _estimate_cost(llm_spec, trace.prompt_tokens, trace.completion_tokens)

        answer = Answer(
            text=text,
            citations=citations if qc.include_citations else [],
            chunks_used=len(results),
            sufficient=sufficient,
            has_sufficient_context=True,
            trace=trace,
        )

        # Store in semantic cache (skip for temporal/versioned queries)
        if self._semantic_cache is not None and not _temporal_active:
            self._semantic_cache.store(query_vec, answer)

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
        cfg = self._config

        query_vec = (await self._embedder.embed([question]))[0]
        results = await self._searcher.search(
            question,
            top_k=qc.top_k,
            alpha=qc.alpha,
            tenant_id=qc.tenant_id,
            allowed_sources=qc.allowed_sources if qc.allowed_sources else None,
        )

        if self._reranker is not None and results:
            results = await self._reranker.rerank(question, results, top_k=qc.top_k)

        # Confidence gate
        if cfg.confidence_threshold > 0.0 and results:
            has_ctx = await self._sufficiency.check(query_vec, results, cfg.confidence_threshold)
            if not has_ctx:
                yield cfg.insufficient_response
                return

        assembler = (
            Assembler(max_context_tokens=qc.max_context_tokens)
            if qc.max_context_tokens
            else self._assembler
        )
        prompt, _, _ = assembler.assemble(question, results)

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
            as_of=qc.as_of,
            version=qc.version,
        )

    async def _expand_and_search(
        self,
        question: str,
        qc: QueryConfig,
        query_vec: list[float],
    ) -> tuple[list[SearchResult], list[str]]:
        """Generate query variants and merge results via RRF."""
        import asyncio
        import json as _json
        import logging

        _log = logging.getLogger(__name__)

        if qc.query_variants:
            variants = list(qc.query_variants)
        else:
            prompt = (
                f"Generate {qc.n_queries} search query variants for the following question. "
                f"Each variant should rephrase the question to improve document retrieval coverage. "
                f"Return as a JSON array of strings.\n\nOriginal: {question}"
            )
            try:
                assert self._llm is not None
                raw = await self._llm.complete(prompt)
                parsed = _json.loads(raw)
                variants = [str(v) for v in parsed[: qc.n_queries]]
            except Exception:
                _log.warning("[RAG] Query expansion parse failure — falling back to single query")
                variants = [question]

        assert self._searcher is not None
        searches = await asyncio.gather(
            *[
                self._searcher.search(
                    v,
                    top_k=qc.top_k,
                    alpha=qc.alpha,
                    tenant_id=qc.tenant_id,
                    allowed_sources=qc.allowed_sources if qc.allowed_sources else None,
                )
                for v in variants
            ],
            return_exceptions=True,
        )

        from ragwise.utils.rrf import rrf

        all_rankings: list[list[str]] = []
        lookup: dict[str, SearchResult] = {}
        for res in searches:
            if isinstance(res, Exception):
                continue
            all_rankings.append([r.id for r in res])  # type: ignore[union-attr]
            for r in res:  # type: ignore[union-attr]
                lookup[r.id] = r

        fused = rrf(all_rankings)
        results = [lookup[doc_id] for doc_id, _ in fused[: qc.top_k] if doc_id in lookup]
        return results, variants

    async def list_stale(
        self,
        as_of: str | None = "now",
        expiring_soon_days: int = 0,
    ) -> tuple[list[Any], list[Any]]:
        """Return (expired, expiring_soon) StaleSource lists.

        Sources with no valid_until metadata are never returned.
        """
        assert self._store is not None, "RAG must be used as a context manager"
        from ragwise.lifecycle import StalenessChecker
        checker = StalenessChecker(self._store)
        return await checker.list_stale(as_of=as_of, expiring_soon_days=expiring_soon_days)

    async def purge_stale(self, as_of: str | None = "now") -> Any:
        """Delete all chunks from sources whose valid_until < as_of."""
        assert self._store is not None, "RAG must be used as a context manager"
        from ragwise.lifecycle import StalenessChecker
        checker = StalenessChecker(self._store)
        return await checker.purge_stale(as_of=as_of)

    async def staleness_report(
        self,
        as_of: str | None = "now",
        expiring_soon_days: int = 30,
    ) -> str:
        """Return a human-readable staleness report string."""
        assert self._store is not None, "RAG must be used as a context manager"
        from ragwise.lifecycle import StalenessChecker
        checker = StalenessChecker(self._store)
        return await checker.staleness_report(as_of=as_of, expiring_soon_days=expiring_soon_days)

    @property
    def cache(self) -> Any:
        """Access the semantic cache for stats/clear/invalidate."""
        return self._semantic_cache

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
