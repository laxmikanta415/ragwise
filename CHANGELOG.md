# Changelog

All notable changes to ragwise will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-04-18

### Added

**Typed configuration (S1-T1)**
- `RAGConfig` — Pydantic model; typos raise `ValueError` at construction, not at first query
- `RAGConfig.from_env()` — reads `RAGWISE_LLM_MODEL`, `RAGWISE_STORE_BACKEND`, `RAGWISE_EMBEDDER` env vars
- `RAGConfig.from_yaml()` — load from YAML config file
- `LLMConfig`, `StoreConfig`, `EmbedderConfig` — nested typed sub-configs

**Document management (S1-T2, S1-T3)**
- `rag.delete(source=...)` — removes all chunks for a source from all 3 backends
- `rag.list_sources()` — returns `list[SourceInfo]` with chunk counts and last-updated timestamps
- `rag.update(source=..., path=...)` — deletes stale chunks then re-ingests; no duplicates ever created
- Re-ingesting a changed file now automatically deletes all prior chunks before upserting new ones

**Ingestion progress and manifest (S1-T5)**
- `await rag.ingest(...)` now returns `IngestResult(chunks_created, skipped, failed_files)` with per-file error detail
- Progress callback API: `rag.ingest(..., on_progress=callback)` fires per file
- Rich progress bar in CLI when `ragwise serve` ingests

**Sufficiency checker (S1-T4)**
- `SufficiencyChecker` replaced — old check measured score consistency; new check measures centroid-distance coverage
- Optional LLM-based verification step for high-stakes queries

**Retrieval observability (S2-T1)**
- `answer.trace` — always populated; never requires opt-in
- `trace.retrieval_ms`, `trace.generation_ms`, `trace.cost_usd` — on every query
- `trace.retrieved_chunks[i].bm25_score`, `.dense_score`, `.rrf_score` — per-chunk component scores
- `trace.cache_hit` — `True` when answer was served from semantic cache
- `trace.query_variants` — list of queries generated when `n_queries > 1`

**Passage-level citations (S2-T2)**
- `answer.citations` is now `list[Citation]` — not `list[str]`
- `Citation(source, text, score, page, chunk_id, char_start)` — full passage text, not just filename
- `citation.explain()` — prints human-readable ranking explanation

**Confidence-gated answers (S2-T3)**
- `RAG(confidence_threshold=0.7)` — returns `answer.has_sufficient_context = False` when retrieval is too weak; LLM is not called
- Prevents hallucination from thin context without any extra code

**`ragwise doctor` CLI (S2-T4)**
- `ragwise doctor` — health check that runs in under 10 seconds
- Checks: credentials, store connectivity, embedding model version, hybrid search, round-trip latency
- Exits 0 on healthy; warns on version mismatch; prints checkmarks per component

**Rerankers (S2-T5)**
- `RAG(reranker="cohere/rerank-4")` — Cohere Rerank 4 cloud reranker (+33–48% accuracy)
- `RAG(reranker="flashrank")` — local reranker, no GPU required
- Both exposed through same `reranker=` string API

**AgentSession + tool suite (S3-T1)**
- `AgentSession` — stateful context across multiple tool calls; deduplicates chunks by `chunk_id`
- `as_claude_tool_suite(rag, max_iterations=5)` — returns 3 tools: `search_documents`, `get_document_context`, `check_context_budget`
- Loop detection: warns when the same (or very similar) query is submitted twice in a session

**Testing infrastructure (S3-T2)**
- `with cassette("tests/cassettes/q.yaml")` — VCR-style recording/replay; zero API calls in CI
- `FakeEmbedder(dim=384)` — deterministic vectors, no API needed
- `assert_retrieval(answer, must_include_source="docs/x.md")` — clear failure message when violated
- `pip install ragwise[testing]` auto-registers pytest fixtures: `fake_rag`, `recorded_rag`

**FastAPI integration (S3-T3)**
- `RAGLifespan` — FastAPI lifespan context manager; manages RAG startup/shutdown
- `Depends(get_rag)` — injects RAG instance into endpoints
- `stream_response(rag.stream_query(q))` — returns `StreamingResponse` with correct content-type

**Temporal metadata filtering (S4-T1)**
- `rag.ingest(..., metadata={"valid_from": "2024-01-01", "valid_until": "2024-12-31"})` — stores validity dates on chunks
- `QueryConfig(as_of="2024-06-15")` — only returns chunks valid on that date; works on all 3 backends
- Chunks with `valid_until` in the past are automatically excluded from all queries

**Semantic query cache (S4-T2)**
- `RAG(cache=True, cache_threshold=0.92)` — caches by query embedding similarity, not SHA-256 exact match
- Cache hit returns in under 10ms; `answer.trace.cache_hit` is always populated
- `RAGWISE_CACHE_REDIS_URL` env var enables Redis backend for cross-process sharing

**Query expansion — RAG-Fusion (S4-T3)**
- `QueryConfig(n_queries=3)` — generates 3 query variants, retrieves for each, fuses with RRF
- `answer.trace.query_variants` — shows the generated variants
- Works with all store backends

**Document TTL and staleness (S4-T4)**
- `rag.list_stale(older_than_days=90)` — returns sources not refreshed in N days
- Chunks with `valid_until` in the past are automatically excluded (no manual pruning needed)

### Changed

- **`answer.citations`** is now `list[Citation]` — was `list[str]` (filenames). Migration: `answer.citations[0].source` replaces `answer.citations[0]`.
- **`LLMCache`** now uses semantic (embedding-based) similarity by default when `cache=True` — was SHA-256 exact match.
- **`SufficiencyChecker`** now uses centroid-distance coverage — was mean-score consistency. More accurate for short and mixed-domain corpora.

### Migration

```python
# Before (v0.1.x)
print(answer.citations)        # ["docs/refund-policy.md"]

# After (v0.2.0)
print(answer.citations[0].source)  # "docs/refund-policy.md"
print(answer.citations[0].text)    # "Refunds are processed within..."
print(answer.citations[0].score)   # 0.91
```

## [0.1.0] - 2026-04-16

### Added

**Hybrid search — on by default**
- `HybridSearcher` — fuses BM25 sparse search with dense vector search via Reciprocal Rank Fusion (RRF, k=60)
- `InMemoryStore` — numpy cosine + rank-bm25, zero dependencies, for development and tests
- `LanceDBStore` — embedded persistent store, no server required (`pip install ragwise[lance]`)
- `PgVectorStore` — PostgreSQL + pgvector + native FTS hybrid search (`pip install ragwise[postgres]`)
- `CrossEncoderReranker` — optional re-ranking step (`pip install ragwise[local-emb]`)
- `SufficiencyChecker` — cosine-based check before calling the LLM

**Streaming**
- `rag.stream_query()` — async generator yielding tokens as they arrive; works with OpenAI, Anthropic, Ollama
- `StreamingLLMProtocol` — protocol for LLMs that support streaming; graceful fallback for custom LLMs

**Agent tools**
- `rag.search()` — public method returning raw `list[SearchResult]` for use as an agent tool
- `ragwise.agent.as_claude_tool()` — Anthropic-compatible tool schema for Claude agents
- `ragwise.agent.as_openai_tool()` — OpenAI-compatible function tool schema
- `SearchResult` exported from `ragwise` top-level

**Multi-tenant isolation**
- `ingest(tenant_id=...)` — tag documents with a tenant identifier at ingest time
- `QueryConfig(tenant_id=..., allowed_sources=[...])` — filter results by tenant or source glob at query time

**Core pipeline**
- `RAG` async context manager — the 4-line API: `async with RAG(llm=...) as rag:`
- `RAGConfig`, `QueryConfig`, `Answer`, `IngestResult`, `Document`, `EvalSchema` typed dataclasses
- Incremental indexing — `ingest()` hashes each file and skips unchanged documents on re-runs
- `force=True` flag on `ingest()` to bypass incremental check

**Ingestion**
- `AutoLoader` — dispatches by file extension (`.txt`, `.md`, `.pdf`, `.html`)
- `TextLoader`, `MarkdownLoader` (extracts headings), `PDFLoader` (pypdf), `HTMLLoader` (bs4)
- Graceful per-file failure — a bad PDF is captured in `IngestResult.errors`, never raises
- `RecursiveChunker` — Chonkie-backed, 512-token default, benchmark winner (69% E2E accuracy, FloTorch Feb 2026)
- `StructureAwareChunker` — splits at Markdown/HTML headings before recursive chunking
- `ContextualChunker` — prepends LLM-generated context summary per chunk, bounded concurrency
- `HierarchicalChunker` — small child chunks for retrieval, parent text stored for generation

**Generation**
- `LLMProtocol` + `resolve_llm()` — string shorthand: `"openai/gpt-4o-mini"`, `"anthropic/claude-haiku-4-5"`, `"ollama/llama3"`
- `OpenAILLM`, `AnthropicLLM`, `OllamaLLM` — all async, lazy client init
- `LLMCache` — in-memory cache with optional Redis backend; SHA-256 keyed

**Evaluation**
- `EvalSchema` — unified dataclass: faithfulness, answer_relevance, context_recall, context_precision, latency_ms
- `rag.eval(dataset)` — RAGAS wrapper (`pip install ragwise[eval]`)
- `assert_eval_passes(scores, ...)` — pytest helper for CI quality gates
- `LangfuseTracer` — production tracing via Langfuse (`pip install ragwise[eval]`)

**Embedding**
- `OpenAIEmbedder` — async, batched, lazy client init
- `SentenceTransformerEmbedder` — offline, thread-pool inference (`pip install ragwise[local-emb]`)
- `resolve_embedder()` — string shorthand: `"openai/text-embedding-3-small"`, `"local/all-MiniLM-L6-v2"`

**CLI**
- `ragwise init` — generates `ragwise_config.py` with typed defaults and inline comments
- `ragwise serve` — Starlette HTTP API on `localhost:8000` with `/health`, `/query`, `/ingest` (`pip install ragwise[serve]`)

[Unreleased]: https://github.com/laxmikanta415/ragwise/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/laxmikanta415/ragwise/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/laxmikanta415/ragwise/releases/tag/v0.1.0
