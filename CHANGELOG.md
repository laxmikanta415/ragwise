# Changelog

All notable changes to ragwise will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `ragx.agent.as_claude_tool()` — Anthropic-compatible tool schema for Claude agents
- `ragx.agent.as_openai_tool()` — OpenAI-compatible function tool schema
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
- `ragwise init` — generates `ragx_config.py` with typed defaults and inline comments
- `ragwise serve` — Starlette HTTP API on `localhost:8000` with `/health`, `/query`, `/ingest` (`pip install ragwise[serve]`)

[Unreleased]: https://github.com/laxmikanta415/ragx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/laxmikanta415/ragx/releases/tag/v0.1.0
