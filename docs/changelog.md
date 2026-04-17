# Changelog

Full changelog is maintained in [CHANGELOG.md](https://github.com/laxmikanta415/ragwise/blob/main/CHANGELOG.md) in the repository root.

## [0.1.0] — 2026-04-16

### Added

- Core RAG pipeline: `async with RAG(...) as rag:` context manager
- Hybrid BM25+dense retrieval on every query, using Reciprocal Rank Fusion
- Three store backends: `InMemoryStore`, `LanceDBStore`, `PgVectorStore`
- LLM adapters: OpenAI, Anthropic, Ollama (local), and custom protocol
- Embedder adapters: OpenAI and local sentence-transformers
- Incremental indexing: `ingest()` skips unchanged files via SHA-256 content hash
- `ragwise init` CLI — generates typed config template
- `ragwise serve` CLI — minimal HTTP API (Starlette)
- Built-in eval: `rag.eval()`, `rag.eval_chunks()`, `assert_eval_passes()` pytest helper
- Production tracing via `LangfuseTracer`
- Advanced chunkers: `ContextualChunker` (LLM-enhanced), `HierarchicalChunker` (child retrieve, parent generate)
- LLM response caching (in-memory)
- `QueryConfig`: `top_k`, `include_citations`, `check_sufficiency`
- Cross-encoder reranker (opt-in)
- 170 tests — all passing
