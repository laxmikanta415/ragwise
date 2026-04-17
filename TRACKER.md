# ragwise Development Tracker

**Last updated:** 2026-04-16
**Current sprint:** S6
**Overall progress:** 39/39 (100%)

---

## Sprint Board

| ID | Title | Status | Depends on |
|----|-------|--------|------------|
| S1-T1 | Project scaffold: pyproject.toml, src/ layout, CI skeleton | ✅ DONE | — |
| S1-T2 | `document.py`: Document dataclass + SourceHash utility | ✅ DONE | S1-T1 |
| S1-T3 | `config.py`: RAGConfig, QueryConfig, Answer, IngestResult | ✅ DONE | S1-T1 |
| S1-T4 | `utils/rrf.py`: Reciprocal Rank Fusion | ✅ DONE | S1-T1 |
| S1-T5 | `utils/tokens.py`: count_tokens(), truncate_to_tokens() | ✅ DONE | S1-T1 |
| S1-T6 | Sprint 1 integration check + seed tickets + TRACKER.md | ✅ DONE | S1-T1, S1-T2, S1-T3, S1-T4, S1-T5 |
| S2-T1 | `loader.py`: LoaderProtocol + AutoLoader (extension dispatch) | ✅ DONE | S1-T2 |
| S2-T2 | `loader.py`: TextLoader + MarkdownLoader (heading extraction) | ✅ DONE | S2-T1 |
| S2-T3 | `loader.py`: PDFLoader (pypdf, graceful per-file failure) | ✅ DONE | S2-T1 |
| S2-T4 | `loader.py`: HTMLLoader (beautifulsoup4) | ✅ DONE | S2-T1 |
| S2-T5 | `chunker.py`: ChunkerProtocol + RecursiveChunker (Chonkie, 512t/64 overlap) | ✅ DONE | S1-T2, S1-T5 |
| S2-T6 | `chunker.py`: StructureAwareChunker (heading-based splits) | ✅ DONE | S2-T5 |
| S3-T1 | `embedding/base.py`: EmbedderProtocol + resolve_embedder() | ✅ DONE | S1-T3 |
| S3-T2 | `embedding/openai.py`: OpenAIEmbedder (async, batched, dim auto-detect) | ✅ DONE | S3-T1 |
| S3-T3 | `embedding/local.py`: SentenceTransformerEmbedder (optional dep guard) | ✅ DONE | S3-T1 |
| S3-T4 | `indexing/base.py`: VectorStore ABC + EmbeddedDoc + SearchResult | ✅ DONE | S1-T2 |
| S3-T5 | `indexing/memory.py`: InMemoryStore (numpy dense + rank-bm25 sparse) | ✅ DONE | S3-T4, S1-T4 |
| S3-T6 | `indexing/lance.py`: LanceDBStore (embedded persistent, optional dep guard) | ✅ DONE | S3-T4 |
| S3-T7 | `indexing/pgvector.py`: PgVectorStore (psycopg async + pgvector + FTS) | ✅ DONE | S3-T4 |
| S3-T8 | `retrieval/search.py`: HybridSearcher (dense + sparse → RRF) | ✅ DONE | S3-T4, S1-T4 |
| S4-T1 | `retrieval/reranker.py`: CrossEncoderReranker (opt-in) | ✅ DONE | S3-T8 |
| S4-T2 | `retrieval/sufficiency.py`: SufficiencyChecker (cosine proxy) | ✅ DONE | S3-T8 |
| S4-T3 | `generation/llm.py`: LLMProtocol + resolve_llm() + OpenAI/Anthropic/Ollama | ✅ DONE | S1-T3 |
| S4-T4 | `generation/cache.py`: LLMCache (in-memory dict, Redis adapter) | ✅ DONE | S4-T3 |
| S4-T5 | `generation/prompts.py`: RAG_PROMPT template + Assembler | ✅ DONE | S1-T5, S1-T3 |
| S4-T6 | `eval/schema.py`: EvalSchema stub | ✅ DONE | S1-T1 |
| S4-T7 | `pipeline.py`: RAG class (context manager, ingest, query, incremental indexing) | ✅ DONE | S2-T6, S3-T8, S4-T3, S4-T4, S4-T5 |
| S4-T8 | `tests/test_pipeline.py`: Integration test with mocked embedder + LLM + InMemoryStore | ✅ DONE | S4-T7 |
| S5-T1 | `cli/main.py`: `ragwise init` command | ✅ DONE | S1-T3 |
| S5-T2 | `cli/serve.py`: `ragwise serve` (Starlette HTTP endpoint) | ✅ DONE | S4-T7 |
| S5-T3 | End-to-end integration tests (fixtures dir, InMemoryStore, mock OpenAI) | ✅ DONE | S4-T8 |
| S5-T4 | GitHub Actions CI (ruff + mypy + pytest + wheel build) | ✅ DONE | S5-T3 |
| S5-T5 | README.md: quickstart, benchmark table, who-it's-for, upgrade path, roadmap | ✅ DONE | S5-T3 |
| S6-T1 | `eval/schema.py`: Full EvalSchema (faithfulness, relevance, recall, precision, latency) | ✅ DONE | S4-T6 |
| S6-T2 | `eval/metrics.py`: `rag.eval()` RAGAS wrapper + `assert_eval_passes()` pytest helper | ✅ DONE | S6-T1 |
| S6-T3 | `eval/chunks.py`: `rag.eval_chunks()` — chunk quality scorer | ✅ DONE | S6-T1 |
| S6-T4 | `eval/tracer.py`: LangfuseTracer (same EvalSchema, opt-in) | ✅ DONE | S6-T1 |
| S6-T5 | `chunker.py`: ContextualChunker (Anthropic contextual retrieval approach) | ✅ DONE | S2-T5 |
| S6-T6 | `chunker.py`: HierarchicalChunker (small retrieve, large generate) | ✅ DONE | S2-T5, S3-T4 |

Status legend: ✅ DONE | 🔄 IN PROGRESS | ⏳ TODO | 🚫 BLOCKED

---

## Sprint Goals

| Sprint | Goal | Status | Done/Total |
|--------|------|--------|------------|
| S1 | Foundation: skeleton + shared types | ✅ DONE | 6/6 |
| S2 | Ingestion: loaders + chunkers | ✅ DONE | 6/6 |
| S3 | Embedding + indexing + hybrid retrieval | ✅ DONE | 8/8 |
| S4 | Generation + core pipeline | ✅ DONE | 8/8 |
| S5 | CLI + packaging + CI | ✅ DONE | 5/5 |
| S6 | Eval loop v0.2 | ✅ DONE | 6/6 |

---

## Blocked Tickets

None
