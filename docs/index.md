# ragwise

**The pip-installable Python RAG library built for 2026 — hybrid search, retrieval observability, temporal filtering, and agent tools on by default.**

No Docker. No background server. No framework lock-in.

```bash
pip install ragwise
```

```python
from ragwise import RAG, QueryConfig

async with RAG(llm="openai/gpt-4o-mini", reranker="flashrank") as rag:
    result = await rag.ingest("./docs/")
    print(result)  # IngestResult(chunks_created=42, skipped=0, failed_files=[])

    answer = await rag.query("What is the refund policy?")
    print(answer.text)
    print(answer.citations[0].text)    # passage text, not just filename
    print(answer.trace.retrieval_ms)   # 34 — always populated
    print(answer.trace.cost_usd)       # 0.00021
```

Hybrid BM25+dense retrieval runs automatically on every query. `answer.trace` is always populated — no setup needed.

## Why ragwise?

| Feature | ragwise | LangChain | LlamaIndex | RAGFlow |
|---------|:----:|:---------:|:----------:|:-------:|
| Lines to get started | **4** | 40+ | 20+ | Docker setup |
| Hybrid search by default | ✅ | ❌ | opt-in | ✅ (Docker) |
| pip install, no server | ✅ | ✅ | ✅ | ❌ |
| Async-first | ✅ | partial | partial | ❌ |
| Streaming responses | ✅ | partial | partial | ❌ |
| Retrieval trace (always-on) | ✅ | ❌ | ❌ | ❌ |
| Passage-level citations | ✅ | ❌ | partial | ❌ |
| Temporal filtering (`as_of`) | ✅ | ❌ | ❌ | ❌ |
| Agent tool built-in | ✅ | ❌ | ❌ | ❌ |
| Multi-tenant isolation | ✅ | ❌ | ❌ | ❌ |
| Built-in eval | ✅ | ❌ | partial | ❌ |

Chunking accuracy: matches the FloTorch Feb 2026 benchmark winner at **69% end-to-end accuracy** using `RecursiveChunker`.

## Core Features

### Hybrid search — on by default
BM25+dense retrieval fused with Reciprocal Rank Fusion. Catches exact keywords *and* semantic meaning on every query, no configuration needed.

### Retrieval observability
```python
answer = await rag.query("...")
print(answer.trace.retrieval_ms)    # 34
print(answer.trace.cost_usd)        # 0.00021
print(answer.citations[0].text)     # passage text
print(answer.citations[0].score)    # 0.91
```

### Temporal filtering
```python
# Query as of any date — no competitor has this
answer = await rag.query(
    "refund policy?",
    config=QueryConfig(as_of="2024-06-15"),
)
```

### Semantic cache
```python
async with RAG(cache=True, cache_threshold=0.92, ...) as rag:
    a = await rag.query("How do refunds work?")
    print(a.trace.cache_hit)  # True on similar queries — <10ms
```

### Streaming
```python
async for token in rag.stream_query("What changed in v2?"):
    print(token, end="", flush=True)
```

### Agent tools (stateful)
```python
from ragwise.agent import as_claude_tool_suite

tools = as_claude_tool_suite(rag, max_iterations=5)
# search_documents + get_document_context + check_context_budget
```

### Multi-tenant isolation
```python
await rag.ingest("./org_a/", tenant_id="org_a")
await rag.ingest("./org_b/", tenant_id="org_b")
answer = await rag.query("policy?", config=QueryConfig(tenant_id="org_a"))
```

### Store upgrade path — no code changes
```
store="memory"                # dev / CI — zero setup, volatile
store="lance://./ragwise-index"  # persistent dev — no server
store="postgresql://..."      # production — pgvector
```

## Next Steps

- [Getting Started](getting-started.md) — install, first query, config
- [Stores](stores.md) — full store documentation
- [API Reference](api/rag.md) — `RAG`, `QueryConfig`, `Answer`
- [Configuration Reference](guides/configuration.md) — full `RAGConfig` + `QueryConfig` field table
- [Observability](guides/observability.md) — `answer.trace`, citations, `ragwise doctor`
- [Temporal Filtering](guides/temporal-filtering.md) — `as_of`, `valid_from/until`, TTL
- [Agent Tools](guides/agent-tools.md) — `AgentSession`, `as_claude_tool_suite`
- [Testing Guide](guides/testing.md) — VCR cassettes, `FakeEmbedder`, pytest plugin
- [FastAPI Integration](guides/fastapi.md) — `RAGLifespan`, `Depends(get_rag)`, streaming
- [Guides: Hybrid Search](guides/hybrid-search.md) — how BM25+dense+RRF works
- [Guides: Streaming](guides/streaming.md) — token streaming for UIs
- [Guides: Evaluation](guides/evaluation.md) — measure and gate retrieval quality
