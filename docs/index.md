# ragwise

**The pip-installable Python RAG library built for 2026 — hybrid search, streaming, and agent tools on by default.**

No Docker. No background server. No framework lock-in.

```bash
pip install ragwise
```

```python
from ragwise import RAG

async with RAG(llm="openai/gpt-4o-mini") as rag:
    await rag.ingest("./docs/")
    answer = await rag.query("What is the refund policy?")
    print(answer.text)        # generated answer
    print(answer.citations)   # source files used
```

That's it. Hybrid BM25+dense retrieval runs automatically on every query.

## Why ragwise?

| Feature | ragwise | LangChain | LlamaIndex | RAGFlow |
|---------|:----:|:---------:|:----------:|:-------:|
| Lines to get started | **4** | 40+ | 20+ | Docker setup |
| Hybrid search by default | ✅ | ❌ | opt-in | ✅ (Docker) |
| pip install, no server | ✅ | ✅ | ✅ | ❌ |
| Async-first | ✅ | partial | partial | ❌ |
| Streaming responses | ✅ | partial | partial | ❌ |
| Agent tool built-in | ✅ | ❌ | ❌ | ❌ |
| Multi-tenant isolation | ✅ | ❌ | ❌ | ❌ |
| Incremental indexing | ✅ | ❌ | opt-in | ❌ |
| Built-in eval | ✅ | ❌ | partial | ❌ |

Chunking accuracy: matches the FloTorch Feb 2026 benchmark winner at **69% end-to-end accuracy** using `RecursiveChunker`.

## Core features

### Hybrid search — on by default
BM25+dense retrieval fused with Reciprocal Rank Fusion. Catches exact keywords *and* semantic meaning on every query, no configuration needed.

### Streaming
```python
async for token in rag.stream_query("What changed in v2?"):
    print(token, end="", flush=True)
```

Works with OpenAI, Anthropic, and Ollama out of the box.

### Agent tools
```python
from ragwise.agent import as_claude_tool

tool = as_claude_tool(rag)              # Anthropic-compatible tool schema
results = await rag.search("query")     # raw retrieval, no generation
```

Pass `as_openai_tool(rag)` for OpenAI function calling. ragwise becomes the retrieval brain for any agent.

### Multi-tenant isolation
```python
await rag.ingest("./org_a/", tenant_id="org_a")
await rag.ingest("./org_b/", tenant_id="org_b")

# org_b docs never appear in org_a queries
answer = await rag.query("policy?", config=QueryConfig(tenant_id="org_a"))
```

### Store upgrade path — no code changes
```
store="memory"                # dev / CI — zero setup, volatile
store="lance://./ragwise-index"  # persistent dev — no server, survives restarts
store="postgresql://..."      # production — pgvector benchmark: 471 QPS @ 50M vectors
```

The API stays identical across all three. Only the `store=` string changes.

## Next steps

- [Getting Started](getting-started.md) — install, first query, config
- [Stores](stores.md) — full store documentation
- [API Reference](api/rag.md) — `RAG`, `QueryConfig`, `Answer`
- [Guides: Hybrid Search](guides/hybrid-search.md) — how BM25+dense+RRF works
- [Guides: Streaming](guides/streaming.md) — token streaming for UIs
- [Guides: Agent Tools](guides/agent-tools.md) — use ragwise inside Claude/GPT agents
- [Guides: Evaluation](guides/evaluation.md) — measure and gate retrieval quality
