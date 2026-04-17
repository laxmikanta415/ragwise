# ragwise

[![CI](https://github.com/laxmikanta415/ragwise/actions/workflows/ci.yml/badge.svg)](https://github.com/laxmikanta415/ragwise/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ragwise)](https://pypi.org/project/ragwise/)
[![Downloads](https://img.shields.io/pypi/dm/ragwise)](https://pypi.org/project/ragwise/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/ragwise)](https://pypi.org/project/ragwise/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://ragwise.readthedocs.io)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**The pip-installable Python RAG library built for 2026 — hybrid search, streaming, and agent tools on by default.**

No Docker. No background server. No framework lock-in. Just Python.

[Docs](https://ragwise.readthedocs.io) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Discussions](https://github.com/laxmikanta415/ragwise/discussions)

![ragwise demo](assets/demo-screenshot-1.png)

<video src="https://github.com/user-attachments/assets/96cd2ae9-e591-4c24-b3bf-8beb505446cc" controls width="100%"></video>

## Install

```bash
pip install ragwise
```

## Quickstart

```python
from ragwise import RAG

async with RAG(llm="openai/gpt-4o-mini") as rag:
    await rag.ingest("./docs/")
    answer = await rag.query("What is the refund policy?")
    print(answer.text)
    print(answer.citations)
```

![Hybrid search — BM25 + dense retrieval fused with RRF, answer with citations](assets/demo-screenshot-2.png)

## Why ragwise?

| Feature | ragwise | LangChain | LlamaIndex | RAGFlow |
|---------|--------|-----------|------------|---------|
| Lines to get started | **4** | 40+ | 20+ | Docker setup |
| Hybrid search on by default | **✓** | ✗ | opt-in | ✓ (Docker) |
| pip install (no server) | **✓** | ✓ | ✓ | ✗ |
| Async-first | **✓** | partial | partial | ✗ |
| Streaming responses | **✓** | partial | partial | ✗ |
| Agent tool built-in | **✓** | ✗ | ✗ | ✗ |
| Multi-tenant isolation | **✓** | ✗ | ✗ | ✗ |
| Incremental indexing | **✓** | ✗ | opt-in | ✗ |
| Built-in eval | **✓** | ✗ | partial | ✗ |

Chunking accuracy: RecursiveChunker matches FloTorch Feb 2026 benchmark winner at 69% end-to-end accuracy.

## Who It's For

**✓ Python developers** who want production-ready RAG as a library, not a platform.  
**✓ Teams already on PostgreSQL** — zero new infrastructure with `store="postgresql://..."`.  
**✓ Anyone who values typed, async-first, minimal-dependency code.**

**✗ Not for you if** you need a no-code UI, knowledge graphs, or agent orchestration — use RAGFlow or LangGraph instead.

## Streaming

```python
async for token in rag.stream_query("What changed in v2?"):
    print(token, end="", flush=True)
```

Works with OpenAI, Anthropic, and Ollama. Falls back gracefully for custom LLMs.

![Streaming — tokens arrive as they're generated, zero waiting](assets/demo-screenshot-3.png)

## Agent Tools

```python
from ragwise.agent import as_claude_tool

# Give your Claude agent access to your document index
tool = as_claude_tool(rag)              # Anthropic-compatible tool schema
results = await rag.search("query", top_k=5)  # raw retrieval, no generation
```

Pass `as_openai_tool(rag)` for OpenAI function calling.

![Agent tools — ready-made Claude and OpenAI tool schemas, raw search results](assets/demo-screenshot-4.png)

## Multi-Tenant Isolation

```python
# Tag documents by tenant at ingest time
await rag.ingest("./org_a_docs/", tenant_id="org_a")
await rag.ingest("./org_b_docs/", tenant_id="org_b")

# Filter by tenant at query time — no code changes to the store
answer = await rag.query("policy?", config=QueryConfig(tenant_id="org_a"))
```

![Multi-tenant isolation — full doc isolation with no store schema changes](assets/demo-screenshot-5.png)

## Store Options

```python
# Development — volatile, zero setup
RAG(store="memory")

# Development — persistent across restarts, embedded
RAG(store="lance://./ragwise-index")

# Production — PostgreSQL + pgvector, native hybrid search
RAG(store="postgresql://user:pass@localhost/mydb")
```

## Upgrade Path

```
store="memory"                  # local dev / tests
    ↓
store="lance://./ragwise-index"  # persistent dev, survives restarts
    ↓
store="postgresql://..."        # production (pgvector benchmark: 471 QPS @ 50M vectors)
```

## Configuration

```python
from ragwise import RAG, QueryConfig

async with RAG(
    embedder="openai/text-embedding-3-small",
    store="lance://./my-index",
    llm="openai/gpt-4o-mini",
    chunk_size=512,
    chunk_overlap=64,
    cache=True,
) as rag:
    await rag.ingest("./docs/", glob="**/*.md")
    answer = await rag.query(
        "What changed in v2?",
        config=QueryConfig(top_k=5, include_citations=True),
    )
```

## Optional Extras

```bash
pip install ragwise[lance]       # LanceDB persistent store
pip install ragwise[postgres]    # PostgreSQL + pgvector
pip install ragwise[local-emb]   # sentence-transformers embedder + reranker
pip install ragwise[eval]        # RAGAS + Langfuse eval loop
pip install ragwise[serve]       # ragwise serve HTTP API
```

## CLI

```bash
ragwise init          # generate ragwise_config.py with defaults
ragwise serve         # start HTTP API on localhost:8000
ragwise serve --port 9000
```

## Roadmap

v0.1.0 is the foundation. Follow [GitHub Discussions](https://github.com/laxmikanta415/ragwise/discussions) for what's coming next — roadmap is shaped by real usage and feedback.

## Community

- **Questions & help** → [GitHub Discussions](https://github.com/laxmikanta415/ragwise/discussions)
- **Bug reports** → [GitHub Issues](https://github.com/laxmikanta415/ragwise/issues)
- **Contributing** → [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT — see [LICENSE](LICENSE)
