# ragwise

[![CI](https://github.com/laxmikanta415/ragwise/actions/workflows/ci.yml/badge.svg)](https://github.com/laxmikanta415/ragwise/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ragwise)](https://pypi.org/project/ragwise/)
[![Downloads](https://img.shields.io/pypi/dm/ragwise)](https://pypi.org/project/ragwise/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/ragwise)](https://pypi.org/project/ragwise/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-ragwise.readthedocs.io-blue)](https://ragwise.readthedocs.io)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**The retrieval layer your agents need — hybrid BM25+dense search, streaming, and agent tools on by default. pip install. No Docker.**

[Docs](https://ragwise.readthedocs.io) · [Changelog](CHANGELOG.md) · [PyPI](https://pypi.org/project/ragwise/) · [Discussions](https://github.com/laxmikanta415/ragwise/discussions)

![ragwise demo](assets/demo-screenshot-1.png)

<video src="https://github.com/user-attachments/assets/96cd2ae9-e591-4c24-b3bf-8beb505446cc" controls width="100%"></video>

---

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
    print(answer.citations)   # ["docs/refund-policy.md"]
```

![Hybrid search — BM25 + dense retrieval fused with RRF, answer with citations](assets/demo-screenshot-2.png)

---

## How it Works

A two-phase pipeline — ingest once, query with hybrid search every time. BM25 and dense retrieval run in parallel and are fused with RRF, scoring **18% higher NDCG** than dense-only.

![How ragwise works — ingest pipeline and hybrid query pipeline](assets/architecture.png)

| | Dense-only | BM25-only | Hybrid (ragwise) |
|---|---|---|---|
| NDCG score | 0.72 | 0.65 | **0.85** |

---

## Why ragwise?

| Feature | ragwise | LangChain | LlamaIndex | RAGFlow |
|---|---|---|---|---|
| Lines to get started | **4** | 40+ | 20+ | Docker setup |
| Hybrid search by default | ✅ | ❌ | opt-in | ✅ (Docker) |
| pip install, no server | ✅ | ✅ | ✅ | ❌ |
| Async-first | ✅ | partial | partial | ❌ |
| Streaming | ✅ | partial | partial | ❌ |
| Agent tool built-in | ✅ | ❌ | ❌ | ❌ |
| Multi-tenant isolation | ✅ | ❌ | ❌ | ❌ |
| Built-in eval | ✅ | ❌ | partial | ❌ |

---

## Agent Tools

Wire your entire document index into any Claude or OpenAI agent in one line. Your agent decides when to search — ragwise handles the retrieval.

```python
from ragwise.agent import as_claude_tool, as_openai_tool

tool = as_claude_tool(rag)    # Anthropic-compatible tool schema
# as_openai_tool(rag)         # OpenAI function calling

response = anthropic.messages.create(
    model="claude-opus-4-6",
    tools=[tool],
    messages=[{"role": "user", "content": question}],
)
```

![Agent tools — ready-made Claude and OpenAI tool schemas](assets/demo-screenshot-4.png)

---

## Streaming

Tokens stream as they're generated. Works with OpenAI, Anthropic, and Ollama — same two lines regardless of provider.

```python
async for token in rag.stream_query("What changed in v3.2?"):
    print(token, end="", flush=True)
```

![Streaming — tokens arrive as they're generated](assets/demo-screenshot-3.png)

---

## Multi-Tenant Isolation

Tag documents at ingest, filter at query time. No store schema changes needed — works with all three backends.

```python
await rag.ingest("./org_a_docs/", tenant_id="org_a")
await rag.ingest("./org_b_docs/", tenant_id="org_b")

answer = await rag.query(
    "What is our data retention policy?",
    config=QueryConfig(tenant_id="org_a"),
)
```

![Multi-tenant isolation — scoped retrieval per tenant](assets/demo-screenshot-5.png)

---

## Store Options

Same API from local dev to production. Change one string — nothing else.

```python
RAG(store="memory")                      # dev — zero setup, volatile
RAG(store="lance://./ragwise-index")     # dev — persistent, no server
RAG(store="postgresql://user:pw@db/x")  # production — pgvector
```

```
memory  →  lance://  →  postgresql://
  ↑            ↑               ↑
 tests      staging        production
```

---

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
        "What changed in v3.2?",
        config=QueryConfig(top_k=5, include_citations=True),
    )
```

---

## Optional Extras

```bash
pip install ragwise[lance]       # LanceDB persistent store
pip install ragwise[postgres]    # PostgreSQL + pgvector
pip install ragwise[local-emb]   # sentence-transformers embedder + reranker
pip install ragwise[eval]        # RAGAS + Langfuse eval loop
pip install ragwise[serve]       # ragwise serve HTTP API
```

---

## CLI

```bash
ragwise init           # generate ragwise_config.py with defaults
ragwise serve          # start HTTP API on localhost:8000
ragwise serve --port 9000
```

---

## Who It's For

**✓ Python developers** who want production-ready RAG as a library, not a platform.  
**✓ AI engineers building agents** — wire your doc index into Claude or GPT in one line.  
**✓ Teams already on PostgreSQL** — zero new infrastructure with `store="postgresql://..."`.  
**✓ Anyone who values typed, async-first, minimal-dependency code.**

**✗ Not for you if** you need a no-code UI, knowledge graphs, or agent orchestration — use RAGFlow or LangGraph instead.

---

## Roadmap

v0.1.0 is the foundation. The roadmap is shaped by real usage — follow [GitHub Discussions](https://github.com/laxmikanta415/ragwise/discussions) to vote on what ships next.

## Community

- **Questions & help** → [GitHub Discussions](https://github.com/laxmikanta415/ragwise/discussions)
- **Bug reports** → [GitHub Issues](https://github.com/laxmikanta415/ragwise/issues)
- **Contributing** → [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT — see [LICENSE](LICENSE)
