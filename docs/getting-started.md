# Getting Started

## Installation

```bash
pip install ragwise
```

For additional backends:

```bash
pip install ragwise[lance]       # LanceDB persistent store
pip install ragwise[postgres]    # PostgreSQL + pgvector
pip install ragwise[local-emb]   # offline sentence-transformers embedder
pip install ragwise[eval]        # RAGAS evaluation + Langfuse tracing
pip install ragwise[serve]       # HTTP API server
```

## Prerequisites

- Python 3.11 or higher
- An OpenAI API key (for the default embedder and LLM) — or use a local model

```bash
export OPENAI_API_KEY="sk-..."
```

## Your first RAG pipeline

```python
import asyncio
from ragwise import RAG

async def main():
    async with RAG(llm="openai/gpt-4o-mini") as rag:
        # Ingest a directory of documents
        result = await rag.ingest("./docs/")
        print(f"Indexed {result.succeeded} files, {result.failed} failed")

        # Query with hybrid search
        answer = await rag.query("What is the refund policy?")
        print(answer.text)
        print("Sources:", answer.citations)

asyncio.run(main())
```

## Streaming responses

Stream tokens as they arrive — no waiting for the full response:

```python
async with RAG(llm="openai/gpt-4o-mini") as rag:
    await rag.ingest("./docs/")
    async for token in rag.stream_query("What changed in v2?"):
        print(token, end="", flush=True)
```

Works with OpenAI, Anthropic, and Ollama. Falls back gracefully for custom LLMs.

## Using ragwise as an agent tool

Give your Claude or OpenAI agent access to your document index:

```python
import anthropic
from ragwise import RAG
from ragwise.agent import as_claude_tool

async with RAG(llm="openai/gpt-4o-mini") as rag:
    await rag.ingest("./docs/")

    # Get the tool definition
    tool = as_claude_tool(rag)

    # Use it in your agent loop
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        tools=[tool],
        messages=[{"role": "user", "content": "What is the refund policy?"}],
    )

    # When the agent calls search_documents, run rag.search()
    if response.stop_reason == "tool_use":
        tool_call = next(b for b in response.content if b.type == "tool_use")
        results = await rag.search(tool_call.input["query"], top_k=tool_call.input.get("top_k", 5))
```

Use `as_openai_tool(rag)` for OpenAI function calling.

## Switching LLMs

```python
# Anthropic
RAG(llm="anthropic/claude-haiku-4-5")

# Ollama (local, no API key)
RAG(llm="ollama/llama3")

# Pass any object that implements .complete(prompt) -> str
RAG(llm=my_custom_llm)
```

## Switching embedders

```python
# Default (OpenAI)
RAG(embedder="openai/text-embedding-3-small")

# Local — no API key, no network calls
RAG(embedder="local/all-MiniLM-L6-v2")  # pip install ragwise[local-emb]
```

## Configuration

```python
from ragwise import RAG, QueryConfig

async with RAG(
    embedder="openai/text-embedding-3-small",
    store="lance://./my-index",      # persistent, no server
    llm="openai/gpt-4o-mini",
    chunk_size=512,
    chunk_overlap=64,
    cache=True,                      # LLM response caching
) as rag:
    await rag.ingest("./docs/", glob="**/*.md")

    answer = await rag.query(
        "What changed in v2?",
        config=QueryConfig(
            top_k=5,
            include_citations=True,
            check_sufficiency=True,
        ),
    )
```

## Multi-tenant isolation

Tag documents by tenant at ingest, filter at query time:

```python
async with RAG(store="lance://./index") as rag:
    # Ingest documents per tenant
    await rag.ingest("./org_a_docs/", tenant_id="org_a")
    await rag.ingest("./org_b_docs/", tenant_id="org_b")

    # Query is isolated — org_b docs never appear
    answer = await rag.query(
        "What is our policy?",
        config=QueryConfig(tenant_id="org_a"),
    )

    # Filter by source glob pattern
    results = await rag.search(
        "refund",
        config=QueryConfig(allowed_sources=["docs/public/*"]),
    )
```

## Incremental indexing

`ingest()` tracks a SHA-256 hash of each file. Re-running on the same directory skips unchanged files automatically:

```python
result1 = await rag.ingest("./docs/")   # indexes all files
result2 = await rag.ingest("./docs/")   # skips unchanged files (result2.succeeded == 0)

# Force full re-index
result3 = await rag.ingest("./docs/", force=True)
```

## Generate a config file

```bash
ragwise init
```

Creates `ragwise_config.py` in the current directory with all defaults and inline comments — a good starting point for any project.

## Next steps

- [Stores](../stores.md) — choose between memory, LanceDB, and PostgreSQL
- [Guides: Hybrid Search](../guides/hybrid-search.md) — understand BM25+dense fusion
- [Guides: Streaming](../guides/streaming.md) — token streaming for UIs
- [Guides: Agent Tools](../guides/agent-tools.md) — ragwise inside Claude/GPT agents
- [Guides: Evaluation](../guides/evaluation.md) — measure retrieval quality
