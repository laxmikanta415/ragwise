# Streaming

ragwise supports token streaming out of the box for OpenAI, Anthropic, and Ollama. Tokens are yielded as they arrive — no waiting for the full response.

## Basic usage

```python
import asyncio
from ragwise import RAG

async def main():
    async with RAG(llm="openai/gpt-4o-mini") as rag:
        await rag.ingest("./docs/")

        async for token in rag.stream_query("What changed in v2?"):
            print(token, end="", flush=True)

        print()  # newline after stream ends

asyncio.run(main())
```

`stream_query()` uses the same retrieval path as `query()` — hybrid BM25+dense search, sufficiency check, context assembly — and then streams the LLM response token by token.

## LLM support

| Provider | Streaming support |
|----------|:-----------------:|
| OpenAI (`openai/gpt-4o-mini`, etc.) | ✅ |
| Anthropic (`anthropic/claude-*`) | ✅ |
| Ollama (`ollama/llama3`, etc.) | ✅ |
| Custom LLM (implements `.complete()` only) | Fallback — yields full response as one token |

If your custom LLM doesn't implement `StreamingLLMProtocol`, `stream_query()` falls back gracefully: it calls `complete()` and yields the full string as a single token. No errors, no exceptions.

## Streaming in a FastAPI endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ragwise import RAG

app = FastAPI()
rag: RAG | None = None

@app.on_event("startup")
async def startup():
    global rag
    rag = RAG(llm="openai/gpt-4o-mini")
    await rag.__aenter__()
    await rag.ingest("./docs/")

@app.get("/stream")
async def stream(q: str):
    async def token_generator():
        async for token in rag.stream_query(q):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")
```

This produces a standard Server-Sent Events (SSE) stream that any frontend can consume.

## Streaming with config

All `QueryConfig` options apply to `stream_query()` exactly as they do to `query()`:

```python
from ragwise import RAG, QueryConfig

async with RAG(llm="openai/gpt-4o-mini", store="lance://./ragwise-index") as rag:
    config = QueryConfig(
        top_k=8,
        tenant_id="org_a",               # multi-tenant isolation
        include_citations=True,
    )

    async for token in rag.stream_query("What is the cancellation policy?", config=config):
        print(token, end="", flush=True)
```

## Difference from `query()`

| | `query()` | `stream_query()` |
|--|-----------|------------------|
| Return type | `Answer` (text + citations) | `AsyncGenerator[str, None]` (tokens) |
| Waits for full response | Yes | No — yields immediately |
| Citations | `answer.citations` | Not included — use `rag.search()` for sources |
| Caching | Hits LLM cache | Bypasses cache |

If you need citations alongside streaming, call `rag.search()` first for the sources, then stream the generation:

```python
results = await rag.search("What is the cancellation policy?", top_k=5)
sources = [r.source for r in results]

async for token in rag.stream_query("What is the cancellation policy?"):
    print(token, end="", flush=True)

print("\nSources:", sources)
```

## Implementing StreamingLLMProtocol

To add streaming to a custom LLM:

```python
from collections.abc import AsyncGenerator
from ragwise.generation.llm import StreamingLLMProtocol

class MyStreamingLLM:
    async def complete(self, prompt: str) -> str:
        # non-streaming fallback
        return "..."

    async def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        # yield tokens as they arrive from your LLM backend
        async for token in my_backend.stream(prompt):
            yield token

# ragwise detects StreamingLLMProtocol at runtime
async with RAG(llm=MyStreamingLLM()) as rag:
    async for token in rag.stream_query("..."):
        print(token, end="", flush=True)
```

ragwise uses `isinstance(llm, StreamingLLMProtocol)` at runtime to decide whether to stream or fall back. The protocol check is structural — no inheritance required.
