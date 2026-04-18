# Agent Tools

ragwise exposes its document index as a ready-made tool for Claude and OpenAI agents. Your agent handles conversation and reasoning; ragwise handles retrieval.

## Overview

ragwise provides two patterns:

| API | Use when |
|---|---|
| `as_claude_tool(rag)` / `as_openai_tool(rag)` | Single-turn, stateless tool call |
| `as_claude_tool_suite(rag)` + `AgentSession` | Multi-turn, stateful — deduplication + loop detection |

## AgentSession (Multi-Turn)

`AgentSession` tracks retrieved chunks across calls, deduplicates by `chunk_id`, and warns when the same query is submitted twice (loop detection):

```python
from ragwise.agent import as_claude_tool_suite, AgentSession

async with RAG(llm="openai/gpt-4o-mini") as rag:
    await rag.ingest("./docs/")

    # Returns 3 tools: search_documents, get_document_context, check_context_budget
    tools = as_claude_tool_suite(rag, max_iterations=5)

    response = anthropic.messages.create(
        model="claude-opus-4-6",
        tools=tools,
        messages=[{"role": "user", "content": "Compare policy A to policy B"}],
    )
```

The three tools:
- **`search_documents`** — retrieves top-k chunks for a query
- **`get_document_context`** — fetches the full parent chunk for a given `chunk_id`
- **`check_context_budget`** — returns how many tokens of context remain before the session is full

### Multi-Hop Example

```python
# Agent makes 3 tool calls:
# 1. search_documents("maternity leave policy")   → retrieves policy A
# 2. search_documents("paternity leave policy")   → retrieves policy B
# 3. check_context_budget()                       → confirms budget before answer

# AgentSession deduplicates: if policy A appears in both searches, it's only
# passed to the LLM once. Loop detection warns if query 2 is too similar to query 1.
```

## Single-Turn Tools

## Claude Agent (Single-Turn)

```python
import asyncio
import anthropic
from ragwise import RAG
from ragwise.agent import as_claude_tool

async def main():
    async with RAG(llm="openai/gpt-4o-mini") as rag:
        await rag.ingest("./docs/")

        tool = as_claude_tool(rag)

        client = anthropic.Anthropic()

        # Agent turn 1 — model decides to use the tool
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=[tool],
            messages=[{"role": "user", "content": "What is the refund policy?"}],
        )

        # Handle tool call
        if response.stop_reason == "tool_use":
            tool_call = next(b for b in response.content if b.type == "tool_use")
            results = await rag.search(
                tool_call.input["query"],
                top_k=tool_call.input.get("top_k", 5),
            )

            # Format results as tool result
            tool_result_content = "\n\n".join(
                f"[{r.source}]\n{r.text}" for r in results
            )

            # Agent turn 2 — model generates answer from retrieved context
            final = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                tools=[tool],
                messages=[
                    {"role": "user", "content": "What is the refund policy?"},
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": tool_result_content,
                            }
                        ],
                    },
                ],
            )
            print(final.content[0].text)

asyncio.run(main())
```

## OpenAI Agent (Single-Turn)

```python
import asyncio
import json
from openai import AsyncOpenAI
from ragwise import RAG
from ragwise.agent import as_openai_tool

async def main():
    async with RAG(llm="openai/gpt-4o-mini") as rag:
        await rag.ingest("./docs/")

        tool = as_openai_tool(rag)
        client = AsyncOpenAI()

        messages = [{"role": "user", "content": "What is the refund policy?"}]

        # Agent turn 1
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            tools=[tool],
            messages=messages,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            results = await rag.search(args["query"], top_k=args.get("top_k", 5))

            context = "\n\n".join(f"[{r.source}]\n{r.text}" for r in results)

            messages += [
                msg,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": context,
                },
            ]

            # Agent turn 2
            final = await client.chat.completions.create(
                model="gpt-4o-mini",
                tools=[tool],
                messages=messages,
            )
            print(final.choices[0].message.content)

asyncio.run(main())
```

## Tool schema reference

### `as_claude_tool(rag)` output

```json
{
  "name": "search_documents",
  "description": "Search the document index and return the most relevant passages. Use this to retrieve context before answering questions.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query"
      },
      "top_k": {
        "type": "integer",
        "description": "Number of passages to return (default: 5)",
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

### `as_openai_tool(rag)` output

```json
{
  "type": "function",
  "function": {
    "name": "search_documents",
    "description": "Search the document index and return the most relevant passages.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 5}
      },
      "required": ["query"]
    }
  }
}
```

## Multi-tenant isolation in agents

Pass `QueryConfig` to `rag.search()` to restrict results to a specific tenant:

```python
from ragwise import QueryConfig

results = await rag.search(
    tool_call.input["query"],
    top_k=tool_call.input.get("top_k", 5),
    config=QueryConfig(tenant_id=current_user_org_id),
)
```

Results are filtered post-retrieval — the agent never sees documents from other tenants.

## `rag.search()` directly

`rag.search()` is the raw retrieval method. Call it directly when you want full control:

```python
from ragwise import RAG, SearchResult

async with RAG(...) as rag:
    results: list[SearchResult] = await rag.search("error code 0x80004005", top_k=5)

    for r in results:
        print(f"[{r.score:.3f}] {r.source}: {r.text[:120]}")
```

`SearchResult` fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique chunk identifier |
| `text` | `str` | Chunk text content |
| `source` | `str` | Source file path |
| `score` | `float` | RRF fusion score |
| `metadata` | `dict` | Arbitrary metadata (tenant_id, content_hash, etc.) |
