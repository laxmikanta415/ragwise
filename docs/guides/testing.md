# Testing Guide

`ragwise.testing` gives you deterministic, CI-friendly tests with zero API calls. The two key primitives are:

- **`FakeEmbedder`** — returns deterministic vectors without any embedding API
- **`cassette()`** — records RAG calls on first run and replays them in CI

Install the testing extras:

```bash
pip install ragwise[testing]
```

## FakeEmbedder

`FakeEmbedder(dim=384)` generates deterministic unit vectors from a hash of the input text. No API key, no network call, no randomness between runs.

```python
from ragwise import RAG
from ragwise.testing import FakeEmbedder

rag = RAG(
    embedder=FakeEmbedder(dim=384),
    llm="openai/gpt-4o-mini",
    store="memory",
)
```

Use it for unit tests where you care about control flow, not embedding quality.

## VCR Cassettes

`cassette()` is a context manager that records all RAG interactions (embedder calls, store queries, LLM calls) to a YAML file on first run, then replays them on subsequent runs — no API calls needed in CI.

```python
from ragwise.testing import cassette, assert_retrieval

async def test_refund_query():
    async with RAG(llm="openai/gpt-4o-mini") as rag:
        await rag.ingest("./docs/")

        with cassette("tests/cassettes/refund.yaml"):
            answer = await rag.query("What is the refund policy?")
            assert_retrieval(answer, must_include_source="docs/refund-policy.md")
            assert "5 business days" in answer.text
```

On first run: records all calls to `tests/cassettes/refund.yaml`.
On subsequent runs: replays from the YAML file — fast, deterministic, no API cost.

Commit cassette files to your repo so CI always has them.

## `assert_retrieval`

`assert_retrieval` fails with a clear error message when the expected source isn't in the retrieved chunks:

```python
assert_retrieval(
    answer,
    must_include_source="docs/refund-policy.md",
    # Optional: check multiple sources
    # must_include_all=["docs/returns.md", "docs/policy.md"],
)
```

Failure message example:
```
AssertionError: expected 'docs/refund-policy.md' in retrieved sources
Retrieved: ['docs/shipping.md', 'docs/contact.md', 'docs/faq.md']
Query: "What is the refund policy?"
Top chunk scores: [0.81, 0.74, 0.61]
```

## pytest Plugin

`pip install ragwise[testing]` auto-registers a pytest plugin that provides two fixtures:

```python
# conftest.py — nothing needed; fixtures are auto-registered

def test_with_fake_rag(fake_rag):
    # fake_rag: RAG instance with FakeEmbedder + memory store
    ...

async def test_with_recorded_rag(recorded_rag):
    # recorded_rag: RAG instance that records/replays via cassette
    ...
```

## Testing with Temporal Filtering

```python
from ragwise.testing import cassette, FakeEmbedder
from ragwise import RAG, QueryConfig

async def test_temporal_filter():
    rag = RAG(embedder=FakeEmbedder(dim=384), store="memory")
    await rag.ingest(
        "./docs/2024/",
        metadata={"valid_from": "2024-01-01", "valid_until": "2024-12-31"},
    )
    await rag.ingest(
        "./docs/2025/",
        metadata={"valid_from": "2025-01-01"},
    )

    answer_2024 = await rag.query("policy?", config=QueryConfig(as_of="2024-06-15"))
    answer_2025 = await rag.query("policy?", config=QueryConfig(as_of="2025-06-15"))

    # 2024 query should only return 2024 chunks
    sources_2024 = {c.source for c in answer_2024.citations}
    assert all("2024" in s for s in sources_2024)
```

## CI Setup

Example GitHub Actions step:

```yaml
- name: Test
  run: pytest tests/ -v
  env:
    # No API keys needed — cassettes cover all calls
    RAGWISE_STORE_BACKEND: memory
```

Cassette files are checked into the repo, so CI never makes real API calls.
