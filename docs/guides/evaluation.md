# Evaluation

ragwise has a built-in eval loop. You can measure retrieval quality offline, gate it in CI, and trace production queries with the same schema.

## Quick eval

```python
from ragwise import RAG

test_dataset = [
    {
        "question": "What is the refund policy?",
        "ground_truth": "30-day full refund, no questions asked.",
        "answer": "You can get a full refund within 30 days.",
        "contexts": ["Our refund policy allows returns within 30 days."],
    },
]

async with RAG(llm="openai/gpt-4o-mini") as rag:
    await rag.ingest("./docs/")
    scores = await rag.eval(test_dataset)
    print(scores.faithfulness)       # 0.0–1.0
    print(scores.answer_relevance)   # 0.0–1.0
    print(scores.context_recall)     # 0.0–1.0
    print(scores.context_precision)  # 0.0–1.0
```

Requires: `pip install ragwise[eval]`

## EvalSchema fields

| Field | Description |
|---|---|
| `faithfulness` | Does the answer only use information from the retrieved context? |
| `answer_relevance` | Does the answer address the question? |
| `context_recall` | Does the retrieved context cover the ground-truth answer? |
| `context_precision` | What fraction of retrieved context is relevant (not noise)? |
| `latency_ms` | End-to-end query latency in milliseconds |
| `chunks_used` | Number of chunks retrieved |
| `sufficient` | Whether the sufficiency check passed |

## Use as a CI gate

`assert_eval_passes()` raises `AssertionError` if scores fall below thresholds — use it in pytest:

```python
# tests/test_rag_quality.py
import pytest
from ragwise import RAG
from ragwise.eval import assert_eval_passes

TEST_DATASET = [
    {
        "question": "What is the refund policy?",
        "ground_truth": "30-day full refund.",
        "answer": "30-day full refund, no questions asked.",
        "contexts": ["Our refund policy allows returns within 30 days."],
    },
]

@pytest.mark.asyncio
async def test_rag_quality():
    async with RAG(llm="openai/gpt-4o-mini") as rag:
        await rag.ingest("./docs/")
        scores = await rag.eval(TEST_DATASET)
        assert_eval_passes(scores, min_faithfulness=0.8, min_relevance=0.7)
```

Run in CI:
```bash
pytest tests/test_rag_quality.py -v
```

The test fails with a clear message if quality regresses:
```
AssertionError: Eval did not pass thresholds:
  faithfulness=0.62 (min=0.80)
  answer_relevance=0.71 (min=0.70)
```

## Chunk quality eval

Before running end-to-end eval, check whether your chunks are well-formed:

```python
from ragwise import RAG

async with RAG(store="memory") as rag:
    docs = await rag._loader.load("./docs/")
    chunk_scores = await rag.eval_chunks(docs)

    print(f"Avg chunk length: {chunk_scores.avg_length_tokens:.0f} tokens")
    print(f"Coherence score:  {chunk_scores.coherence_score:.3f}")  # 0=random, 1=perfect
    print(f"Overlap ratio:    {chunk_scores.overlap_ratio:.3f}")    # how much adjacent chunks share
    print(f"Total chunks:     {chunk_scores.total_chunks}")
```

A coherence score below 0.3 suggests chunks are splitting mid-sentence or mid-thought. Try increasing `chunk_size` or switching to `chunker="contextual"`.

## Production tracing with Langfuse

Connect ragwise to [Langfuse](https://langfuse.com) for production observability:

```python
from ragwise import RAG
from ragwise.eval import LangfuseTracer

tracer = LangfuseTracer(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
)

async with RAG(llm="openai/gpt-4o-mini").set_tracer(tracer) as rag:
    await rag.ingest("./docs/")
    answer = await rag.query("What is the refund policy?")
    # Every query is traced automatically
```

Each trace includes: query text, generated answer, citations, faithfulness/relevance scores, latency, and chunk count. Failures in the tracer are swallowed — they never break production queries.

Requires: `pip install ragwise[eval]` + Langfuse account (free tier available).

## Dataset format

`rag.eval()` accepts any list of dicts with these keys:

| Key | Required | Description |
|-----|----------|-------------|
| `question` | ✅ | The user query |
| `ground_truth` | ✅ | Expected answer (for recall/precision) |
| `answer` | ✅ | The generated answer to evaluate |
| `contexts` | ✅ | List of retrieved context strings |

Build your dataset by running `rag.query()` on a golden set and capturing the output:

```python
rows = []
for q, expected in golden_pairs:
    answer = await rag.query(q)
    rows.append({
        "question": q,
        "ground_truth": expected,
        "answer": answer.text,
        "contexts": [c.text for c in answer.citations],
    })

scores = await rag.eval(rows)
```
