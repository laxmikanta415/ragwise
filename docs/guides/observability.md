# Observability

Every `rag.query()` call populates `answer.trace` — always, with no opt-in required. If retrieval is slow or answers are wrong, the trace tells you exactly where time was spent and which chunks were returned.

## Retrieval Trace

```python
answer = await rag.query("What is the refund policy?")

# Latency and cost
print(answer.trace.retrieval_ms)    # 34
print(answer.trace.generation_ms)   # 812
print(answer.trace.cost_usd)        # 0.00021

# Cache hit?
print(answer.trace.cache_hit)       # True / False
print(answer.trace.query_variants)  # queries generated with n_queries > 1
```

## Per-Chunk Scores

`answer.trace.retrieved_chunks` contains every chunk considered, with BM25, dense, and fused RRF scores:

```python
for chunk in answer.trace.retrieved_chunks:
    print(chunk.source)      # "docs/refund-policy.md"
    print(chunk.bm25_score)  # 0.71
    print(chunk.dense_score) # 0.88
    print(chunk.rrf_score)   # 0.93  (fused score used for ranking)
    print(chunk.text[:80])   # first 80 chars of the chunk
```

## Passage Citations

`answer.citations` contains the top-ranked chunks returned to the LLM, with full passage text:

```python
for c in answer.citations:
    print(c.source)    # "docs/refund-policy.md"
    print(c.text)      # "Refunds are processed within 5 business days..."
    print(c.score)     # 0.91
    print(c.page)      # 3
    print(c.chunk_id)  # "chunk_abc123"
    c.explain()        # prints human-readable ranking explanation
```

Prior to v0.2.0, `answer.citations` was `list[str]` (filenames). It is now `list[Citation]`. Migration: replace `answer.citations[0]` with `answer.citations[0].source`.

## Confidence Gating

When retrieval is too weak, avoid calling the LLM entirely:

```python
async with RAG(llm="openai/gpt-4o-mini", confidence_threshold=0.7) as rag:
    answer = await rag.query("...")
    if not answer.has_sufficient_context:
        print("Not enough evidence — answer withheld")
    else:
        print(answer.text)
```

The threshold is the minimum centroid-distance coverage score. If the retrieved chunks don't cover enough of the query's semantic space, `has_sufficient_context` is set to `False` and the LLM is not called.

## `ragwise doctor`

`ragwise doctor` runs a health check in under 10 seconds and prints a checkmark for each component:

```bash
ragwise doctor
```

Checks performed:
- Embedding model credentials and connectivity
- Store backend reachability (memory / LanceDB / pgvector)
- Hybrid search (BM25 + dense) round-trip latency
- Embedding model version consistency with existing index
- Reranker availability (if configured)

Exit codes: `0` = healthy, `1` = one or more checks failed.

## Trace Fields Reference

| Field | Type | Description |
|---|---|---|
| `retrieval_ms` | `float` | Time spent in BM25 + dense retrieval + reranking |
| `generation_ms` | `float` | Time spent in LLM generation |
| `cost_usd` | `float` | Estimated API cost (tokens × rate) |
| `cache_hit` | `bool` | `True` if answer was served from semantic cache |
| `query_variants` | `list[str]` | Generated query variants (RAG-Fusion) |
| `retrieved_chunks` | `list[RetrievedChunk]` | All chunks considered, with scores |

| `RetrievedChunk` field | Type | Description |
|---|---|---|
| `source` | `str` | Source file path |
| `text` | `str` | Chunk text |
| `bm25_score` | `float` | Raw BM25 score |
| `dense_score` | `float` | Cosine similarity score |
| `rrf_score` | `float` | Reciprocal Rank Fusion score (final ranking) |
| `chunk_id` | `str` | Stable chunk identifier |
