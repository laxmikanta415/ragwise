# Hybrid Search

ragwise runs hybrid BM25+dense retrieval on every query automatically. No configuration needed.

## Why hybrid?

Dense (embedding) search finds semantically similar text. BM25 (sparse) search finds exact keyword matches. Neither is best alone:

| Query type | Dense alone | BM25 alone | Hybrid |
|---|---|---|---|
| "show me the refund policy" | ✅ | ✅ | ✅ |
| "error code 0x80004005" | ❌ poor | ✅ exact | ✅ |
| "what does the SDK do conceptually" | ✅ | ❌ no keywords | ✅ |

Industry benchmarks consistently show hybrid outperforms either alone by 5–15% on NDCG@10 across mixed corpora.

## How RRF fusion works

ragwise uses **Reciprocal Rank Fusion (RRF)** to merge dense and sparse results into a single ranked list.

For each document `d` that appears in any result list:

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

where `k = 60` (standard constant) and `rank_i(d)` is the document's rank in result list `i`. Documents not in a list get a rank of `∞` (score contribution = 0).

RRF is rank-based, so it requires no score normalization between dense and sparse results — they use different scales (cosine similarity vs BM25 score), but ranks are universal.

### Example

| Doc | Dense rank | BM25 rank | RRF score |
|-----|-----------|-----------|-----------|
| A | 1 | 5 | 1/61 + 1/65 = 0.0321 |
| B | 3 | 1 | 1/63 + 1/61 = 0.0322 |
| C | 2 | 10 | 1/62 + 1/70 = 0.0304 |

Doc B wins despite not ranking first in either list — RRF rewards consistent presence across both signals.

## Per-store implementation

### InMemoryStore (`store="memory"`)
- Dense: numpy cosine similarity — `np.dot(query_vec, doc_vecs.T) / (norms * query_norm)`
- Sparse: `rank-bm25` library — standard BM25 Okapi implementation
- Both run in-process. No I/O.

### LanceDBStore (`store="lance://..."`)
- Dense: LanceDB native ANN index (IVF-PQ by default)
- Sparse: LanceDB full-text search (FTS) index built automatically on first `ingest()`
- Both queries run as embedded (no network, no server process)

### PgVectorStore (`store="postgresql://..."`)
- Dense: `pgvector` HNSW index — `embedding <=> query_vector ORDER BY ... LIMIT k`
- Sparse: PostgreSQL FTS — `tsvector` column + `ts_rank` scoring; `to_tsvector('english', content) @@ plainto_tsquery('english', query)`
- Both run in a single SQL query using `UNION ALL` + RRF in SQL: no round-trips

## Tuning

The default `top_k=5` returns 5 fused results. Increase for wider coverage:

```python
from ragwise import QueryConfig

answer = await rag.query(
    "What changed in v2?",
    config=QueryConfig(top_k=10),
)
```

To inspect raw retrieved chunks before generation:

```python
from ragwise.retrieval.search import HybridSearcher

searcher = HybridSearcher(store=rag._store, embedder=rag._embedder)
results = await searcher.search("error code 0x80004005", top_k=5)
for r in results:
    print(r.score, r.source, r.text[:100])
```

## Reranker (opt-in)

For higher precision, add a cross-encoder reranker after retrieval:

```python
RAG(
    llm="openai/gpt-4o-mini",
    reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",  # or "bge-reranker-v2-m3"
)
```

The reranker scores each retrieved chunk against the query using a model that jointly encodes both. Adds ~50–150ms per query but improves precision by ~8–12% on BEIR benchmarks.
