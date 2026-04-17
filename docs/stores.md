# Store Options

ragwise supports three store backends that cover the full spectrum from local development to production.

## Comparison

| Store | Setup | Persistence | Scale | Use case |
|-------|-------|-------------|-------|----------|
| `memory` | zero | none (volatile) | ≤500K docs | tests, demos, CI |
| `lance://` | zero | disk | ≤50M docs | dev, prototyping |
| `postgresql://` | Postgres + pgvector | disk | 50M+ docs | production |

## InMemoryStore

```python
RAG(store="memory")
```

- Zero setup — no files, no server
- Data lost when the process exits
- Uses numpy for dense search and rank-bm25 for sparse search
- Recommended for: unit tests, CI, quick demos

## LanceDBStore

```python
pip install ragwise[lance]
```

```python
RAG(store="lance://./ragwise-index")
```

- Embedded — no server process
- Persistent across restarts
- Scales to tens of millions of documents
- Full-text search (FTS) index built automatically
- Recommended for: local development, prototyping, single-machine deployments

The path after `lance://` is relative to where your script runs.

## PgVectorStore

```python
pip install ragwise[postgres]
```

```python
RAG(store="postgresql://user:pass@localhost/mydb")
```

- Uses PostgreSQL with the `pgvector` extension
- Native hybrid search: `pgvector` for dense, `tsvector`/`ts_rank` for sparse — combined in one SQL query
- pgvector benchmark: 471 QPS at 50M vectors, 99% recall ([source](https://github.com/pgvector/pgvector#performance))
- No new infrastructure if your team already runs PostgreSQL
- Table `ragx_docs` and indexes are created automatically on first use
- Recommended for: production, multi-instance deployments, teams on PostgreSQL

### Quick setup with Docker (development only)

```bash
docker run -d \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  pgvector/pgvector:pg17

ragx_store = "postgresql://postgres:password@localhost/postgres"
```

## Upgrade path

```python
# Phase 1 — local dev
async with RAG(store="memory") as rag:
    ...

# Phase 2 — persistent dev, survives restarts
async with RAG(store="lance://./ragwise-index") as rag:
    ...

# Phase 3 — production
async with RAG(store="postgresql://user:pass@db.example.com/mydb") as rag:
    ...
```

The API stays identical across all three. Only the `store=` string changes.
