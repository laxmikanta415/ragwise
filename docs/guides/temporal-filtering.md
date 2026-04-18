# Temporal Filtering

Temporal filtering lets you query your index as of a specific date — only returning chunks that were valid on that date. This is useful for policies, regulations, versioned documentation, and any corpus where documents expire.

No competitor has this feature built in.

## The Problem It Solves

Without temporal filtering:
- A refund policy updated in March still returns old January chunks if both are in the index
- Regulatory documents from previous years can surface alongside current ones
- "What was our policy on X?" is unanswerable — you'd need a separate index per version

With temporal filtering: tag documents at ingest with `valid_from` and `valid_until`, then query with `as_of` to get only the chunks that were valid on that date.

## Ingest with Validity Dates

```python
await rag.ingest(
    "./policies/2024/",
    metadata={
        "valid_from": "2024-01-01",
        "valid_until": "2024-12-31",
    },
)

await rag.ingest(
    "./policies/2025/",
    metadata={
        "valid_from": "2025-01-01",
        # no valid_until — currently valid
    },
)
```

Dates are stored as ISO 8601 strings (`YYYY-MM-DD`). Both `valid_from` and `valid_until` are optional — omitting either means "no bound on that side."

## Query with `as_of`

```python
from ragwise import QueryConfig

# Returns only chunks valid on 2024-06-15
answer = await rag.query(
    "What is the refund policy?",
    config=QueryConfig(as_of="2024-06-15"),
)
```

Chunks with `valid_until` before the `as_of` date are excluded. Chunks without `valid_from`/`valid_until` metadata are always included (opt-in semantics — no metadata = always visible).

## TTL: Automatic Expiry

Chunks with `valid_until` in the past are automatically excluded from all queries — you don't need to explicitly set `as_of`. This prevents stale policies from leaking into answers.

```python
# Ingest with an expiry date
await rag.ingest(
    "./offers/limited-time/",
    metadata={"valid_until": "2024-03-31"},
)

# After 2024-03-31, these chunks are excluded from every query
answer = await rag.query("What promotions are available?")
# No limited-time offer chunks in answer.citations after March 31st
```

## Staleness Report

Find source files that haven't been refreshed in N days:

```python
stale = await rag.list_stale(older_than_days=90)
for doc in stale:
    print(doc.source)        # "docs/old-policy.md"
    print(doc.last_updated)  # datetime
    print(doc.chunk_count)   # 12
```

## Backend Behavior

Temporal filtering is implemented consistently across all three backends:

| Backend | Mechanism |
|---|---|
| `memory` | Python list comprehension filtering on chunk metadata |
| `lance://` | LanceDB filter expression: `valid_from <= as_of AND (valid_until IS NULL OR valid_until > as_of)` |
| `postgresql://` | SQL WHERE clause on metadata JSONB columns with GIN index |

All three produce identical results. You can develop with `memory`, test with `lance://`, and deploy with `postgresql://` without changing any query code.

## Edge Cases

**No `as_of` set, document has `valid_until` in the past:** chunk is excluded — expired is expired.

**No `as_of` set, document has no dates:** chunk is always included.

**`as_of` set, document has no dates:** chunk is always included — temporal filter only applies to chunks with metadata.

**`valid_from` in the future:** chunk is excluded until that date is reached (useful for pre-staging future documents).
