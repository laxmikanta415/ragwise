# API Reference

## Authentication

All API requests require a Bearer token in the Authorization header:

```
Authorization: Bearer sk-your-api-key-here
```

API keys are generated in Dashboard → Settings → API Keys.

## Base URL

```
https://api.company.com/v2
```

## Endpoints

### POST /documents/ingest
Ingest one or more documents into your index.

**Request body:**
```json
{
  "files": ["./docs/"],
  "tenant_id": "org_123",
  "chunk_size": 512
}
```

**Response:**
```json
{
  "succeeded": 42,
  "failed": 0,
  "skipped": 18,
  "errors": []
}
```

### POST /documents/query
Query your document index and get a generated answer.

**Request body:**
```json
{
  "question": "What is the refund policy?",
  "top_k": 5,
  "include_citations": true,
  "tenant_id": "org_123"
}
```

**Response:**
```json
{
  "text": "We offer a 30-day money-back guarantee...",
  "citations": ["docs/refund-policy.md"],
  "latency_ms": 342,
  "chunks_used": 3
}
```

### POST /documents/search
Return raw search results without generation.

**Request body:**
```json
{
  "query": "error code 0x80004005",
  "top_k": 5
}
```

### GET /health
Returns service health status.

## Rate Limits

| Plan | Queries/min | Ingest/hour |
|------|-------------|-------------|
| Starter | 10 | 100 |
| Pro | 100 | 1,000 |
| Business | 500 | 10,000 |
| Enterprise | Custom | Custom |

## Error Codes

| Code | Meaning |
|------|---------|
| 401 | Invalid or expired API key |
| 429 | Rate limit exceeded |
| 413 | File too large (max 50MB) |
| 422 | Invalid request body |
| 500 | Internal server error — contact support |
