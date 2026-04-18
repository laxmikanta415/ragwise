# FastAPI Integration

`ragwise.fastapi` provides three building blocks for production HTTP APIs:

- **`RAGLifespan`** — manages RAG startup and shutdown as a FastAPI lifespan context
- **`get_rag`** — dependency for injecting RAG into endpoints
- **`stream_response`** — wraps `rag.stream_query()` as a `StreamingResponse`

```bash
pip install ragwise[serve]
```

## Minimal App

```python
from ragwise.fastapi import RAGLifespan, get_rag, stream_response
from fastapi import FastAPI, Depends

app = FastAPI(lifespan=RAGLifespan(llm="openai/gpt-4o-mini"))

@app.get("/query")
async def query(q: str, rag=Depends(get_rag)):
    answer = await rag.query(q)
    return {
        "text": answer.text,
        "citations": [{"source": c.source, "text": c.text} for c in answer.citations],
        "cost_usd": answer.trace.cost_usd,
    }

@app.get("/stream")
async def stream(q: str, rag=Depends(get_rag)):
    return stream_response(rag.stream_query(q))
```

## Multi-Tenant Endpoint

```python
from ragwise import QueryConfig

@app.get("/query/{org_id}")
async def query_org(org_id: str, q: str, rag=Depends(get_rag)):
    answer = await rag.query(q, config=QueryConfig(tenant_id=org_id))
    return {"text": answer.text}
```

## Ingest Endpoint

```python
from fastapi import UploadFile, File
import tempfile, os

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), rag=Depends(get_rag)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = await rag.ingest(tmp_path)
    os.unlink(tmp_path)
    return {"chunks_created": result.chunks_created, "failed": result.failed_files}
```

## RAGLifespan Configuration

`RAGLifespan` accepts any keyword argument that `RAG()` accepts:

```python
app = FastAPI(
    lifespan=RAGLifespan(
        llm="openai/gpt-4o-mini",
        store="postgresql://user:pw@db/myindex",
        embedder="openai/text-embedding-3-small",
        reranker="flashrank",
        cache=True,
        cache_threshold=0.92,
        confidence_threshold=0.7,
    )
)
```

## `stream_response` Content-Type

`stream_response()` returns a `StreamingResponse` with `media_type="text/event-stream"` — compatible with the EventSource browser API.

```javascript
// Browser
const es = new EventSource('/stream?q=refund+policy')
es.onmessage = (e) => console.log(e.data)
```

## Testing FastAPI Apps

```python
from fastapi.testclient import TestClient
from ragwise.testing import FakeEmbedder

def test_query_endpoint(monkeypatch):
    # Override embedder for tests
    monkeypatch.setenv("RAGWISE_EMBEDDER", "fake/384")
    client = TestClient(app)
    response = client.get("/query?q=refund+policy")
    assert response.status_code == 200
    assert "text" in response.json()
```
