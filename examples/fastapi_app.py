"""Complete ragwise + FastAPI example — shows all major v0.2 features.

Run with:
    uvicorn examples.fastapi_app:app --reload

Endpoints:
    GET  /query?q=...         — one-shot RAG answer
    GET  /stream?q=...        — streaming SSE answer
    POST /ingest?path=...     — ingest a directory
    GET  /sources             — list indexed sources
    DELETE /source?path=...   — delete a source
"""
from __future__ import annotations

from functools import partial

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from ragwise import RAG, RAGConfig
from ragwise.fastapi import RAGLifespan, get_rag, stream_response

config = RAGConfig(
    llm="openai/gpt-4o-mini",
    embedder="openai/text-embedding-3-small",
    store="memory",
)

app = FastAPI(
    title="ragwise API",
    lifespan=partial(RAGLifespan, config=config),
)


@app.get("/query")
async def query(q: str, rag: RAG = Depends(get_rag)):
    answer = await rag.query(q)
    return {
        "answer": answer.text,
        "sources": answer.citation_sources,
        "chunks_used": answer.chunks_used,
        "has_context": answer.has_sufficient_context,
    }


@app.get("/stream")
async def stream(q: str, rag: RAG = Depends(get_rag)):
    return stream_response(rag.stream_query(q))


@app.post("/ingest")
async def ingest(path: str, rag: RAG = Depends(get_rag)):
    result = await rag.ingest(path)
    return {"succeeded": result.succeeded, "failed": result.failed, "errors": result.errors}


@app.get("/sources")
async def sources(rag: RAG = Depends(get_rag)):
    assert rag._store is not None
    return {"sources": await rag._store.list_sources()}


@app.delete("/source")
async def delete_source(path: str, rag: RAG = Depends(get_rag)):
    assert rag._store is not None
    await rag._store.delete(path)
    return JSONResponse({"deleted": path})
