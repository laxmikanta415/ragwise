"""ragx serve — minimal Starlette HTTP API (requires pip install ragwise[serve])."""
from __future__ import annotations

try:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except ImportError as _e:
    raise ImportError("pip install ragwise[serve]") from _e


from ragwise import RAG, RAGConfig

_rag: RAG | None = None


def _load_config() -> RAGConfig:
    """Load config from ragx_config.py in cwd, or return defaults."""
    try:
        import importlib

        mod = importlib.import_module("ragx_config")
        config: RAGConfig = mod.config
        return config
    except (ImportError, AttributeError):
        return RAGConfig()


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def query_endpoint(request: Request) -> JSONResponse:
    global _rag
    if _rag is None:
        return JSONResponse({"error": "RAG not initialized"}, status_code=503)
    body = await request.json()
    question: str = body.get("question", "")
    from ragwise import QueryConfig

    qc = QueryConfig(top_k=body.get("top_k", 10))
    answer = await _rag.query(question, config=qc)
    return JSONResponse(
        {
            "answer": answer.text,
            "citations": answer.citations,
            "chunks_used": answer.chunks_used,
        }
    )


async def ingest_endpoint(request: Request) -> JSONResponse:
    global _rag
    if _rag is None:
        return JSONResponse({"error": "RAG not initialized"}, status_code=503)
    body = await request.json()
    path: str = body.get("path", ".")
    result = await _rag.ingest(path)
    return JSONResponse(
        {
            "succeeded": result.succeeded,
            "failed": result.failed,
            "errors": result.errors,
        }
    )


def create_app(config: RAGConfig | None = None) -> Starlette:
    cfg = config or _load_config()

    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/query", query_endpoint, methods=["POST"]),
            Route("/ingest", ingest_endpoint, methods=["POST"]),
        ]
    )

    @app.on_event("startup")
    async def startup() -> None:
        global _rag
        _rag = RAG(config=cfg)
        await _rag.__aenter__()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        global _rag
        if _rag is not None:
            await _rag.__aexit__(None, None, None)

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    app = create_app()
    uvicorn.run(app, host=host, port=port)
