"""ragwise FastAPI integration — RAGLifespan, get_rag, stream_response."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
except ImportError as e:
    raise ImportError("pip install ragwise[serve]") from e

from ragwise import RAG, RAGConfig


@asynccontextmanager
async def RAGLifespan(app: FastAPI, **rag_kwargs: Any) -> AsyncGenerator[None, None]:
    """FastAPI lifespan that initialises a RAG instance and stores it on app.state.

    Example::

        from ragwise.fastapi import RAGLifespan
        from functools import partial

        app = FastAPI(lifespan=partial(RAGLifespan, config=RAGConfig(llm="openai/gpt-4o-mini")))

    The RAG instance is available via ``app.state.rag`` or via ``Depends(get_rag)``.
    """
    config: RAGConfig | None = rag_kwargs.pop("config", None)
    rag = RAG(config=config, **rag_kwargs)
    app.state.rag = await rag.__aenter__()
    try:
        yield
    finally:
        await rag.__aexit__(None, None, None)


def get_rag(request: Request) -> RAG:
    """FastAPI dependency — injects the shared RAG instance.

    Example::

        @app.get("/query")
        async def query(q: str, rag: RAG = Depends(get_rag)):
            answer = await rag.query(q)
            return {"answer": answer.text}
    """
    return request.app.state.rag  # type: ignore[no-any-return]


def stream_response(stream: AsyncGenerator[str, None]) -> StreamingResponse:
    """Wrap a ``rag.stream_query()`` async generator as a Server-Sent Events response.

    Example::

        @app.get("/stream")
        async def stream_endpoint(q: str, rag: RAG = Depends(get_rag)):
            return stream_response(rag.stream_query(q))
    """

    async def _generate() -> AsyncGenerator[str, None]:
        try:
            async for token in stream:
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(_generate(), media_type="text/event-stream")
