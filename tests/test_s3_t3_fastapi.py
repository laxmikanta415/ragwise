"""S3-T3: FastAPI integration — RAGLifespan, get_rag, stream_response."""
from __future__ import annotations

import pytest

try:
    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")


def _make_app(tmp_path):
    from functools import partial

    from ragwise.fastapi import RAGLifespan, get_rag, stream_response
    from ragwise.testing import FakeEmbedder, FakeLLM

    app = FastAPI(
        lifespan=partial(
            RAGLifespan,
            embedder=FakeEmbedder(),
            llm=FakeLLM(response="test answer"),
            cache=False,
        )
    )

    @app.get("/query")
    async def query(q: str, rag=Depends(get_rag)):
        answer = await rag.query(q)
        return {"answer": answer.text}

    @app.get("/stream")
    async def stream(q: str, rag=Depends(get_rag)):
        return stream_response(rag.stream_query(q))

    @app.post("/ingest")
    async def ingest(path: str, rag=Depends(get_rag)):
        result = await rag.ingest(path)
        return {"succeeded": result.succeeded}

    @app.get("/sources")
    async def sources(rag=Depends(get_rag)):
        assert rag._store is not None
        return {"sources": await rag._store.list_sources()}

    return app


def test_raglifespan_initializes_rag(tmp_path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        assert hasattr(app.state, "rag")
        assert app.state.rag is not None


def test_get_rag_dependency(tmp_path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/query?q=hello")
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data


def test_ingest_and_sources(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("Refund policy: 30 days.")
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(f"/ingest?path={tmp_path}")
        assert r.status_code == 200
        assert r.json()["succeeded"] >= 1

        r2 = client.get("/sources")
        assert r2.status_code == 200


def test_stream_response_yields_events(tmp_path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        with client.stream("GET", "/stream?q=hello") as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            body = r.read().decode()

    assert "data: " in body


def test_stream_response_done_event(tmp_path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        with client.stream("GET", "/stream?q=hello") as r:
            body = r.read().decode()

    assert "data: [DONE]" in body
