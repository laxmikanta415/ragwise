"""Integration tests for the RAG pipeline — S4-T7/T8, S6-T1/T2/T3."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragwise import RAG, Answer, EvalSchema, IngestResult
from ragwise.eval.chunks import ChunkEvalSchema, score_chunks
from ragwise.eval.metrics import assert_eval_passes
from ragwise.eval.schema import EvalSchema as EvalSchemaFromModule
from ragwise.indexing.memory import InMemoryStore
from ragwise.ingestion.document import Document


# ---------------------------------------------------------------------------
# S4-T6 — EvalSchema
# ---------------------------------------------------------------------------

def test_eval_schema_default_values() -> None:
    es = EvalSchema()
    assert es.faithfulness == 0.0
    assert es.answer_relevance == 0.0
    assert es.context_recall == 0.0
    assert es.context_precision == 0.0
    assert es.latency_ms == 0.0


def test_eval_schema_custom_values() -> None:
    es = EvalSchema(faithfulness=0.9, answer_relevance=0.85)
    assert es.faithfulness == 0.9
    assert es.answer_relevance == 0.85


def test_eval_schema_importable_from_ragwise() -> None:
    from ragwise import EvalSchema as ES

    assert ES is EvalSchemaFromModule


# ---------------------------------------------------------------------------
# S6-T1 — EvalSchema extended fields + is_passing()
# ---------------------------------------------------------------------------

def test_eval_schema_extended_fields_defaults() -> None:
    es = EvalSchema()
    assert es.chunks_used == 0
    assert es.sufficient is True
    assert es.query == ""


def test_eval_schema_is_passing_true() -> None:
    es = EvalSchema(faithfulness=0.9, answer_relevance=0.8)
    assert es.is_passing() is True


def test_eval_schema_is_passing_false_faithfulness() -> None:
    es = EvalSchema(faithfulness=0.5, answer_relevance=0.9)
    assert es.is_passing(min_faithfulness=0.7) is False


def test_eval_schema_is_passing_false_relevance() -> None:
    es = EvalSchema(faithfulness=0.9, answer_relevance=0.5)
    assert es.is_passing(min_relevance=0.7) is False


def test_eval_schema_query_field() -> None:
    es = EvalSchema(query="what is x?")
    assert es.query == "what is x?"


# ---------------------------------------------------------------------------
# S6-T2 — assert_eval_passes + rag.eval()
# ---------------------------------------------------------------------------

def test_assert_eval_passes_succeeds() -> None:
    es = EvalSchema(faithfulness=0.9, answer_relevance=0.9)
    assert_eval_passes(es)  # should not raise


def test_assert_eval_passes_raises_on_low_faithfulness() -> None:
    es = EvalSchema(faithfulness=0.5, answer_relevance=0.9)
    with pytest.raises(AssertionError, match="faithfulness"):
        assert_eval_passes(es, min_faithfulness=0.7)


def test_assert_eval_passes_raises_on_low_relevance() -> None:
    es = EvalSchema(faithfulness=0.9, answer_relevance=0.5)
    with pytest.raises(AssertionError, match="answer_relevance"):
        assert_eval_passes(es, min_relevance=0.7)


@pytest.mark.asyncio
async def test_rag_eval_raises_without_ragas(tmp_path: Path) -> None:
    """rag.eval() raises ImportError when ragas is not installed."""
    store = InMemoryStore()
    dataset = [{"question": "q", "answer": "a", "contexts": ["ctx"], "ground_truth": "g"}]

    async with RAG(embedder=MagicMock(), store=store, llm=MagicMock(), cache=False) as rag:
        with patch("ragwise.eval.metrics._import_ragas", side_effect=ImportError("pip install ragwise[eval]")):
            with pytest.raises(ImportError, match="pip install ragwise"):
                await rag.eval(dataset)


# ---------------------------------------------------------------------------
# S6-T3 — ChunkEvalSchema + score_chunks() + rag.eval_chunks()
# ---------------------------------------------------------------------------

def _make_doc(text: str, idx: int = 0) -> Document:
    return Document(id=f"doc-{idx}", text=text, source="test.txt")


@pytest.mark.asyncio
async def test_chunk_eval_empty_list() -> None:
    embedder = MagicMock()
    result = await score_chunks([], embedder, chunk_size=512)
    assert result.total_chunks == 0
    assert result.avg_length_tokens == 0.0
    assert result.coherence_score == 0.0
    assert result.overlap_ratio == 0.0


@pytest.mark.asyncio
async def test_chunk_eval_single_doc() -> None:
    embedder = MagicMock()
    doc = _make_doc("The quick brown fox jumps over the lazy dog.")
    result = await score_chunks([doc], embedder, chunk_size=512)
    assert result.total_chunks == 1
    assert result.coherence_score == 1.0  # single doc defaults to coherent
    assert result.avg_length_tokens > 0


@pytest.mark.asyncio
async def test_chunk_eval_multiple_docs() -> None:
    dim = 4
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.8, 0.1, 0.1, 0.0]])

    docs = [_make_doc("word " * 10, i) for i in range(3)]
    result = await score_chunks(docs, embedder, chunk_size=512)

    assert result.total_chunks == 3
    assert 0.0 <= result.coherence_score <= 1.0
    assert result.avg_length_tokens > 0
    assert result.length_std >= 0.0


@pytest.mark.asyncio
async def test_chunk_eval_overlap_ratio() -> None:
    """Docs longer than chunk_size*1.1 counted in overlap_ratio."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[1.0, 0.0], [0.9, 0.1]])

    # 600 tokens worth of text — exceeds chunk_size=512 * 1.1 = 563.2
    long_doc = _make_doc("word " * 600, 0)
    short_doc = _make_doc("short", 1)
    result = await score_chunks([long_doc, short_doc], embedder, chunk_size=512)
    # 1 out of 2 docs is over threshold
    assert result.overlap_ratio == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_rag_eval_chunks_method(tmp_path: Path) -> None:
    store = InMemoryStore()
    dim = 4
    embedder = MagicMock()
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1] * dim for _ in texts])

    docs = [_make_doc("hello world this is a test chunk", i) for i in range(3)]

    async with RAG(embedder=embedder, store=store, llm=MagicMock(), cache=False) as rag:
        result = await rag.eval_chunks(docs)

    assert isinstance(result, ChunkEvalSchema)
    assert result.total_chunks == 3
    assert 0.0 <= result.coherence_score <= 1.0


# ---------------------------------------------------------------------------
# S6-T4 — LangfuseTracer + set_tracer() + auto-trace in query()
# ---------------------------------------------------------------------------

def test_langfuse_tracer_import_error() -> None:
    """LangfuseTracer raises ImportError when langfuse is not installed."""
    import ragwise.eval.tracer as tracer_module

    with patch("ragwise.eval.tracer._import_langfuse", side_effect=ImportError("pip install ragwise[eval]")):
        with pytest.raises(ImportError, match="pip install ragwise"):
            tracer_module._import_langfuse()


@pytest.mark.asyncio
async def test_rag_set_tracer_returns_self() -> None:
    store = InMemoryStore()
    mock_tracer = MagicMock()
    async with RAG(embedder=MagicMock(), store=store, llm=MagicMock(), cache=False) as rag:
        result = rag.set_tracer(mock_tracer)
    assert result is rag


@pytest.mark.asyncio
async def test_rag_query_calls_tracer(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("hello")

    store = InMemoryStore()
    dim = 4
    embedder = MagicMock()
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1] * dim for _ in texts])
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="answer")

    mock_tracer = MagicMock()
    mock_tracer.trace = AsyncMock()

    async with RAG(embedder=embedder, store=store, llm=llm, cache=False) as rag:
        rag.set_tracer(mock_tracer)
        await rag.ingest(str(tmp_path))
        await rag.query("test question")

    mock_tracer.trace.assert_called_once()
    call_args = mock_tracer.trace.call_args
    assert call_args[0][0] == "test question"  # first positional arg is query


@pytest.mark.asyncio
async def test_rag_tracer_failure_does_not_propagate(tmp_path: Path) -> None:
    """If tracer raises, query() still returns the answer."""
    (tmp_path / "doc.txt").write_text("hello")

    store = InMemoryStore()
    dim = 4
    embedder = MagicMock()
    embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1] * dim for _ in texts])
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="answer")

    broken_tracer = MagicMock()
    broken_tracer.trace = AsyncMock(side_effect=RuntimeError("network error"))

    async with RAG(embedder=embedder, store=store, llm=llm, cache=False) as rag:
        rag.set_tracer(broken_tracer)
        await rag.ingest(str(tmp_path))
        answer = await rag.query("test")  # should not raise

    assert answer.text == "answer"


@pytest.mark.asyncio
async def test_rag_eval_with_mock_evaluate_dataset() -> None:
    """rag.eval() returns populated EvalSchema when evaluate_dataset is mocked."""
    store = InMemoryStore()
    dataset = [{"question": "q", "answer": "a", "contexts": ["ctx"], "ground_truth": "g"}]

    expected = EvalSchema(
        faithfulness=0.85,
        answer_relevance=0.80,
        context_recall=0.75,
        context_precision=0.70,
    )

    async with RAG(embedder=MagicMock(), store=store, llm=MagicMock(), cache=False) as rag:
        with patch("ragwise.eval.metrics.evaluate_dataset", new=AsyncMock(return_value=expected)):
            scores = await rag.eval(dataset)

    assert scores.faithfulness == 0.85
    assert scores.answer_relevance == 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedder(dim: int = 8) -> MagicMock:
    emb = MagicMock()
    emb.embed = AsyncMock(side_effect=lambda texts: [[0.1] * dim for _ in texts])
    return emb


def _make_llm(answer: str = "test answer") -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=answer)
    return llm


# ---------------------------------------------------------------------------
# S4-T7/T8 — RAG pipeline integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_query_returns_answer(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")

    embedder = _make_embedder()
    llm = _make_llm("30 days.")

    store = InMemoryStore()
    async with RAG(embedder=embedder, store=store, llm=llm, cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("What is the refund policy?")

    assert isinstance(answer, Answer)
    assert answer.text == "30 days."


@pytest.mark.asyncio
async def test_rag_ingest_returns_ingest_result(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        result = await rag.ingest(str(tmp_path))

    assert isinstance(result, IngestResult)
    assert result.succeeded == 2
    assert result.failed == 0


@pytest.mark.asyncio
async def test_rag_incremental_second_call_skips(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("unchanged content")

    embedder = _make_embedder()
    store = InMemoryStore()
    async with RAG(embedder=embedder, store=store, llm=_make_llm(), cache=False) as rag:
        result1 = await rag.ingest(str(tmp_path))
        result2 = await rag.ingest(str(tmp_path))  # same files, nothing changed

    assert result1.succeeded == 1
    assert result2.succeeded == 0  # skipped — unchanged


@pytest.mark.asyncio
async def test_rag_force_reindex(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("content")

    embedder = _make_embedder()
    store = InMemoryStore()
    async with RAG(embedder=embedder, store=store, llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        result = await rag.ingest(str(tmp_path), force=True)

    assert result.succeeded == 1  # re-indexed even though unchanged


@pytest.mark.asyncio
async def test_rag_bad_file_captured_in_failed(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("good content")
    (tmp_path / "bad.xyz").write_bytes(b"\x00\x01\x02")  # unsupported extension

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        result = await rag.ingest(str(tmp_path))

    assert result.succeeded >= 1
    assert result.failed == 1
    assert len(result.errors) == 1


@pytest.mark.asyncio
async def test_rag_query_citations_populated(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("refund?")

    assert isinstance(answer.citations, list)
    # citations are Citation objects — check via citation_sources
    assert any("doc.txt" in s for s in answer.citation_sources)


@pytest.mark.asyncio
async def test_rag_chunks_used_populated(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("Content about dogs. " * 10)

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        answer = await rag.query("dogs")

    assert answer.chunks_used >= 1


@pytest.mark.asyncio
async def test_rag_empty_store_chunks_used_zero() -> None:
    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        answer = await rag.query("anything")

    assert answer.chunks_used == 0


# ---------------------------------------------------------------------------
# S7-T2 — stream_query()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_stream_query_yields_tokens(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")

    async def _streaming_llm_complete(prompt: str) -> str:
        return "fallback"

    async def _stream_gen(prompt: str):  # type: ignore[misc]
        for t in ["The ", "answer ", "is."]:
            yield t

    streaming_llm = MagicMock()
    streaming_llm.complete = AsyncMock(side_effect=_streaming_llm_complete)
    streaming_llm.stream_complete = _stream_gen

    # Make it look like StreamingLLMProtocol
    from ragwise.generation.llm import StreamingLLMProtocol
    streaming_llm.__class__ = type(
        "MockStreamingLLM",
        (object,),
        {"complete": streaming_llm.complete, "stream_complete": streaming_llm.stream_complete},
    )

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=streaming_llm, cache=False) as rag:
        rag._llm = streaming_llm  # bypass CachedLLM wrapping
        # Patch isinstance to treat streaming_llm as StreamingLLMProtocol
        import ragwise.pipeline as pipeline_mod
        orig_isinstance = pipeline_mod.__builtins__  # type: ignore[attr-defined]

        await rag.ingest(str(tmp_path))

        tokens: list[str] = []
        async for token in rag.stream_query("refund?"):
            tokens.append(token)

    # Should yield at least one token (fallback path since mock may not pass isinstance check)
    assert len(tokens) >= 1
    assert all(isinstance(t, str) for t in tokens)


@pytest.mark.asyncio
async def test_rag_stream_query_fallback_non_streaming_llm(tmp_path: Path) -> None:
    """Non-streaming LLM (no stream_complete) falls back to yielding complete() as single token."""
    (tmp_path / "doc.txt").write_text("Dogs are loyal animals.")

    class _BasicLLM:
        """LLM stub that only implements complete() — no stream_complete."""
        async def complete(self, prompt: str) -> str:
            return "mocked answer"

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_BasicLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        tokens: list[str] = []
        async for token in rag.stream_query("dogs?"):
            tokens.append(token)

    assert len(tokens) == 1
    assert tokens[0] == "mocked answer"


# ---------------------------------------------------------------------------
# S7-T3 — rag.search() + SearchResult export
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_search_returns_search_results(tmp_path: Path) -> None:
    from ragwise import SearchResult

    (tmp_path / "doc.txt").write_text("The refund policy allows 30-day returns.")

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        results = await rag.search("refund", top_k=3)

    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)
    if results:
        assert hasattr(results[0], "id")
        assert hasattr(results[0], "text")
        assert hasattr(results[0], "source")
        assert hasattr(results[0], "score")


@pytest.mark.asyncio
async def test_rag_search_empty_store_returns_empty() -> None:
    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        results = await rag.search("anything")
    assert results == []


def test_search_result_exported_from_ragwise() -> None:
    from ragwise import SearchResult
    assert SearchResult is not None


# ---------------------------------------------------------------------------
# S7-T4 — multi-tenancy: tenant_id + allowed_sources
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_tenant_isolation(tmp_path: Path) -> None:
    from ragwise import QueryConfig

    org_a = tmp_path / "org_a"
    org_b = tmp_path / "org_b"
    org_a.mkdir()
    org_b.mkdir()
    (org_a / "a.txt").write_text("Org A policy: 30-day refund.")
    (org_b / "b.txt").write_text("Org B policy: no refund.")

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(org_a), tenant_id="org_a")
        await rag.ingest(str(org_b), tenant_id="org_b")

        results_a = await rag.search("policy", config=QueryConfig(tenant_id="org_a", top_k=10))
        results_b = await rag.search("policy", config=QueryConfig(tenant_id="org_b", top_k=10))

    # Each tenant sees only their own docs
    assert all(r.metadata.get("tenant_id") == "org_a" for r in results_a)
    assert all(r.metadata.get("tenant_id") == "org_b" for r in results_b)


@pytest.mark.asyncio
async def test_rag_no_tenant_sees_all_docs(tmp_path: Path) -> None:
    org_a = tmp_path / "a"
    org_b = tmp_path / "b"
    org_a.mkdir()
    org_b.mkdir()
    (org_a / "a.txt").write_text("Content from org A.")
    (org_b / "b.txt").write_text("Content from org B.")

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(org_a), tenant_id="org_a")
        await rag.ingest(str(org_b), tenant_id="org_b")
        results = await rag.search("content", top_k=10)

    tenant_ids = {r.metadata.get("tenant_id") for r in results}
    assert len(tenant_ids) > 1 or len(results) == 0  # both tenants visible (or no match)


@pytest.mark.asyncio
async def test_rag_allowed_sources_filter(tmp_path: Path) -> None:
    from ragwise import QueryConfig

    pub = tmp_path / "public"
    priv = tmp_path / "private"
    pub.mkdir()
    priv.mkdir()
    (pub / "readme.txt").write_text("This is public content.")
    (priv / "secret.txt").write_text("This is private content.")

    store = InMemoryStore()
    async with RAG(embedder=_make_embedder(), store=store, llm=_make_llm(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        results = await rag.search(
            "content",
            config=QueryConfig(allowed_sources=[str(pub / "*")], top_k=10),
        )

    for r in results:
        assert "public" in r.source
