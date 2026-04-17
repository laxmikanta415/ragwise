"""S3-T2: ragwise.testing — FakeEmbedder, FakeLLM, cassette, assert_retrieval."""
from __future__ import annotations

import pytest

from ragwise import RAG, Answer, Citation
from ragwise.testing import (
    FakeEmbedder,
    FakeLLM,
    GoldenEntry,
    RAGCassette,
    assert_retrieval,
    cassette,
    generate_golden_dataset,
    load_golden_dataset,
)


# ---------------------------------------------------------------------------
# FakeEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_embedder_deterministic() -> None:
    emb = FakeEmbedder(dim=64)
    v1 = await emb.embed(["hello world"])
    v2 = await emb.embed(["hello world"])
    assert v1 == v2


@pytest.mark.asyncio
async def test_fake_embedder_different_texts() -> None:
    emb = FakeEmbedder(dim=64)
    v1 = await emb.embed(["hello world"])
    v2 = await emb.embed(["completely different text about taxes"])
    assert v1 != v2


@pytest.mark.asyncio
async def test_fake_embedder_unit_normalized() -> None:
    import numpy as np
    emb = FakeEmbedder(dim=128)
    vecs = await emb.embed(["test"])
    norm = float(np.linalg.norm(vecs[0]))
    assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_llm_default_response() -> None:
    llm = FakeLLM()
    result = await llm.complete("What is the policy?")
    assert "Fake answer for:" in result


@pytest.mark.asyncio
async def test_fake_llm_custom_response() -> None:
    llm = FakeLLM(response="Custom response text")
    result = await llm.complete("anything")
    assert result == "Custom response text"


@pytest.mark.asyncio
async def test_fake_llm_stream() -> None:
    llm = FakeLLM(response="abc")
    tokens = []
    async for t in llm.stream("prompt"):
        tokens.append(t)
    assert "".join(tokens) == "abc"


# ---------------------------------------------------------------------------
# Cassette
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cassette_record_then_replay(tmp_path) -> None:
    cass_path = tmp_path / "test.yaml"
    fake_emb = FakeEmbedder(dim=32)
    fake_llm = FakeLLM(response="Policy answer")

    # Record
    with cassette(cass_path, mode="record") as cas:
        emb = cas.wrap_embedder(fake_emb)
        llm = cas.wrap_llm(fake_llm)
        r1 = await emb.embed(["refund policy"])
        a1 = await llm.complete("What is the refund policy?")

    assert cass_path.exists()

    # Replay — even if inner objects raise, cassette returns recorded values
    class BrokenEmbedder:
        async def embed(self, texts):
            raise RuntimeError("should not be called in replay")

    class BrokenLLM:
        async def complete(self, prompt):
            raise RuntimeError("should not be called in replay")

    with cassette(cass_path, mode="replay") as cas:
        emb2 = cas.wrap_embedder(BrokenEmbedder())
        llm2 = cas.wrap_llm(BrokenLLM())
        r2 = await emb2.embed(["refund policy"])
        a2 = await llm2.complete("What is the refund policy?")

    assert r1 == r2
    assert a1 == a2


def test_cassette_yaml_is_readable(tmp_path) -> None:
    import yaml

    cass_path = tmp_path / "readable.yaml"
    cas = RAGCassette(cass_path, mode="record")
    cas._put("embedder.embed", "abc123", [["test"]], [[0.1, 0.2]])
    cas.flush()

    data = yaml.safe_load(cass_path.read_text())
    assert isinstance(data, list)
    assert data[0]["call"] == "embedder.embed"
    assert data[0]["args_hash"] == "abc123"
    assert "response" in data[0]


def test_cassette_auto_records_on_first_run(tmp_path) -> None:
    cass_path = tmp_path / "auto.yaml"
    cas = RAGCassette(cass_path, mode="auto")
    assert cas._is_recording is True  # file doesn't exist yet


def test_cassette_auto_replays_when_file_exists(tmp_path) -> None:
    cass_path = tmp_path / "existing.yaml"
    cass_path.write_text("- call: test\n  args_hash: x\n  args: []\n  response: 42\n")
    cas = RAGCassette(cass_path, mode="auto")
    assert cas._is_recording is False  # file exists → replay


# ---------------------------------------------------------------------------
# assert_retrieval
# ---------------------------------------------------------------------------


def _make_answer(sources: list[str], score: float = 0.8) -> Answer:
    citations = [
        Citation(text="passage", source=s, chunk_id=f"c{i}", final_score=score)
        for i, s in enumerate(sources)
    ]
    return Answer(text="test", citations=citations, chunks_used=len(citations))


def test_assert_retrieval_must_include_source_pass() -> None:
    ans = _make_answer(["docs/policy.md", "docs/faq.md"])
    assert_retrieval(ans, must_include_source="docs/policy.md")  # should not raise


def test_assert_retrieval_must_include_source_fail() -> None:
    ans = _make_answer(["docs/faq.md"])
    with pytest.raises(AssertionError, match="policy.md"):
        assert_retrieval(ans, must_include_source="docs/policy.md")


def test_assert_retrieval_score_pass() -> None:
    ans = _make_answer(["docs/x.md"], score=0.9)
    assert_retrieval(ans, top_chunk_score_above=0.8)


def test_assert_retrieval_score_fail() -> None:
    ans = _make_answer(["docs/x.md"], score=0.4)
    with pytest.raises(AssertionError, match="0.8"):
        assert_retrieval(ans, top_chunk_score_above=0.8)


def test_assert_retrieval_no_citations_score_fail() -> None:
    ans = Answer(text="no context", citations=[], chunks_used=0)
    with pytest.raises(AssertionError, match="no citations"):
        assert_retrieval(ans, top_chunk_score_above=0.5)


def test_assert_retrieval_must_cite_n_sources_pass() -> None:
    ans = _make_answer(["a.md", "b.md", "c.md"])
    assert_retrieval(ans, must_cite_n_sources=2)


def test_assert_retrieval_must_cite_n_sources_fail() -> None:
    ans = _make_answer(["a.md"])
    with pytest.raises(AssertionError, match="3"):
        assert_retrieval(ans, must_cite_n_sources=3)


# ---------------------------------------------------------------------------
# generate_golden_dataset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_golden_dataset(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("The refund policy is 30 days.")
    out = tmp_path / "golden.yaml"

    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        await rag.ingest(str(tmp_path))
        entries = await generate_golden_dataset(rag, n=3, output_path=out)

    assert len(entries) >= 1
    assert all(isinstance(e, GoldenEntry) for e in entries)
    assert out.exists()

    loaded = load_golden_dataset(out)
    assert len(loaded) == len(entries)
    assert loaded[0].expected_source == entries[0].expected_source
