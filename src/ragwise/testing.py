"""ragwise.testing — FakeEmbedder, FakeLLM, VCR cassettes, assert_retrieval helpers."""
from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ragwise.config import Answer

# ---------------------------------------------------------------------------
# Fake embedder / LLM — zero network, deterministic
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic embedder backed by seeded numpy RNG — no API calls.

    Vectors are unit-normalized and reproducible: same text always produces
    the same vector. Different texts produce (pseudo-)random distinct vectors.

    Example::

        emb = FakeEmbedder(dim=384)
        vecs = await emb.embed(["hello", "world"])
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            seed = abs(hash(t)) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v = v / np.linalg.norm(v)
            vecs.append(v.tolist())
        return vecs


class FakeLLM:
    """Deterministic LLM — returns a fixed string, no API calls.

    Example::

        llm = FakeLLM()
        answer = await llm.complete("What is the refund policy?")
    """

    def __init__(self, response: str = "") -> None:
        self._response = response

    async def complete(self, prompt: str) -> str:
        return self._response or f"Fake answer for: {prompt[:50]}"

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        text = self._response or f"Fake answer for: {prompt[:50]}"
        for char in text:
            yield char


# ---------------------------------------------------------------------------
# VCR cassette — record/replay for embedder + LLM calls
# ---------------------------------------------------------------------------


class RAGCassette:
    """Records embedder.embed() and llm.complete() calls to YAML; replays on subsequent runs."""

    def __init__(self, path: str | Path, mode: str = "auto") -> None:
        self._path = Path(path)
        self._mode = mode
        self._records: list[dict[str, Any]] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._path.exists():
            content = yaml.safe_load(self._path.read_text())
            self._records = content or []

    @property
    def _is_recording(self) -> bool:
        self._ensure_loaded()
        return self._mode == "record" or (self._mode == "auto" and not self._path.exists())

    def _args_hash(self, args: Any) -> str:
        return hashlib.md5(str(args).encode()).hexdigest()[:12]

    def _get(self, call: str, args_hash: str) -> Any:
        for rec in self._records:
            if rec["call"] == call and rec["args_hash"] == args_hash:
                return rec["response"]
        raise KeyError(f"No cassette record for {call}/{args_hash} — re-run in record mode")

    def _put(self, call: str, args_hash: str, args: Any, response: Any) -> None:
        self._records.append({"call": call, "args_hash": args_hash, "args": args, "response": response})

    def flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(yaml.dump(self._records, default_flow_style=False))

    def wrap_embedder(self, embedder: Any) -> Any:
        return _CassetteEmbedder(embedder, self)

    def wrap_llm(self, llm: Any) -> Any:
        return _CassetteLLM(llm, self)


class _CassetteEmbedder:
    def __init__(self, inner: Any, cassette: RAGCassette) -> None:
        self._inner = inner
        self._cas = cassette

    async def embed(self, texts: list[str]) -> list[list[float]]:
        h = self._cas._args_hash(texts)
        if self._cas._is_recording:
            result: list[list[float]] = await self._inner.embed(texts)
            self._cas._put("embedder.embed", h, texts, result)
            return result
        return self._cas._get("embedder.embed", h)  # type: ignore[no-any-return]


class _CassetteLLM:
    def __init__(self, inner: Any, cassette: RAGCassette) -> None:
        self._inner = inner
        self._cas = cassette

    async def complete(self, prompt: str) -> str:
        h = self._cas._args_hash(prompt)
        if self._cas._is_recording:
            result: str = await self._inner.complete(prompt)
            self._cas._put("llm.complete", h, prompt[:200], result)
            return result
        return self._cas._get("llm.complete", h)  # type: ignore[no-any-return]


@contextmanager
def cassette(path: str | Path, mode: str = "auto") -> Generator[RAGCassette, None, None]:
    """Context manager that records/replays ragwise API calls.

    On first run (file absent, mode='auto'): records all calls to the YAML file.
    On subsequent runs: replays from file — no network calls made.

    Example::

        with cassette("tests/cassettes/my_test.yaml") as cas:
            emb = cas.wrap_embedder(real_embedder)
            llm = cas.wrap_llm(real_llm)
            async with RAG(embedder=emb, llm=llm, cache=False) as rag:
                answer = await rag.query("What is the policy?")
    """
    cas = RAGCassette(path, mode)
    try:
        yield cas
    finally:
        if cas._is_recording and cas._records:
            cas.flush()


# ---------------------------------------------------------------------------
# assert_retrieval — test assertions for Answer objects
# ---------------------------------------------------------------------------


def assert_retrieval(
    answer: Answer,
    *,
    must_include_source: str | None = None,
    top_chunk_score_above: float | None = None,
    must_cite_n_sources: int | None = None,
) -> None:
    """Assert properties of a RAG answer.

    Raises AssertionError with a descriptive message on failure.

    Example::

        assert_retrieval(answer, must_include_source="docs/policy.md")
        assert_retrieval(answer, top_chunk_score_above=0.5)
        assert_retrieval(answer, must_cite_n_sources=2)
    """
    if must_include_source is not None:
        sources = answer.citation_sources
        if not any(must_include_source in s for s in sources):
            raise AssertionError(
                f"Expected source {must_include_source!r} in citations, "
                f"but got: {sources}"
            )

    if top_chunk_score_above is not None:
        if not answer.citations:
            raise AssertionError(
                f"Expected top chunk score > {top_chunk_score_above}, but no citations were returned"
            )
        top_score = answer.citations[0].final_score
        if top_score <= top_chunk_score_above:
            raise AssertionError(
                f"Expected top chunk score > {top_chunk_score_above}, got {top_score:.4f}"
            )

    if must_cite_n_sources is not None:
        unique_sources = len(set(answer.citation_sources))
        if unique_sources < must_cite_n_sources:
            raise AssertionError(
                f"Expected at least {must_cite_n_sources} unique source(s) cited, "
                f"got {unique_sources}: {answer.citation_sources}"
            )


# ---------------------------------------------------------------------------
# Golden dataset generation
# ---------------------------------------------------------------------------


@dataclass
class GoldenEntry:
    query: str
    expected_answer: str
    expected_source: str
    chunk_id: str


async def generate_golden_dataset(
    rag: Any,
    n: int = 50,
    output_path: str | Path | None = None,
) -> list[GoldenEntry]:
    """Generate Q/A pairs from the ingested corpus (1 LLM call per pair).

    For each sampled chunk, asks the LLM to generate a question answerable
    from that chunk. Saves to YAML if output_path is provided.

    Example::

        entries = await generate_golden_dataset(rag, n=20, output_path="tests/golden.yaml")
    """
    assert rag._store is not None, "RAG must be used as a context manager"
    assert rag._llm is not None

    # Sample docs from the store
    store = rag._store
    docs = getattr(store, "_docs", [])
    if not docs:
        return []

    import random
    sample = random.sample(docs, min(n, len(docs)))

    entries: list[GoldenEntry] = []
    for doc in sample:
        prompt = (
            f"Given this passage, write one specific question that can be answered directly "
            f"from it, then answer it. Format: Q: <question>\\nA: <answer>\\n\\nPassage:\\n{doc.text}"
        )
        response = await rag._llm.complete(prompt)
        lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
        question = next((ln[2:].strip() for ln in lines if ln.startswith("Q:")), doc.text[:50] + "?")
        answer = next((ln[2:].strip() for ln in lines if ln.startswith("A:")), response)
        entries.append(GoldenEntry(
            query=question,
            expected_answer=answer,
            expected_source=doc.source,
            chunk_id=doc.id,
        ))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"query": e.query, "expected_answer": e.expected_answer,
             "expected_source": e.expected_source, "chunk_id": e.chunk_id}
            for e in entries
        ]
        Path(output_path).write_text(yaml.dump(data, default_flow_style=False))

    return entries


def load_golden_dataset(path: str | Path) -> list[GoldenEntry]:
    """Load a golden dataset from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or []
    return [GoldenEntry(**item) for item in data]
