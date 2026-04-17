"""Tests for LLMProtocol, resolve_llm, OpenAILLM, AnthropicLLM, LLMCache, CachedLLM, Assembler — S4-T3/T4/T5."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragwise.generation import (
    AnthropicLLM,
    Assembler,
    CachedLLM,
    LLMCache,
    LLMProtocol,
    OllamaLLM,
    OpenAILLM,
    RAG_PROMPT,
    resolve_llm,
)
from ragwise.indexing.base import SearchResult


# ---------------------------------------------------------------------------
# S4-T3 — resolve_llm + LLM adapters
# ---------------------------------------------------------------------------

def test_resolve_llm_openai() -> None:
    llm = resolve_llm("openai/gpt-4o-mini")
    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gpt-4o-mini"


def test_resolve_llm_anthropic() -> None:
    llm = resolve_llm("anthropic/claude-haiku-4-5-20251001")
    assert isinstance(llm, AnthropicLLM)


def test_resolve_llm_ollama() -> None:
    llm = resolve_llm("ollama/llama3")
    assert isinstance(llm, OllamaLLM)
    assert llm.model == "llama3"


def test_resolve_llm_passthrough() -> None:
    mock = MagicMock()
    mock.complete = AsyncMock()
    assert resolve_llm(mock) is mock


def test_resolve_llm_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        resolve_llm("unknown/model")


def test_llm_protocol_structural() -> None:
    llm: LLMProtocol = OpenAILLM()  # type: ignore[assignment]
    assert callable(llm.complete)


@pytest.mark.asyncio
async def test_openai_llm_complete_mocked() -> None:
    msg = MagicMock()
    msg.content = "Paris"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]

    llm = OpenAILLM()
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=resp)
    llm._client = mock_client

    result = await llm.complete("What is the capital of France?")
    assert result == "Paris"


@pytest.mark.asyncio
async def test_anthropic_llm_complete_mocked() -> None:
    content_block = MagicMock()
    content_block.text = "The answer is 42."
    resp = MagicMock()
    resp.content = [content_block]

    llm = AnthropicLLM()
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=resp)
    llm._client = mock_client

    result = await llm.complete("What is the answer?")
    assert result == "The answer is 42."


# ---------------------------------------------------------------------------
# S4-T4 — LLMCache + CachedLLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_cache_memory_get_miss() -> None:
    cache = LLMCache("memory")
    assert await cache.get("nonexistent") is None


@pytest.mark.asyncio
async def test_cached_llm_second_call_no_api_call() -> None:
    base_llm = MagicMock()
    base_llm.complete = AsyncMock(return_value="cached answer")
    cache = LLMCache("memory")
    cached = CachedLLM(llm=base_llm, cache=cache)

    r1 = await cached.complete("hello")
    r2 = await cached.complete("hello")

    assert r1 == r2 == "cached answer"
    assert base_llm.complete.call_count == 1  # only called once


@pytest.mark.asyncio
async def test_cached_llm_different_prompts_different_keys() -> None:
    base_llm = MagicMock()
    base_llm.complete = AsyncMock(side_effect=["a1", "a2"])
    cache = LLMCache("memory")
    cached = CachedLLM(llm=base_llm, cache=cache)

    r1 = await cached.complete("prompt A")
    r2 = await cached.complete("prompt B")

    assert r1 == "a1"
    assert r2 == "a2"
    assert base_llm.complete.call_count == 2


@pytest.mark.asyncio
async def test_cached_llm_disabled_calls_every_time() -> None:
    base_llm = MagicMock()
    base_llm.complete = AsyncMock(return_value="answer")
    cache = LLMCache(False)
    cached = CachedLLM(llm=base_llm, cache=cache)

    await cached.complete("same prompt")
    await cached.complete("same prompt")

    assert base_llm.complete.call_count == 2


# ---------------------------------------------------------------------------
# S4-T5 — Assembler
# ---------------------------------------------------------------------------

def _sr(id: str, text: str, source: str) -> SearchResult:
    return SearchResult(id=id, text=text, source=source, score=0.9)


def test_assembler_includes_results_in_prompt() -> None:
    asm = Assembler()
    r1 = _sr("a", "Refunds are accepted within 30 days.", "policy.txt")
    r2 = _sr("b", "Contact support at help@example.com.", "contact.txt")
    prompt, _, _ = asm.assemble("What is the refund policy?", [r1, r2])
    assert "Refunds are accepted" in prompt
    assert "What is the refund policy?" in prompt


def test_assembler_empty_results_no_error() -> None:
    asm = Assembler()
    prompt, citations, dropped = asm.assemble("question", [])
    assert "question" in prompt
    assert citations == []
    assert dropped == []


def test_assembler_citations_unique() -> None:
    asm = Assembler()
    r1 = _sr("a", "chunk one", "doc.txt")
    r2 = _sr("b", "chunk two", "doc.txt")  # same source
    _, citations, _ = asm.assemble("q", [r1, r2])
    # Both chunks cited (one per chunk), same source
    assert all(c.source == "doc.txt" for c in citations)
    assert len(citations) == 2


def test_assembler_respects_token_budget() -> None:
    """With a very small budget, only the first result fits."""
    asm = Assembler(max_context_tokens=30)
    long_text = "word " * 200
    r1 = _sr("a", long_text, "a.txt")
    r2 = _sr("b", "short", "b.txt")
    _, citations, dropped = asm.assemble("q", [r1, r2])
    assert len(citations) + len(dropped) == 2


def test_rag_prompt_has_placeholders() -> None:
    assert "{context}" in RAG_PROMPT
    assert "{question}" in RAG_PROMPT


# ---------------------------------------------------------------------------
# S6-T6 — Assembler uses parent_text when present
# ---------------------------------------------------------------------------

def test_assembler_uses_parent_text_when_present() -> None:
    from ragwise.indexing.base import SearchResult

    result = SearchResult(
        id="c1",
        text="short child chunk",
        source="doc.txt",
        score=0.9,
        metadata={"parent_text": "This is the full parent paragraph with much more detail."},
    )

    asm = Assembler()
    prompt, _, _ = asm.assemble("what is this?", [result])
    assert "full parent paragraph" in prompt
    assert "short child chunk" not in prompt


def test_assembler_falls_back_to_text_without_parent() -> None:
    from ragwise.indexing.base import SearchResult

    result = SearchResult(id="c2", text="standalone chunk", source="doc.txt", score=0.9)
    asm = Assembler()
    prompt, _, _ = asm.assemble("query", [result])
    assert "standalone chunk" in prompt


# ---------------------------------------------------------------------------
# S7-T1 — StreamingLLMProtocol + stream_complete() on all adapters
# ---------------------------------------------------------------------------

from ragwise.generation.llm import StreamingLLMProtocol


def test_streaming_protocol_exists() -> None:
    assert StreamingLLMProtocol is not None


def test_openai_llm_implements_streaming_protocol() -> None:
    llm = OpenAILLM()
    assert isinstance(llm, StreamingLLMProtocol)


def test_anthropic_llm_implements_streaming_protocol() -> None:
    llm = AnthropicLLM()
    assert isinstance(llm, StreamingLLMProtocol)


def test_ollama_llm_implements_streaming_protocol() -> None:
    llm = OllamaLLM()
    assert isinstance(llm, StreamingLLMProtocol)


@pytest.mark.asyncio
async def test_openai_stream_complete_yields_tokens() -> None:
    async def _fake_stream() -> None:  # type: ignore[misc]
        tokens = ["Hello", " world", "!"]
        for t in tokens:
            chunk = MagicMock()
            chunk.choices[0].delta.content = t
            yield chunk

    llm = OpenAILLM()
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_fake_stream())
    llm._client = mock_client

    tokens: list[str] = []
    async for token in llm.stream_complete("prompt"):
        tokens.append(token)

    assert tokens == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_cached_llm_stream_complete_bypasses_cache() -> None:
    from ragwise.generation.cache import CachedLLM, LLMCache

    async def _gen() -> None:  # type: ignore[misc]
        yield "token1"
        yield "token2"

    inner = MagicMock(spec=StreamingLLMProtocol)
    inner.stream_complete = MagicMock(return_value=_gen())

    cache = LLMCache()
    cached = CachedLLM(llm=inner, cache=cache)

    tokens: list[str] = []
    async for token in cached.stream_complete("prompt"):
        tokens.append(token)

    assert tokens == ["token1", "token2"]
    # complete() should NOT have been called
    inner.complete.assert_not_called()  # type: ignore[attr-defined]
