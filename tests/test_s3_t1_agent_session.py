"""S3-T1: AgentSession + as_claude_tool_suite + as_openai_tool_suite."""
from __future__ import annotations

import logging

import pytest

from ragwise import RAG
from ragwise.agent import as_claude_tool, as_claude_tool_suite, as_openai_tool_suite
from ragwise.agent_session import AgentSession
from ragwise.testing import FakeEmbedder, FakeLLM


async def _make_rag(tmp_path):
    rag = RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False)
    await rag.__aenter__()
    (tmp_path / "a.txt").write_text("Refund policy: 30 day returns accepted.")
    (tmp_path / "b.txt").write_text("Shipping takes 3-5 business days.")
    await rag.ingest(str(tmp_path))
    return rag


@pytest.mark.asyncio
async def test_agent_session_deduplication(tmp_path) -> None:
    rag = await _make_rag(tmp_path)
    try:
        session = AgentSession(rag, max_iterations=5)
        chunks1 = await session.search("refund policy", top_k=3)
        chunks2 = await session.search("refund policy", top_k=3)

        # Second search returns no NEW chunks (all deduplicated)
        total_unique = len(session.retrieved_chunks)
        assert total_unique == len(chunks1)  # chunks2 are all dupes
        assert len(chunks2) == 0
    finally:
        await rag.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_agent_session_loop_warning(tmp_path, caplog) -> None:
    rag = await _make_rag(tmp_path)
    try:
        session = AgentSession(rag)
        with caplog.at_level(logging.WARNING, logger="ragwise.agent_session"):
            await session.search("refund policy")
            await session.search("refund policy")  # identical query → loop warning

        assert any("loop" in r.message.lower() for r in caplog.records)
    finally:
        await rag.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_get_document_context(tmp_path) -> None:
    rag = await _make_rag(tmp_path)
    try:
        session = AgentSession(rag)
        results = await session.search("refund", top_k=1)
        assert results, "Expected at least one result"

        chunk_id = results[0].chunk_id
        context = await session.get_context(chunk_id, window=2)
        # Should return the chunk itself at minimum
        assert any(c.chunk_id == chunk_id for c in context)
    finally:
        await rag.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_check_context_budget(tmp_path) -> None:
    rag = await _make_rag(tmp_path)
    try:
        session = AgentSession(rag, max_iterations=5, context_budget_tokens=8000)
        await session.search("refund")
        await session.search("shipping")
        await session.search("policy returns")

        budget = session.context_budget_status()
        assert budget["iteration"] == 3
        assert budget["tokens_used"] > 0
        assert budget["chunks_retrieved"] > 0
        assert budget["tokens_remaining"] <= 8000
    finally:
        await rag.__aexit__(None, None, None)


def test_tool_suite_anthropic_format() -> None:
    tools = as_claude_tool_suite(None)
    assert len(tools) == 3
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool


def test_tool_suite_openai_format() -> None:
    tools = as_openai_tool_suite(None)
    assert len(tools) == 3
    for tool in tools:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "parameters" in tool["function"]


def test_tool_suite_names() -> None:
    tools = as_claude_tool_suite(None)
    names = {t["name"] for t in tools}
    assert names == {"search_documents", "get_document_context", "check_context_budget"}


def test_backward_compat_single_tool() -> None:
    tool = as_claude_tool(None)
    assert isinstance(tool, dict)
    assert tool["name"] == "search_documents"
    assert "input_schema" in tool
