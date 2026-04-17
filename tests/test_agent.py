"""Tests for ragx.agent — as_claude_tool() and as_openai_tool()."""
from __future__ import annotations

from ragwise.agent import as_claude_tool, as_openai_tool


def test_claude_tool_name() -> None:
    tool = as_claude_tool(None)
    assert tool["name"] == "search_documents"


def test_claude_tool_has_description() -> None:
    tool = as_claude_tool(None)
    assert "description" in tool
    assert len(tool["description"]) > 10


def test_claude_tool_input_schema_required() -> None:
    tool = as_claude_tool(None)
    assert tool["input_schema"]["required"] == ["query"]


def test_claude_tool_input_schema_properties() -> None:
    tool = as_claude_tool(None)
    props = tool["input_schema"]["properties"]
    assert "query" in props
    assert "top_k" in props
    assert props["query"]["type"] == "string"
    assert props["top_k"]["type"] == "integer"


def test_openai_tool_type() -> None:
    tool = as_openai_tool(None)
    assert tool["type"] == "function"


def test_openai_tool_function_name() -> None:
    tool = as_openai_tool(None)
    assert tool["function"]["name"] == "search_documents"


def test_openai_tool_required() -> None:
    tool = as_openai_tool(None)
    assert tool["function"]["parameters"]["required"] == ["query"]


def test_openai_tool_properties() -> None:
    tool = as_openai_tool(None)
    props = tool["function"]["parameters"]["properties"]
    assert "query" in props
    assert "top_k" in props


def test_tools_exported_from_ragx() -> None:
    from ragwise import as_claude_tool as act
    from ragwise import as_openai_tool as aot

    assert act is as_claude_tool
    assert aot is as_openai_tool
