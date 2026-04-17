"""Agent tool adapters — make ragwise callable from Claude and OpenAI agents."""
from __future__ import annotations

from typing import Any


def as_claude_tool(rag: Any) -> dict[str, Any]:
    """Return an Anthropic-compatible tool definition for ``rag.search()``.

    Pass the returned dict directly to ``anthropic.messages.create(tools=[...])``.

    Example::

        from ragwise.agent import as_claude_tool

        tool = as_claude_tool(rag)
        response = client.messages.create(
            model="claude-opus-4-6",
            tools=[tool],
            messages=[{"role": "user", "content": "What is the refund policy?"}],
        )
    """
    return {
        "name": "search_documents",
        "description": (
            "Search the document index and return the most relevant passages. "
            "Call this before answering any question that requires specific knowledge "
            "from the indexed documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — use natural language",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    }


def as_openai_tool(rag: Any) -> dict[str, Any]:
    """Return an OpenAI-compatible function tool definition for ``rag.search()``.

    Pass the returned dict directly to ``openai.chat.completions.create(tools=[...])``.

    Example::

        from ragwise.agent import as_openai_tool

        tool = as_openai_tool(rag)
        response = client.chat.completions.create(
            model="gpt-4o",
            tools=[tool],
            messages=[{"role": "user", "content": "What is the refund policy?"}],
        )
    """
    return {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the document index and return the most relevant passages. "
                "Call this before answering questions that require specific knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of passages to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }


def as_claude_tool_suite(rag: Any, max_iterations: int = 5) -> list[dict[str, Any]]:
    """Return 3 Anthropic-compatible tool defs for stateful agent search.

    Use with ``AgentSession`` to track state across calls::

        from ragwise.agent import as_claude_tool_suite
        from ragwise.agent_session import AgentSession

        tools = as_claude_tool_suite(rag)
        session = AgentSession(rag, max_iterations=5)
        # Dispatch tool calls to session.search(), session.get_context(), session.context_budget_status()
    """
    return [
        {
            "name": "search_documents",
            "description": (
                "Search the document index and return the most relevant passages. "
                "Tracks which chunks have already been retrieved to avoid duplicates. "
                "Call this before answering any question that requires specific knowledge."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — use natural language",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of passages to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_document_context",
            "description": (
                "Retrieve the chunks immediately surrounding a known chunk_id. "
                "Use this to get more context around a relevant passage without a new search."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk_id returned by a prior search_documents call",
                    },
                    "window": {
                        "type": "integer",
                        "description": "Number of chunks before and after to return (default: 2)",
                        "default": 2,
                    },
                },
                "required": ["chunk_id"],
            },
        },
        {
            "name": "check_context_budget",
            "description": (
                "Check how many chunks and tokens have been retrieved so far, "
                "and how many remain before hitting the context budget."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
    ]


def as_openai_tool_suite(rag: Any, max_iterations: int = 5) -> list[dict[str, Any]]:
    """Return 3 OpenAI-compatible function tool defs for stateful agent search.

    Use with ``AgentSession`` to track state across calls::

        from ragwise.agent import as_openai_tool_suite
        from ragwise.agent_session import AgentSession

        tools = as_openai_tool_suite(rag)
        session = AgentSession(rag, max_iterations=5)
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": (
                    "Search the document index and return the most relevant passages. "
                    "Tracks which chunks have already been retrieved to avoid duplicates."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "top_k": {"type": "integer", "description": "Number of passages to return", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document_context",
                "description": "Retrieve chunks surrounding a known chunk_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "chunk_id from a prior search"},
                        "window": {"type": "integer", "description": "Chunks before/after to return", "default": 2},
                    },
                    "required": ["chunk_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_context_budget",
                "description": "Check chunks and tokens retrieved so far vs budget.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
