"""Agent tool adapters — make ragx callable from Claude and OpenAI agents."""
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
