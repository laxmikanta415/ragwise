"""Token counting and truncation utilities for LLM context budget enforcement."""
from __future__ import annotations

import functools

import tiktoken


@functools.lru_cache(maxsize=16)
def _get_encoding(model: str) -> tiktoken.Encoding:
    """Return a cached tiktoken encoding for the given model name.

    Falls back to cl100k_base for unknown models (covers Anthropic, local models).
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return the number of tokens in *text* for the given model."""
    if not text:
        return 0
    return len(_get_encoding(model).encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Return *text* truncated so it contains at most *max_tokens* tokens.

    Uses encode → slice → decode to avoid any character-count estimation.
    Returns the original string unchanged if it already fits within the budget.
    """
    enc = _get_encoding(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
