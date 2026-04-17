"""Tests for count_tokens() and truncate_to_tokens() — S1-T5."""
from __future__ import annotations

from ragwise.utils import count_tokens, truncate_to_tokens


def test_count_tokens_positive() -> None:
    assert count_tokens("hello world") > 0


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0


def test_count_tokens_known_value() -> None:
    # tiktoken gpt-4o: "hello world" = 2 tokens
    assert count_tokens("hello world") == 2


def test_truncate_to_tokens_respects_limit() -> None:
    long_text = "word " * 100  # ~100 tokens
    result = truncate_to_tokens(long_text, 10)
    assert count_tokens(result) <= 10


def test_truncate_to_tokens_short_text_unchanged() -> None:
    short_text = "hi"
    assert truncate_to_tokens(short_text, 1000) == short_text


def test_truncate_to_tokens_exact_limit_unchanged() -> None:
    text = "hello world"
    token_count = count_tokens(text)
    assert truncate_to_tokens(text, token_count) == text


def test_truncate_to_tokens_returns_string() -> None:
    result = truncate_to_tokens("some text here", 2)
    assert isinstance(result, str)


def test_unknown_model_fallback_no_exception() -> None:
    # Should not raise — falls back to cl100k_base
    count = count_tokens("hello", model="unknown-model-xyz")
    assert count > 0


def test_unknown_model_truncate_no_exception() -> None:
    result = truncate_to_tokens("hello world", 1, model="anthropic/claude-3")
    assert isinstance(result, str)
    assert count_tokens(result, model="anthropic/claude-3") <= 1
