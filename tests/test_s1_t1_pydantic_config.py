"""S1-T1: Pydantic RAGConfig — typed config, fail-fast validation, from_env, from_yaml."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragwise import QueryConfig, RAGConfig


# ---------- RAGConfig validation ----------

def test_ragconfig_defaults() -> None:
    cfg = RAGConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64
    assert cfg.batch_size == 32


def test_ragconfig_invalid_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_size must be > 0"):
        RAGConfig(chunk_size=-1)


def test_ragconfig_invalid_chunk_size_zero() -> None:
    with pytest.raises(ValidationError, match="chunk_size must be > 0"):
        RAGConfig(chunk_size=0)


def test_ragconfig_invalid_batch_size() -> None:
    with pytest.raises(ValidationError, match="batch_size must be > 0"):
        RAGConfig(batch_size=0)


def test_ragconfig_overlap_gte_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap.*must be < chunk_size"):
        RAGConfig(chunk_size=100, chunk_overlap=100)


def test_ragconfig_overlap_greater_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap.*must be < chunk_size"):
        RAGConfig(chunk_size=100, chunk_overlap=200)


def test_ragconfig_valid_custom() -> None:
    cfg = RAGConfig(llm="anthropic/claude-haiku-4-5", chunk_size=256, chunk_overlap=32)
    assert cfg.llm == "anthropic/claude-haiku-4-5"
    assert cfg.chunk_size == 256


# ---------- QueryConfig validation ----------

def test_queryconfig_invalid_alpha_above() -> None:
    with pytest.raises(ValidationError, match="alpha must be in"):
        QueryConfig(alpha=1.5)


def test_queryconfig_invalid_alpha_below() -> None:
    with pytest.raises(ValidationError, match="alpha must be in"):
        QueryConfig(alpha=-0.1)


def test_queryconfig_invalid_top_k() -> None:
    with pytest.raises(ValidationError, match="top_k must be > 0"):
        QueryConfig(top_k=0)


def test_queryconfig_invalid_threshold() -> None:
    with pytest.raises(ValidationError, match="sufficiency_threshold must be in"):
        QueryConfig(sufficiency_threshold=2.0)


def test_queryconfig_defaults() -> None:
    qc = QueryConfig()
    assert qc.alpha == 0.5
    assert qc.top_k == 10
    assert qc.allowed_sources == []


# ---------- from_env ----------

def test_ragconfig_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGWISE_LLM", "anthropic/claude-opus-4-7")
    monkeypatch.setenv("RAGWISE_STORE", "lance://./test-index")
    monkeypatch.setenv("RAGWISE_CHUNK_SIZE", "256")
    monkeypatch.setenv("RAGWISE_BATCH_SIZE", "16")

    cfg = RAGConfig.from_env()
    assert cfg.llm == "anthropic/claude-opus-4-7"
    assert cfg.store == "lance://./test-index"
    assert cfg.chunk_size == 256
    assert cfg.batch_size == 16


def test_ragconfig_from_env_defaults_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RAGWISE_LLM", "RAGWISE_STORE", "RAGWISE_EMBEDDER"):
        monkeypatch.delenv(key, raising=False)
    cfg = RAGConfig.from_env()
    assert cfg.llm == "openai/gpt-4o-mini"
    assert cfg.store == "memory"


def test_ragconfig_from_env_cache_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGWISE_CACHE", "false")
    cfg = RAGConfig.from_env()
    assert cfg.cache is False


# ---------- from_yaml ----------

def test_ragconfig_from_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "ragwise.yaml"
    yaml_file.write_text("llm: anthropic/claude-haiku-4-5\nchunk_size: 384\nbatch_size: 64\n")
    cfg = RAGConfig.from_yaml(yaml_file)
    assert cfg.llm == "anthropic/claude-haiku-4-5"
    assert cfg.chunk_size == 384
    assert cfg.batch_size == 64


def test_ragconfig_from_yaml_empty_file(tmp_path: Path) -> None:
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")
    cfg = RAGConfig.from_yaml(yaml_file)
    assert cfg.chunk_size == 512  # defaults preserved


def test_ragconfig_from_yaml_invalid_value(tmp_path: Path) -> None:
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("chunk_size: -10\n")
    with pytest.raises(ValidationError, match="chunk_size must be > 0"):
        RAGConfig.from_yaml(yaml_file)
