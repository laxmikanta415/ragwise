"""Shared pytest fixtures for ragwise tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def tmp_docs_dir(tmp_path: Path) -> Path:
    """A temporary directory pre-populated with sample documents."""
    (tmp_path / "sample.txt").write_text("This is a sample document about refund policies.")
    (tmp_path / "sample.md").write_text("# Guide\n\nReturns are accepted within 30 days.")
    return tmp_path


@pytest.fixture
def sample_text() -> str:
    return "ragwise is a pip-installable Python RAG library with hybrid search on by default."


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedder returning 384-dim zero vectors."""
    emb = MagicMock()
    emb.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 384 for _ in texts])
    return emb


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM that always returns 'test answer'."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="test answer")
    return llm
