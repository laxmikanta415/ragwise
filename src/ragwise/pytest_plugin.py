"""ragwise pytest plugin — registers fake_rag and recorded_rag fixtures.

Register by adding to conftest.py::

    from ragwise.pytest_plugin import *  # noqa: F401, F403

Or install as a pytest plugin via the entry point in pyproject.toml.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import pytest

from ragwise import RAG
from ragwise.testing import FakeEmbedder, FakeLLM


@pytest.fixture
async def fake_rag() -> AsyncGenerator[RAG, None]:
    """A RAG instance wired with FakeEmbedder and FakeLLM — no network, no API keys."""
    async with RAG(embedder=FakeEmbedder(), llm=FakeLLM(), cache=False) as rag:
        yield rag


@pytest.fixture
def fake_embedder() -> Generator[FakeEmbedder, None, None]:
    """A FakeEmbedder(dim=384) instance."""
    yield FakeEmbedder(dim=384)


@pytest.fixture
def fake_llm() -> Generator[FakeLLM, None, None]:
    """A FakeLLM instance."""
    yield FakeLLM()
