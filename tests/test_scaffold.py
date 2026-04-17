"""S1-T1 scaffold smoke tests — verifies the package installs and all stubs are importable."""
from __future__ import annotations

import ragwise
import ragwise.embedding
import ragwise.eval
import ragwise.generation
import ragwise.indexing
import ragwise.ingestion
import ragwise.retrieval
import ragwise.utils


def test_version() -> None:
    assert ragwise.__version__ == "0.2.0"


def test_all_subpackages_importable() -> None:
    # Each sub-package must be importable without errors
    for mod in (
        ragwise.embedding,
        ragwise.eval,
        ragwise.generation,
        ragwise.indexing,
        ragwise.ingestion,
        ragwise.retrieval,
        ragwise.utils,
    ):
        assert hasattr(mod, "__all__")
