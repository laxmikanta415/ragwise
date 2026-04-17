"""Tests for Document dataclass and hash_source() — S1-T2."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ragwise.ingestion import Document, hash_source


def test_document_auto_id() -> None:
    doc = Document()
    assert isinstance(doc.id, str)
    assert len(doc.id) == 32  # uuid4().hex is 32 hex chars


def test_document_explicit_fields() -> None:
    doc = Document(id="x", text="t", source="s")
    assert doc.id == "x"
    assert doc.text == "t"
    assert doc.source == "s"
    assert doc.metadata == {}


def test_document_two_auto_id_instances_not_equal() -> None:
    doc1 = Document(text="same", source="same")
    doc2 = Document(text="same", source="same")
    # Different auto-generated ids → not equal
    assert doc1.id != doc2.id
    assert doc1 != doc2


def test_document_is_frozen() -> None:
    doc = Document(id="x", text="t", source="s")
    with pytest.raises(FrozenInstanceError):
        doc.text = "mutated"  # type: ignore[misc]


def test_document_metadata_default_is_independent() -> None:
    doc1 = Document()
    doc2 = Document()
    # Each instance gets its own dict — no shared mutable default
    assert doc1.metadata is not doc2.metadata


def test_hash_source_returns_16_char_hex(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello world")
    result = hash_source(f)
    assert len(result) == 16
    assert all(c in "0123456789abcdef" for c in result)


def test_hash_source_consistent(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_bytes(b"consistent content")
    assert hash_source(f) == hash_source(f)


def test_hash_source_changes_with_content(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_bytes(b"version 1")
    h1 = hash_source(f)
    f.write_bytes(b"version 2")
    h2 = hash_source(f)
    assert h1 != h2
