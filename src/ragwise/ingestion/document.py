"""Document dataclass and source-hashing utility for incremental indexing."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Document:
    """Universal data container flowing through the entire ragwise pipeline."""

    id: str = field(default_factory=lambda: uuid4().hex)
    text: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def hash_source(path: Path) -> str:
    """Return the first 16 hex characters of the SHA-256 hash of a file's bytes.

    Reads raw bytes (encoding-agnostic) for consistent hashing across platforms.
    Used by ingest() to skip unchanged files during incremental indexing.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]
