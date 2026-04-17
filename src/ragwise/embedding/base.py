"""EmbedderProtocol and resolve_embedder() factory."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbedderProtocol(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def resolve_embedder(spec: str | Any) -> EmbedderProtocol:
    """Return an embedder instance from a string shorthand or pass-through.

    Supported prefixes:
      - ``"openai/*"``                    → OpenAIEmbedder
      - ``"local/*"`` / ``"sentence-transformers/*"`` → SentenceTransformerEmbedder

    If *spec* is already an object with an ``embed`` method it is returned unchanged.
    Raises ``ValueError`` for unrecognised strings.
    """
    if not isinstance(spec, str):
        if hasattr(spec, "embed"):
            return spec  # type: ignore[no-any-return]  # duck-type pass-through
        raise ValueError(f"Expected a string spec or an embedder object, got {type(spec)!r}")

    prefix = spec.split("/")[0].lower()

    if prefix == "openai":
        from ragwise.embedding.openai import OpenAIEmbedder

        model = "/".join(spec.split("/")[1:]) or "text-embedding-3-small"
        return OpenAIEmbedder(model=model)

    if prefix in ("local", "sentence-transformers"):
        from ragwise.embedding.local import SentenceTransformerEmbedder

        model = "/".join(spec.split("/")[1:]) or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model=model)

    raise ValueError(f"Unknown embedder spec: {spec!r}. Expected 'openai/*' or 'local/*'.")
