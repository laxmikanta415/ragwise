"""SentenceTransformer embedder — optional dep, runs in thread pool."""
from __future__ import annotations

try:
    from sentence_transformers import SentenceTransformer as _ST
except ImportError as _e:
    raise ImportError("pip install ragwise[local-emb]") from _e


class SentenceTransformerEmbedder:
    """Embeds text locally using sentence-transformers (runs in thread pool)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self._model = _ST(model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        import anyio

        def _encode() -> list[list[float]]:
            vectors = self._model.encode(texts, batch_size=self.batch_size)
            return vectors.tolist()  # type: ignore[no-any-return]

        return await anyio.to_thread.run_sync(_encode)
