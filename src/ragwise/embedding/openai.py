"""OpenAI embedder — async, batched."""
from __future__ import annotations

from typing import Any


class OpenAIEmbedder:
    """Embeds text using the OpenAI embeddings API in batches.

    The OpenAI client is created lazily on the first embed() call so that
    instantiation does not require OPENAI_API_KEY to be set in the environment.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 32,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._api_key = api_key
        self._client: Any = None  # created lazily on first embed()

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            # When api_key is None the SDK reads OPENAI_API_KEY from env at call time.
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = await client.embeddings.create(
                model=self.model,
                input=batch,
            )
            results.extend(item.embedding for item in response.data)
        return results
