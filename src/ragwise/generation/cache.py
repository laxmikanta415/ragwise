"""LLMCache and CachedLLM — prompt-level caching to eliminate redundant LLM calls."""
from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator
from typing import Any

from ragwise.generation.llm import LLMProtocol, StreamingLLMProtocol


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:32]


class LLMCache:
    """Prompt cache with in-memory or Redis backend."""

    def __init__(self, backend: str | bool | None = "memory") -> None:
        self._backend = backend
        self._store: dict[str, str] = {}
        self._redis: Any = None

        if isinstance(backend, str) and backend.startswith("redis"):
            try:
                import redis.asyncio as aioredis
            except ImportError as e:
                raise ImportError("pip install redis") from e
            self._redis = aioredis.from_url(backend)  # type: ignore[no-untyped-call]

    async def get(self, key: str) -> str | None:
        if not self._backend:
            return None
        if self._redis is not None:
            value: bytes | None = await self._redis.get(key)
            return value.decode() if value is not None else None
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:
        if not self._backend:
            return
        if self._redis is not None:
            await self._redis.set(key, value)
        else:
            self._store[key] = value


class CachedLLM:
    """Wraps an LLM with prompt-level caching. Streaming bypasses cache."""

    def __init__(self, llm: LLMProtocol, cache: LLMCache) -> None:
        self._llm = llm
        self._cache = cache

    async def complete(self, prompt: str) -> str:
        key = _cache_key(prompt)
        cached = await self._cache.get(key)
        if cached is not None:
            return cached
        result = await self._llm.complete(prompt)
        await self._cache.set(key, result)
        return result

    async def stream_complete(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream tokens — bypasses cache (streaming responses are not cacheable)."""
        if isinstance(self._llm, StreamingLLMProtocol):
            async for token in self._llm.stream_complete(prompt):
                yield token
        else:
            yield await self._llm.complete(prompt)
