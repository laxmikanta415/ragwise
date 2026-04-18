"""SemanticCache — embedding-similarity cache replacing SHA-256 exact match."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


class MemorySemanticCache:
    """In-process embedding similarity cache.

    Stores (query_embedding, Answer) pairs. Lookup finds the nearest stored
    embedding and returns its answer if the cosine similarity exceeds *threshold*.

    Example::

        cache = MemorySemanticCache(threshold=0.92)
        cache.store(query_vec, answer)
        cached, sim = cache.lookup(query_vec)
    """

    def __init__(self, threshold: float = 0.92) -> None:
        self.threshold = threshold
        self._entries: list[tuple[np.ndarray, Any]] = []
        self._hits: int = 0
        self._misses: int = 0

    def lookup(
        self, query_vec: list[float], threshold: float | None = None
    ) -> tuple[Any, float | None]:
        """Return (answer, similarity) if a similar cached query exists, else (None, None)."""
        if not self._entries:
            self._misses += 1
            return None, None

        t = threshold if threshold is not None else self.threshold
        qv = np.array(query_vec, dtype=np.float32)
        qnorm = float(np.linalg.norm(qv))
        if qnorm == 0:
            self._misses += 1
            return None, None

        best_score = -1.0
        best_answer: Any = None
        for emb, answer in self._entries:
            enorm = float(np.linalg.norm(emb))
            if enorm == 0:
                continue
            score = float(np.dot(emb, qv) / (enorm * qnorm))
            if score > best_score:
                best_score = score
                best_answer = answer

        if best_answer is not None and best_score >= t:
            self._hits += 1
            return best_answer, best_score

        self._misses += 1
        return None, None

    def store(self, query_vec: list[float], answer: Any) -> None:
        """Store an answer indexed by query embedding."""
        emb = np.array(query_vec, dtype=np.float32)
        self._entries.append((emb, answer))

    def invalidate(self, query_vec: list[float], threshold: float = 0.99) -> int:
        """Remove entries whose embedding is above *threshold* similar to query_vec.

        Returns the number of entries removed.
        """
        qv = np.array(query_vec, dtype=np.float32)
        qnorm = float(np.linalg.norm(qv))
        if qnorm == 0:
            return 0

        before = len(self._entries)
        self._entries = [
            (emb, ans)
            for emb, ans in self._entries
            if float(np.dot(emb, qv) / (float(np.linalg.norm(emb)) * qnorm + 1e-10)) < threshold
        ]
        return before - len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "entries": len(self._entries),
        }


class RedisSemanticCache:
    """Redis-backed embedding similarity cache (survives restarts).

    Requires ``pip install ragwise[cache]`` (redis>=5.0).
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        threshold: float = 0.92,
        ttl: int = 86400,
    ) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError("pip install ragwise[cache]") from e

        self.threshold = threshold
        self.ttl = ttl
        self._redis = aioredis.from_url(url)
        self._hits: int = 0
        self._misses: int = 0

    async def lookup(
        self, query_vec: list[float], threshold: float | None = None
    ) -> tuple[Any, float | None]:
        import json as _json

        t = threshold if threshold is not None else self.threshold
        qv = np.array(query_vec, dtype=np.float32)
        qnorm = float(np.linalg.norm(qv))
        if qnorm == 0:
            self._misses += 1
            return None, None

        keys: list[bytes] = await self._redis.keys("ragwise:cache:*")
        best_score = -1.0
        best_answer: Any = None

        for key in keys:
            raw: bytes | None = await self._redis.get(key)
            if raw is None:
                continue
            try:
                entry = _json.loads(raw)
                emb = np.array(entry["embedding"], dtype=np.float32)
                enorm = float(np.linalg.norm(emb))
                if enorm == 0:
                    continue
                score = float(np.dot(emb, qv) / (enorm * qnorm))
                if score > best_score:
                    best_score = score
                    best_answer = entry["answer"]
            except Exception:
                continue

        if best_answer is not None and best_score >= t:
            self._hits += 1
            return best_answer, best_score

        self._misses += 1
        return None, None

    async def store(self, query_vec: list[float], answer: Any) -> None:
        import hashlib
        import json as _json

        key = "ragwise:cache:" + hashlib.sha256(str(query_vec[:8]).encode()).hexdigest()[:16]
        entry = {"embedding": [float(x) for x in query_vec], "answer": answer}
        await self._redis.set(key, _json.dumps(entry), ex=self.ttl)

    async def clear(self) -> None:
        keys: list[bytes] = await self._redis.keys("ragwise:cache:*")
        if keys:
            await self._redis.delete(*keys)
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }
