"""PgVectorStore — PostgreSQL + pgvector + FTS (optional dep)."""
from __future__ import annotations

import json
from typing import Any

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError as _e:
    raise ImportError("pip install ragwise[postgres]") from _e

from ragwise.indexing.base import EmbeddedDoc, SearchResult, VectorStore

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {table} (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    embedding vector({dim}),
    metadata JSONB DEFAULT '{{}}',
    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS {table}_fts_idx ON {table} USING GIN (fts);
CREATE INDEX IF NOT EXISTS {table}_emb_idx ON {table} USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


class PgVectorStore(VectorStore):
    """Production vector store backed by PostgreSQL + pgvector + FTS."""

    def __init__(self, dsn: str, table: str = "ragx_docs") -> None:
        self._dsn = dsn
        self._table = table
        self._dim: int | None = None

    async def _connect(self) -> psycopg.AsyncConnection[Any]:
        conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        await register_vector(conn)
        return conn

    async def _ensure_table(self, conn: psycopg.AsyncConnection[Any], dim: int) -> None:
        ddl = _DDL.format(table=self._table, dim=dim)
        async with conn.cursor() as cur:
            await cur.execute(ddl)

    async def upsert(self, docs: list[EmbeddedDoc]) -> None:
        if not docs:
            return
        dim = len(docs[0].embedding)
        conn = await self._connect()
        async with conn:
            await self._ensure_table(conn, dim)
            sql = f"""
                INSERT INTO {self._table} (id, text, source, content_hash, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    source = EXCLUDED.source,
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """
            async with conn.cursor() as cur:
                for doc in docs:
                    import numpy as np

                    emb = np.array(doc.embedding, dtype=np.float32)
                    content_hash = doc.metadata.get("content_hash", "")
                    meta_json = json.dumps(doc.metadata)
                    await cur.execute(sql, (doc.id, doc.text, doc.source, content_hash, emb, meta_json))

    async def dense_search(self, query_vec: list[float], top_k: int) -> list[SearchResult]:
        import numpy as np

        conn = await self._connect()
        async with conn:
            sql = f"""
                SELECT id, text, source, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self._table}
                ORDER BY score DESC
                LIMIT %s
            """
            async with conn.cursor() as cur:
                qv = np.array(query_vec, dtype=np.float32)
                await cur.execute(sql, (qv, top_k))
                rows = await cur.fetchall()
        return [
            SearchResult(
                id=r[0], text=r[1], source=r[2],
                metadata=r[3] if isinstance(r[3], dict) else json.loads(r[3] or "{}"),
                score=float(r[4]),
            )
            for r in rows
        ]

    async def sparse_search(self, query: str, top_k: int) -> list[SearchResult]:
        conn = await self._connect()
        async with conn:
            sql = f"""
                SELECT id, text, source, metadata,
                       ts_rank(fts, plainto_tsquery('english', %s)) AS score
                FROM {self._table}
                WHERE fts @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """
            async with conn.cursor() as cur:
                await cur.execute(sql, (query, query, top_k))
                rows = await cur.fetchall()
        return [
            SearchResult(
                id=r[0], text=r[1], source=r[2],
                metadata=r[3] if isinstance(r[3], dict) else json.loads(r[3] or "{}"),
                score=float(r[4]),
            )
            for r in rows
        ]

    async def delete(self, source: str) -> None:
        conn = await self._connect()
        async with conn, conn.cursor() as cur:
            await cur.execute(f"DELETE FROM {self._table} WHERE source = %s", (source,))

    async def get_indexed_sources(self) -> dict[str, str]:
        conn = await self._connect()
        async with conn, conn.cursor() as cur:
            await cur.execute(
                f"SELECT DISTINCT source, content_hash FROM {self._table}"
            )
            rows = await cur.fetchall()
        return {r[0]: r[1] for r in rows}
