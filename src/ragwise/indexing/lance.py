"""LanceDBStore — embedded persistent vector store (optional dep)."""
from __future__ import annotations

import contextlib
import json
from typing import Any

try:
    import lancedb
except ImportError as _e:
    raise ImportError("pip install ragwise[lance]") from _e

from ragwise.indexing.base import EmbeddedDoc, SearchResult, VectorStore

_TABLE = "ragx_docs"


def _row_to_search_result(row: dict[str, Any], score: float) -> SearchResult:
    metadata: dict[str, Any] = {}
    raw = row.get("metadata_json", "{}")
    if raw:
        with contextlib.suppress(Exception):
            metadata = json.loads(raw)
    return SearchResult(
        id=str(row["id"]),
        text=str(row["text"]),
        source=str(row["source"]),
        score=score,
        metadata=metadata,
    )


class LanceDBStore(VectorStore):
    """Persistent embedded store backed by LanceDB."""

    def __init__(self, uri: str = "./ragx-lance") -> None:
        self._uri = str(uri)
        self._db = lancedb.connect(self._uri)
        self._table: Any = None  # lazily opened/created

    def _get_table(self) -> Any:
        if self._table is not None:
            return self._table
        if _TABLE in self._db.table_names():
            self._table = self._db.open_table(_TABLE)
        return self._table

    async def upsert(self, docs: list[EmbeddedDoc]) -> None:
        if not docs:
            return

        import pyarrow as pa

        rows = [
            {
                "id": d.id,
                "text": d.text,
                "source": d.source,
                "embedding": d.embedding,
                "metadata_json": json.dumps(d.metadata),
            }
            for d in docs
        ]
        dim = len(docs[0].embedding)

        if _TABLE not in self._db.table_names():
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("embedding", pa.list_(pa.float32(), dim)),
                    pa.field("metadata_json", pa.string()),
                ]
            )
            self._table = self._db.create_table(_TABLE, data=rows, schema=schema)
            with contextlib.suppress(Exception):
                self._table.create_fts_index("text")
        else:
            tbl = self._get_table()
            # Merge-insert for upsert semantics
            existing_ids = {str(r["id"]) for r in tbl.to_arrow().to_pydict().get("id", [])}
            new_rows = [r for r in rows if r["id"] not in existing_ids]
            update_rows = [r for r in rows if r["id"] in existing_ids]
            if new_rows:
                tbl.add(new_rows)
            for row in update_rows:
                tbl.delete(f"id = '{row['id']}'")
                tbl.add([row])
            self._table = tbl

    async def dense_search(self, query_vec: list[float], top_k: int) -> list[SearchResult]:
        tbl = self._get_table()
        if tbl is None:
            return []
        try:
            results = tbl.search(query_vec).limit(top_k).to_list()
        except Exception:
            return []
        return [_row_to_search_result(r, r.get("_distance", 0.0)) for r in results]

    async def sparse_search(self, query: str, top_k: int) -> list[SearchResult]:
        tbl = self._get_table()
        if tbl is None:
            return []
        try:
            results = tbl.search(query, query_type="fts").limit(top_k).to_list()
        except Exception:
            return []
        return [_row_to_search_result(r, r.get("score", 0.0)) for r in results]

    async def delete(self, source: str) -> None:
        tbl = self._get_table()
        if tbl is None:
            return
        tbl.delete(f"source = '{source}'")

    async def get_indexed_sources(self) -> dict[str, str]:
        tbl = self._get_table()
        if tbl is None:
            return {}
        data = tbl.to_arrow().to_pydict()
        sources = data.get("source", [])
        meta_jsons = data.get("metadata_json", [])
        result: dict[str, str] = {}
        for src, mjson in zip(sources, meta_jsons, strict=False):
            if src not in result:
                with contextlib.suppress(Exception):
                    meta = json.loads(mjson or "{}")
                    result[str(src)] = meta.get("content_hash", "")
                if str(src) not in result:
                    result[str(src)] = ""
        return result
