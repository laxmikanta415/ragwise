"""S1-T5: Ingestion progress callbacks and failed_files in IngestResult."""
from __future__ import annotations

import pytest

from ragwise import RAG, IngestResult


@pytest.mark.asyncio
async def test_ingest_progress_callback(tmp_path, mock_embedder, mock_llm) -> None:
    (tmp_path / "a.txt").write_text("Document A.")
    (tmp_path / "b.txt").write_text("Document B.")
    (tmp_path / "c.txt").write_text("Document C.")

    progress_calls: list[tuple[str, int, int]] = []

    def on_progress(current_file: str, files_done: int, total_files: int) -> None:
        progress_calls.append((current_file, files_done, total_files))

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        await rag.ingest(str(tmp_path), on_progress=on_progress)

    assert len(progress_calls) == 3
    # total_files is always 3
    assert all(t == 3 for _, _, t in progress_calls)
    # files_done increments
    done_values = sorted(d for _, d, _ in progress_calls)
    assert done_values == [1, 2, 3]


@pytest.mark.asyncio
async def test_ingest_result_has_failed_files(tmp_path, mock_embedder, mock_llm) -> None:
    (tmp_path / "good.txt").write_text("Valid document.")
    # .xyz is unsupported — will fail
    (tmp_path / "bad.xyz").write_text("Unsupported format.")

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        result = await rag.ingest(str(tmp_path))

    assert result.failed == 1
    assert len(result.failed_files) == 1
    assert "bad.xyz" in result.failed_files[0]
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_ingest_failed_file_continues(tmp_path, mock_embedder, mock_llm) -> None:
    (tmp_path / "a.txt").write_text("Good A.")
    (tmp_path / "b.xyz").write_text("Bad file.")
    (tmp_path / "c.txt").write_text("Good C.")

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        result = await rag.ingest(str(tmp_path))

    assert result.succeeded == 2
    assert result.failed == 1
    assert len(result.failed_files) == 1


@pytest.mark.asyncio
async def test_ingest_result_failed_files_default_empty(tmp_path, mock_embedder, mock_llm) -> None:
    (tmp_path / "doc.txt").write_text("All good.")

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        result = await rag.ingest(str(tmp_path))

    assert result.failed_files == []
    assert result.failed == 0


def test_ingest_result_dataclass_fields() -> None:
    r = IngestResult(succeeded=3, failed=1, errors=["x: err"], failed_files=["x.xyz"])
    assert r.failed_files == ["x.xyz"]


def test_ingest_result_failed_files_defaults_empty() -> None:
    r = IngestResult(succeeded=0, failed=0)
    assert r.failed_files == []


@pytest.mark.asyncio
async def test_ingest_metadata_stored_on_chunks(tmp_path, mock_embedder, mock_llm) -> None:
    from ragwise.indexing.memory import InMemoryStore

    (tmp_path / "doc.txt").write_text("Policy document.")

    async with RAG(llm=mock_llm, embedder=mock_embedder) as rag:
        assert isinstance(rag._store, InMemoryStore)
        await rag.ingest(str(tmp_path), metadata={"valid_from": "2024-01-01", "version": "v2"})

        chunks = rag._store._docs
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.get("valid_from") == "2024-01-01"
            assert chunk.metadata.get("version") == "v2"
