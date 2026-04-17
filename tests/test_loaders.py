"""Tests for LoaderProtocol, AutoLoader, LoadResult, TextLoader, MarkdownLoader — S2-T1/T2."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragwise.ingestion import AutoLoader, LoadResult, LoaderProtocol
from ragwise.ingestion.loader import MarkdownLoader, TextLoader, _EXT_MAP


# ---------------------------------------------------------------------------
# S2-T1 — AutoLoader dispatch and LoadResult
# ---------------------------------------------------------------------------

def test_autoloader_unknown_extension_raises_valueerror(tmp_path: Path) -> None:
    f = tmp_path / "file.xyz"
    f.write_text("content")
    with pytest.raises(ValueError, match=r"\.xyz"):
        AutoLoader().load(f)


def test_autoloader_txt_dispatches_successfully(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello")
    docs = AutoLoader().load(f)
    assert len(docs) == 1


def test_autoloader_md_dispatches_successfully(tmp_path: Path) -> None:
    f = tmp_path / "test.md"
    f.write_text("# heading")
    docs = AutoLoader().load(f)
    assert len(docs) == 1


def test_autoloader_pdf_raises_valueerror_on_corrupt(tmp_path: Path) -> None:
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError):
        AutoLoader().load(f)


def test_autoloader_extension_map_has_expected_keys() -> None:
    assert ".txt" in _EXT_MAP
    assert ".md" in _EXT_MAP
    assert ".pdf" in _EXT_MAP
    assert ".html" in _EXT_MAP
    assert ".htm" in _EXT_MAP


def test_autoloader_case_insensitive_extension(tmp_path: Path) -> None:
    f = tmp_path / "test.TXT"
    f.write_text("hello")
    docs = AutoLoader().load(f)
    assert len(docs) == 1


def test_load_dir_returns_load_result(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    result = AutoLoader().load_dir(tmp_path)
    assert isinstance(result, LoadResult)
    assert isinstance(result.docs, list)
    assert isinstance(result.errors, list)


def test_load_dir_skips_directories(tmp_path: Path) -> None:
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    result = AutoLoader().load_dir(tmp_path)
    assert isinstance(result, LoadResult)


def test_load_dir_txt_succeeds(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    result = AutoLoader().load_dir(tmp_path)
    assert len(result.docs) == 2
    assert result.errors == []


def test_load_dir_unknown_extension_captured_in_errors(tmp_path: Path) -> None:
    (tmp_path / "file.xyz").write_text("data")
    result = AutoLoader().load_dir(tmp_path)
    assert len(result.errors) == 1
    assert ".xyz" in result.errors[0]


def test_loader_protocol_structural_compatibility() -> None:
    loader: LoaderProtocol = AutoLoader()  # type: ignore[assignment]
    assert callable(loader.load)


# ---------------------------------------------------------------------------
# S2-T2 — TextLoader and MarkdownLoader
# ---------------------------------------------------------------------------

def test_text_loader_returns_one_document(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    docs = TextLoader().load(f)
    assert len(docs) == 1
    assert docs[0].text == "hello world"
    assert docs[0].source == str(f)
    assert docs[0].metadata["extension"] == ".txt"


def test_text_loader_unicode_decode_fallback(tmp_path: Path) -> None:
    f = tmp_path / "latin.txt"
    f.write_bytes(b"caf\xe9")  # latin-1 encoded, not valid utf-8
    docs = TextLoader().load(f)
    assert len(docs) == 1
    assert isinstance(docs[0].text, str)  # no exception raised


def test_markdown_loader_extracts_headings(tmp_path: Path) -> None:
    f = tmp_path / "guide.md"
    f.write_text("# Title\n\nsome text\n\n## Section\n\nmore text")
    docs = MarkdownLoader().load(f)
    assert len(docs) == 1
    assert docs[0].metadata["headings"] == ["Title", "Section"]


def test_markdown_loader_no_headings(tmp_path: Path) -> None:
    f = tmp_path / "plain.md"
    f.write_text("just some plain text\nno headings here")
    docs = MarkdownLoader().load(f)
    assert docs[0].metadata["headings"] == []


def test_markdown_loader_source_and_extension(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Hello")
    docs = MarkdownLoader().load(f)
    assert docs[0].source == str(f)
    assert docs[0].metadata["extension"] == ".md"


def test_markdown_loader_unicode_decode_fallback(tmp_path: Path) -> None:
    f = tmp_path / "bad.md"
    f.write_bytes(b"# Title\ncaf\xe9")
    docs = MarkdownLoader().load(f)
    assert len(docs) == 1
    assert "Title" in docs[0].metadata["headings"]


# ---------------------------------------------------------------------------
# S2-T3 — PDFLoader
# ---------------------------------------------------------------------------

# Minimal valid PDF with one page of extractable text
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
    b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 44>>\nstream\n"
    b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
    b"endstream\nendobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000274 00000 n \n"
    b"0000000369 00000 n \n"
    b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n441\n%%EOF\n"
)


def test_pdf_loader_returns_one_doc_per_page(tmp_path: Path) -> None:
    from ragwise.ingestion.loader import PDFLoader

    f = tmp_path / "test.pdf"
    f.write_bytes(_MINIMAL_PDF)
    docs = PDFLoader().load(f)
    assert len(docs) == 1
    assert "Hello World" in docs[0].text


def test_pdf_loader_page_metadata_starts_at_1(tmp_path: Path) -> None:
    from ragwise.ingestion.loader import PDFLoader

    f = tmp_path / "test.pdf"
    f.write_bytes(_MINIMAL_PDF)
    docs = PDFLoader().load(f)
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["total_pages"] == 1
    assert docs[0].metadata["extension"] == ".pdf"
    assert docs[0].source == str(f)


def test_pdf_loader_corrupt_raises_valueerror(tmp_path: Path) -> None:
    from ragwise.ingestion.loader import PDFLoader

    f = tmp_path / "bad.pdf"
    f.write_bytes(b"not a pdf at all")
    with pytest.raises(ValueError, match=str(f.name)):
        PDFLoader().load(f)


def test_pdf_loader_empty_page_skipped(tmp_path: Path) -> None:
    from pypdf import PdfWriter
    from ragwise.ingestion.loader import PDFLoader

    f = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(f, "wb") as fh:
        writer.write(fh)
    docs = PDFLoader().load(f)
    assert docs == []  # blank page has no text


def test_autoloader_pdf_dispatches_correctly(tmp_path: Path) -> None:
    f = tmp_path / "test.pdf"
    f.write_bytes(_MINIMAL_PDF)
    docs = AutoLoader().load(f)
    assert len(docs) == 1


# ---------------------------------------------------------------------------
# S2-T4 — HTMLLoader
# ---------------------------------------------------------------------------

def test_html_loader_strips_tags(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>Hello world</p></body></html>")
    docs = AutoLoader().load(f)
    assert len(docs) == 1
    assert "<" not in docs[0].text
    assert ">" not in docs[0].text
    assert "Hello world" in docs[0].text


def test_html_loader_extracts_title(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_text("<html><head><title>My Page</title></head><body>content</body></html>")
    from ragwise.ingestion.loader import HTMLLoader
    docs = HTMLLoader().load(f)
    assert docs[0].metadata["title"] == "My Page"


def test_html_loader_no_title(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>No title here</p></body></html>")
    from ragwise.ingestion.loader import HTMLLoader
    docs = HTMLLoader().load(f)
    assert docs[0].metadata["title"] == ""


def test_html_loader_htm_extension(tmp_path: Path) -> None:
    f = tmp_path / "page.htm"
    f.write_text("<html><body>content</body></html>")
    docs = AutoLoader().load(f)
    assert len(docs) == 1
    assert docs[0].metadata["extension"] == ".htm"


def test_html_loader_source_and_extension(tmp_path: Path) -> None:
    f = tmp_path / "doc.html"
    f.write_text("<html><body>text</body></html>")
    from ragwise.ingestion.loader import HTMLLoader
    docs = HTMLLoader().load(f)
    assert docs[0].source == str(f)
    assert docs[0].metadata["extension"] == ".html"
