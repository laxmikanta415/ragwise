"""Document loaders — LoaderProtocol, AutoLoader, and concrete loader stubs."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol

from ragwise.ingestion.document import Document


class LoaderProtocol(Protocol):
    def load(self, path: Path) -> list[Document]:
        ...


class LoadResult(NamedTuple):
    docs: list[Document]
    errors: list[str]


def _read_text_safe(path: Path) -> str:
    """Read file as UTF-8; fall back to replacement chars on decode errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


class TextLoader:
    """Loads plain-text files."""

    def load(self, path: Path) -> list[Document]:
        text = _read_text_safe(path)
        return [Document(text=text, source=str(path), metadata={"extension": path.suffix.lower()})]


class MarkdownLoader:
    """Loads Markdown files with heading extraction."""

    def load(self, path: Path) -> list[Document]:
        text = _read_text_safe(path)
        headings = [
            line.lstrip("#").strip()
            for line in text.splitlines()
            if line.startswith("#")
        ]
        return [
            Document(
                text=text,
                source=str(path),
                metadata={"extension": path.suffix.lower(), "headings": headings},
            )
        ]


class PDFLoader:
    """Loads PDF files, one Document per non-empty page."""

    def load(self, path: Path) -> list[Document]:
        from pypdf import PdfReader  # already a core dep

        try:
            reader = PdfReader(str(path), strict=False)
        except Exception as exc:
            raise ValueError(f"{path}: failed to open PDF — {exc}") from exc

        if reader.is_encrypted:
            raise ValueError(f"{path}: PDF is encrypted")

        total = len(reader.pages)
        docs: list[Document] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            docs.append(
                Document(
                    text=text,
                    source=str(path),
                    metadata={
                        "extension": ".pdf",
                        "page": i,
                        "total_pages": total,
                    },
                )
            )
        return docs


class HTMLLoader:
    """Loads HTML files, extracting plain text and title metadata."""

    def load(self, path: Path) -> list[Document]:
        from bs4 import BeautifulSoup

        content = _read_text_safe(path)
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string
        ext = path.suffix.lower()
        return [
            Document(
                text=text,
                source=str(path),
                metadata={"extension": ext, "title": title},
            )
        ]


# Extension → loader class (case-insensitive matching applied at dispatch time)
_EXT_MAP: dict[str, type[TextLoader | MarkdownLoader | PDFLoader | HTMLLoader]] = {
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".pdf": PDFLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
}


class AutoLoader:
    """Dispatches document loading by file extension."""

    def load(self, path: Path) -> list[Document]:
        """Load a single file using the appropriate loader.

        Raises:
            ValueError: if the extension is not supported.
            NotImplementedError: delegated from stub loaders not yet implemented.
        """
        ext = path.suffix.lower()
        loader_cls = _EXT_MAP.get(ext)
        if loader_cls is None:
            raise ValueError(f"Unsupported file extension: {ext!r} (path: {path})")
        return loader_cls().load(path)

    def load_dir(self, path: Path, glob: str = "**/*") -> LoadResult:
        """Load all matching files under *path*, collecting per-file errors.

        Skips directories. Per-file exceptions are caught and appended to
        LoadResult.errors; successful docs are collected in LoadResult.docs.
        """
        docs: list[Document] = []
        errors: list[str] = []

        for file_path in path.glob(glob):
            if not file_path.is_file():
                continue
            try:
                docs.extend(self.load(file_path))
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")

        return LoadResult(docs=docs, errors=errors)
