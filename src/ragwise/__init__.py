"""ragx — Production-grade RAG in 4 lines. Hybrid search on by default."""

from ragwise.agent import as_claude_tool, as_openai_tool
from ragwise.config import Answer, IngestResult, QueryConfig, RAGConfig
from ragwise.eval.schema import EvalSchema
from ragwise.indexing.base import SearchResult
from ragwise.pipeline import RAG

__version__ = "0.1.0"

__all__ = [
    "RAGConfig",
    "QueryConfig",
    "Answer",
    "IngestResult",
    "EvalSchema",
    "SearchResult",
    "RAG",
    "as_claude_tool",
    "as_openai_tool",
]
