"""ragwise — Production-grade RAG in 4 lines. Hybrid search on by default."""

from ragwise.agent import as_claude_tool, as_claude_tool_suite, as_openai_tool, as_openai_tool_suite
from ragwise.agent_session import AgentSession
from ragwise.config import Answer, IngestResult, QueryConfig, RAGConfig
from ragwise.eval.schema import EvalSchema
from ragwise.indexing.base import SearchResult
from ragwise.models import Citation, QueryTrace, RetrievedChunk
from ragwise.pipeline import RAG

__version__ = "0.2.0"

__all__ = [
    "RAGConfig",
    "QueryConfig",
    "Answer",
    "IngestResult",
    "EvalSchema",
    "SearchResult",
    "Citation",
    "RetrievedChunk",
    "QueryTrace",
    "RAG",
    "as_claude_tool",
    "as_openai_tool",
    "as_claude_tool_suite",
    "as_openai_tool_suite",
    "AgentSession",
]
