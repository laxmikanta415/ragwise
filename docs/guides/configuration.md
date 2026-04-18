# Configuration Reference

ragwise uses Pydantic models for all configuration. Typos raise `ValueError` at construction time — not at first query.

## RAGConfig

```python
from ragwise import RAGConfig

# From environment variables
config = RAGConfig.from_env()

# From YAML file
config = RAGConfig.from_yaml("./ragwise_config.yaml")

# Inline
config = RAGConfig(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
    store=StoreConfig(backend="lance", path="./ragwise-index"),
)
```

### Environment Variables

| Env Var | RAGConfig field | Example |
|---|---|---|
| `RAGWISE_LLM_MODEL` | `llm.model` | `openai/gpt-4o-mini` |
| `RAGWISE_LLM_API_KEY` | `llm.api_key` | `sk-...` |
| `RAGWISE_STORE_BACKEND` | `store.backend` | `lance`, `postgresql`, `memory` |
| `RAGWISE_STORE_PATH` | `store.path` | `./ragwise-index` |
| `RAGWISE_EMBEDDER` | `embedder.model` | `openai/text-embedding-3-small` |
| `RAGWISE_CACHE_REDIS_URL` | `cache.redis_url` | `redis://localhost:6379` |

## RAG Constructor

```python
from ragwise import RAG

async with RAG(
    llm="openai/gpt-4o-mini",         # or RAGConfig object
    store="memory",                    # memory | lance://path | postgresql://...
    embedder="openai/text-embedding-3-small",
    reranker="flashrank",              # flashrank | cohere/rerank-4 | None
    chunk_size=512,
    chunk_overlap=64,
    cache=True,
    cache_threshold=0.92,
    confidence_threshold=0.7,
) as rag:
    ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm` | `str \| LLMConfig` | required | LLM provider/model string or config |
| `store` | `str \| StoreConfig` | `"memory"` | Vector store backend |
| `embedder` | `str \| EmbedderConfig` | `"openai/text-embedding-3-small"` | Embedding model |
| `reranker` | `str \| None` | `None` | Reranker: `"flashrank"` or `"cohere/rerank-4"` |
| `chunk_size` | `int` | `512` | Target chunk size in tokens |
| `chunk_overlap` | `int` | `64` | Overlap between consecutive chunks |
| `cache` | `bool` | `False` | Enable semantic query cache |
| `cache_threshold` | `float` | `0.92` | Cosine similarity threshold for cache hits |
| `confidence_threshold` | `float \| None` | `None` | Min centroid-distance score; skips LLM if below |

## QueryConfig

```python
from ragwise import QueryConfig

answer = await rag.query(
    "What is the refund policy?",
    config=QueryConfig(
        top_k=5,
        tenant_id="org_a",
        allowed_sources=["docs/policies/*.md"],
        as_of="2024-06-15",
        n_queries=3,
        include_citations=True,
    ),
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `top_k` | `int` | `4` | Number of chunks to retrieve |
| `tenant_id` | `str \| None` | `None` | Scopes retrieval to this tenant |
| `allowed_sources` | `list[str] \| None` | `None` | Source glob patterns to restrict retrieval |
| `as_of` | `str \| None` | `None` | ISO date; filters to chunks valid on this date |
| `n_queries` | `int` | `1` | Number of query variants (RAG-Fusion) |
| `include_citations` | `bool` | `True` | Whether to populate `answer.citations` |
| `cache_threshold` | `float \| None` | `None` | Override instance-level cache threshold |

## Ingest Options

```python
result = await rag.ingest(
    "./docs/",
    glob="**/*.md",
    tenant_id="org_a",
    metadata={"valid_from": "2024-01-01", "valid_until": "2024-12-31"},
    force=False,          # True to bypass incremental hash check
    on_progress=callback, # called per file: fn(file_path, status)
)
# result: IngestResult(chunks_created=42, skipped=3, failed_files=[])
```

## LLM Provider Strings

| String | Provider | Notes |
|---|---|---|
| `"openai/gpt-4o-mini"` | OpenAI | Requires `OPENAI_API_KEY` |
| `"openai/gpt-4o"` | OpenAI | — |
| `"anthropic/claude-haiku-4-5"` | Anthropic | Requires `ANTHROPIC_API_KEY` |
| `"anthropic/claude-opus-4-6"` | Anthropic | — |
| `"ollama/llama3"` | Ollama | Requires local Ollama server |

## Embedder Provider Strings

| String | Provider | Notes |
|---|---|---|
| `"openai/text-embedding-3-small"` | OpenAI | Default |
| `"openai/text-embedding-3-large"` | OpenAI | Higher accuracy |
| `"local/all-MiniLM-L6-v2"` | sentence-transformers | `pip install ragwise[local-emb]` |
| `"local/bge-large-en-v1.5"` | sentence-transformers | Higher accuracy, larger |
