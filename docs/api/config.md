# Config

Configuration dataclasses for ragwise. All are frozen and typed.

## RAGConfig

Top-level configuration — passed implicitly through `RAG(...)` constructor parameters.

::: ragwise.config.RAGConfig
    options:
      show_source: true

## QueryConfig

Per-query options passed to `rag.query(..., config=QueryConfig(...))`.

::: ragwise.config.QueryConfig
    options:
      show_source: true

## Answer

Returned by `rag.query()`.

::: ragwise.config.Answer
    options:
      show_source: true

## IngestResult

Returned by `rag.ingest()`. Never raises — failures are captured in `errors`.

::: ragwise.config.IngestResult
    options:
      show_source: true
