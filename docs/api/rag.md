# RAG

The `RAG` class is the single entry point for all ragwise functionality. It is an async context manager.

## Usage

```python
from ragwise import RAG

async with RAG(llm="openai/gpt-4o-mini") as rag:
    await rag.ingest("./docs/")
    answer = await rag.query("What is the refund policy?")
```

## Reference

::: ragwise.pipeline.RAG
    options:
      show_source: true
      members_order: source
