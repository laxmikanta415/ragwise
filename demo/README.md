# ragwise Demo

Interactive terminal demo showcasing hybrid search, streaming, agent tools, and multi-tenancy.

## Run it

```bash
pip install ragwise rich
export OPENAI_API_KEY="sk-..."
cd demo/
python demo.py
```

## What it shows

1. **Hybrid search** — ingest 4 docs, query with BM25+dense RRF
2. **Streaming** — tokens arriving in real time from the LLM
3. **Agent tools** — `as_claude_tool()` schema + `rag.search()` raw results
4. **Multi-tenancy** — two tenants, full isolation, no schema changes
