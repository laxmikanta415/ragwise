# Contributing to ragx

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development setup

```bash
git clone https://github.com/laxmikanta415/ragx.git
cd ragx
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Verify your setup:

```bash
python -m pytest tests/ -x -q      # all tests pass
python -m ruff check src/           # no lint errors
python -m mypy src/ragx/            # no type errors
```

## Branch naming

| Type | Prefix | Example |
|------|--------|---------|
| New feature | `feature/` | `feature/streaming-responses` |
| Bug fix | `fix/` | `fix/lance-fts-index` |
| Documentation | `docs/` | `docs/evaluation-guide` |
| Chore / deps | `chore/` | `chore/bump-chonkie` |

All PRs target `main`.

## Commit style

Use the imperative mood. Reference the issue number if one exists.

```
add HyPE pre-indexing to HybridSearcher (#42)
fix incremental ingest skipping on Windows paths (#38)
docs: add PostgreSQL production guide
```

## Before submitting a PR

- [ ] `pytest tests/ -x -q` passes
- [ ] `ruff check src/ --fix` — no remaining violations
- [ ] `mypy src/ragx/` — no errors
- [ ] New behaviour has tests
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`
- [ ] PR description links the related issue

## Adding a new store backend

1. Subclass `VectorStore` in `src/ragx/indexing/base.py`
2. Implement all five abstract methods: `upsert`, `dense_search`, `sparse_search`, `delete`, `get_indexed_sources`
3. Guard the optional import with `try/except ImportError`
4. Add a string shorthand to `_resolve_store()` in `src/ragx/pipeline.py`
5. Add tests in `tests/test_indexing.py`
6. Document the new store in `docs/stores.md`

## Running only specific tests

```bash
pytest tests/test_pipeline.py -x -q       # pipeline only
pytest tests/ -k "test_rag_query" -v      # by name pattern
pytest tests/ --co -q                     # list collected tests
```

## Reporting bugs

Please open an issue using the [bug report template](https://github.com/laxmikanta415/ragx/issues/new?template=bug_report.yml).

## Questions

Open a [GitHub Discussion](https://github.com/laxmikanta415/ragx/discussions) in the Q&A category.
