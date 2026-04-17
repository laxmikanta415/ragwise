# Contributing

See the full [CONTRIBUTING.md](https://github.com/laxmikanta415/ragwise/blob/main/CONTRIBUTING.md) in the repository root for complete guidelines.

## Quick reference

```bash
git clone https://github.com/laxmikanta415/ragwise.git
cd ragwise
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -x -q

# Lint + format
python -m ruff check src/ --fix

# Type check
python -m mypy src/ragwise/
```

## Branching

- `feature/your-feature` for new functionality
- `fix/issue-description` for bug fixes
- `docs/what-you-updated` for documentation changes

All PRs target `main`.

## Before submitting

- [ ] All tests pass: `pytest tests/ -x -q`
- [ ] Ruff is clean: `ruff check src/`
- [ ] Mypy is clean: `mypy src/ragwise/`
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] PR links to the related issue

## Questions?

Open a [GitHub Discussion](https://github.com/laxmikanta415/ragwise/discussions) for questions, ideas, or feedback. For bugs, open an [Issue](https://github.com/laxmikanta415/ragwise/issues).
