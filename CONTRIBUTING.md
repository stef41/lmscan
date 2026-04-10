# Contributing to lmscan

Thanks for your interest in contributing! lmscan is an open-source AI text detection tool and we welcome contributions of all kinds.

## Quick Start

```bash
git clone https://github.com/stef41/lmscan.git
cd lmscan
pip install -e .
python -m pytest tests/ -q
```

## What We're Looking For

- **New model fingerprints** — Add vocabulary markers and phrases for LLMs we don't cover yet
- **Multilingual support** — Slop word dictionaries for non-English languages
- **Better calibration data** — Labeled datasets of AI vs human text
- **Bug reports** — Especially false positives/negatives on real-world text
- **Documentation** — Usage examples, tutorials, comparisons

## Adding a Model Fingerprint

The easiest way to contribute — edit `src/lmscan/fingerprint.py` and add a new `_ModelProfile`:

```python
"Model Name": _ModelProfile(
    vocabulary=["word1", "word2", ...],
    phrases=["characteristic phrase", ...],
    hedging=["hedging pattern", ...],
),
```

## Running Tests

```bash
pip install -e . pytest
python -m pytest tests/ -q
```

All 150 tests must pass. Add tests for any new functionality.

## Code Style

- We use `ruff` for linting
- Type hints everywhere
- `from __future__ import annotations` at the top of every file
- Zero external dependencies (stdlib only)

## Pull Request Process

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`python -m pytest tests/ -q`)
5. Run lint (`ruff check src/ tests/`)
6. Submit a PR

We aim to review PRs within 48 hours.
