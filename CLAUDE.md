# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TranslateGemma UI — a local translation tool powered by the TranslateGemma 4B model. Provides a Gradio-based browser GUI for text translation and SRT subtitle translation with glossary support.

## Commands

```bash
# Run the app
uv run python main.py

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_smoke.py::test_import

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --group dev <package>
```

## Architecture

- **Entry point**: `main.py` — launches the Gradio app
- **Package**: `src/translate_gemma_ui/` — src layout, installed as editable via uv
- **Tests**: `tests/` — pytest, configured in `pyproject.toml`
- **Spec**: `SPEC.md` — detailed product specification (in Traditional Chinese)

Key dependencies: Gradio (UI), PyTorch (inference), Transformers (model loading).

## Conventions

- Python >= 3.12, use modern syntax (type unions with `|`, etc.)
- Ruff for linting and formatting (line-length 120, rules: E, F, I, UP)
- Package manager: uv (not pip directly)
- Conventional commits in commit messages
