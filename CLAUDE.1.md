# Oumi Development Guide

## Build/Lint/Test Commands

- Setup: `make setup` (creates conda env, installs dependencies)
- Run all tests: `make test`
- Run a single test: `pytest tests/path/to/test.py::test_function_name`
- Format code: `make format` (ruff formatter)

## Design Principles

- Minimal Abstractions: Reduce complexity, avoid over-engineering
- Explicit Over Implicit: Clear, readable code over clever tricks
- Testable Design: Write code with testing in mind

## Code Style

- Follow Google Python Style Guide
- Use absolute imports only (no relative imports)
- Type annotations required for all functions
- Docstrings: Use Google-style with descriptive verbs ("Builds" not "Build")
- Use list, dict for type hints, instead of typing.List, typing.Dict
- Formatting: 88-character line limit
- Name code entities according to PEP8 conventions
- Error handling: Use specific exceptions with informative messages
- No wildcard imports (from x import *)

## Imports

Sort imports with isort (handled by pre-commit):

1. Standard library
2. Third-party packages
3. Oumi packages (oumi.*)
