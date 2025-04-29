# Oumi Development Guide

## Build/Lint/Test Commands
- Setup: `make setup` (creates conda env, installs dependencies)
- Run all tests: `make test` 
- Run a single test: `pytest tests/path/to/test.py::test_function_name`
- Lint and check: `make check` (runs pre-commit hooks)
- Format code: `make format` (ruff formatter)
- Type check: `pre-commit run pyright` or `pre-commit run pyright --all-files`


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

## CLI Argument Style
- CLI commands support both shorthand and longform arguments
- Shorthand arguments are expanded to their fully qualified equivalents
- Both forms can be used interchangeably, but are mutually exclusive for the same parameter
- Examples:
  - Shorthand: `--model llama3` expands to `--model.model_name llama3`
  - Shorthand: `--dataset alpaca` expands to `--data.train.datasets[0].dataset_name alpaca`
- To add new shorthand arguments, update the `SHORTHAND_MAPPINGS` dictionary in `cli_utils.py`

## CLI Styling and Logging
The CLI now supports improved visual presentation with the following features:
- Enhanced output with colors, animations, and formatting
- Two styling modes: 'full' (default) and 'none' (plain text)
- Control styling via `oumi env --styling <mode>` or environment variables
- Automatic terminal capability detection
- Spinners for long-running operations
- Success/error indicators for operations
- Fancy ASCII art logo for terminals that support it
- Colorized and enhanced logging with better readability
- Rich error traceback formatting

Styling can be controlled via:
- Command: `oumi env --styling full` or `oumi env --styling none`
- Environment variables:
  - `OUMI_NO_STYLE=1` - Disable all styling
  - `OUMI_STYLE_LEVEL=none|full` - Set specific styling level

You can test the log styling with:
- Command: `oumi env --test-logs`

## Imports
Sort imports with isort (handled by pre-commit):
1. Standard library
2. Third-party packages
3. Oumi packages (oumi.*)