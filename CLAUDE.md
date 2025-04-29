# Oumi Development Guide for Claude

## Build/Lint/Test Commands
```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run linters and type checking
pre-commit run --all-files        # Run all hooks
ruff format .                     # Format code
pyright                           # Type checking

# Run tests
pytest tests/                     # All tests
pytest tests/unit/               # Unit tests only
pytest tests/path/to/test_file.py # Single test file
pytest tests/path/to/test_file.py::TestClass::test_function  # Single test
```

## Code Style Guidelines
- Follow Google's Python Style Guide
- Use absolute imports only, no relative imports
- Use descriptive names for functions, variables, and classes
- Add type annotations for all function parameters and return values
- Handle errors with contextual information
- Format code with ruff (configured in pyproject.toml)
- Use docstrings in Google format for all public functions and classes
- Add Apache License header to all source files
- Follow commit message conventions from existing commits