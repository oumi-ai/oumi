# 1. Style and Conventions

## 1.1 Style Guide

LeMa follows Google's [Python Style Guide](https://google.github.io/styleguide/pyguide.html)
for how to format and structure code.

## 1.2. Pre-Commit Hooks

LeMa uses [Pre Commit](https://pre-commit.com/) to enforce style checks. To configure, run
```
pip install '.[dev]'
pre-commit install
```

The pre-commit hooks will now be run before each commit. You can also run the hooks manually via:

```
pre-commit run  # run all hooks on changed files
pre-commit run --all-files  # or, run all hooks on all files
```


## 1.3. Code Formatting

LeMa uses the [ruff](https://github.com/astral-sh/ruff) formatter for code formatting.
These checks run through pre-commit (see section 1.2). These checks can also be
run manually via:

```
pre-commit run ruff --all-files
```

The configuration is stored in [pyproject.toml](pyproject.toml) and 
[.pre-commit-config.yaml](.pre-commit-config.yaml).


# 2. Type Annotations and Static Type Checking

TODO: Configure `pyright`, and provide guidance.

# 3. Imports and `__init__.py`

All imports in LeMa should be absolute.


# 4. Documentation

TODO: Configure `sphinx` (or similar), and provide guidance.
