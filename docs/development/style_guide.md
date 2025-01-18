# Style Guide

## Style and Conventions

### Style Guide

Oumi follows Google's [Python Style Guide](https://google.github.io/styleguide/pyguide.html)
for how to format and structure code.

Opt for using descriptive-style verbs (ex. "Builds") over imperative-style (ex. "Build")
for docstrings; see
[style guide section 3.8.3](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
for more details.

### Pre-Commit Hooks

Oumi uses [Pre Commit](https://pre-commit.com/) to enforce style checks. To configure,
run either `make setup`, or:

```shell
pip install '.[dev]'
pre-commit install
```

The pre-commit hooks will now be run before each commit. You can also run the hooks manually via:

```shell
pre-commit run  # run all hooks on changed files
pre-commit run --all-files  # or, run all hooks on all files
```

### Code Formatting

Oumi uses the [ruff](https://github.com/astral-sh/ruff) formatter for code formatting.
These checks run through pre-commit (see section 1.2). These checks can also be
run manually via:

```shell
pre-commit run ruff --all-files
```

The configuration is stored in [pyproject.toml](https://github.com/oumi-ai/oumi/blob/main/pyproject.toml) and
[.pre-commit-config.yaml](https://github.com/oumi-ai/oumi/blob/main/.pre-commit-config.yaml).

## Type Annotations and Static Type Checking

Oumi aims to annotate all functions with type annotations (introduced in
[PEP 526](https://www.python.org/dev/peps/pep-0526/)).

To validate type annotations using [PyRight](https://github.com/microsoft/pyright), run one of these commands:

```shell
pre-commit run pyright
pre-commit run pyright --all-files
```

## Imports and `__init__.py`

All imports in Oumi should be absolute.

## Documentation

We use Sphinx for documentation. To build and serve the documentation, use the following commands:

```shell
make docs-rebuild # cleanup everything, rebuild apidocs, rebuild html
make docs-serve   # serve existing html
```

Documentation is generated from the docstrings in the code. We use Google-style
docstrings for the Python code and [reStructuredText](https://docutils.sourceforge.io/rst.html) for the rest.

For an example of the available directives in docstrings, see the
[Google Sphinx Style Guide](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google).

For an overview of the available sphinx-specific directives, see the
[Sphinx documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).
