# General makefile
# Conda environment name
CONDA_ENV := lema
CONDA_ACTIVE := $(shell conda info --envs | grep -q "*" && echo "true" || echo "false")
CONDA_RUN := conda run -n $(CONDA_ENV)

# Source directory
SRC_DIR := .
TEST_DIR := tests
DOCS_DIR := docs/.sphinx

# Sphinx documentation variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = $(DOCS_DIR)
DOCS_BUILDDIR      = $(DOCS_DIR)/_build

# Default target
ARGS :=
USERNAME := $(shell whoami)
.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  setup       - Set up the project (create conda env if not exists, install dependencies)"
	@echo "  upgrade     - Upgrade project dependencies"
	@echo "  clean       - Remove generated files and directories"
	@echo "  check       - Run pre-commit hooks"
	@echo "  torchfix    - Run TorchFix static analysis"
	@echo "  format      - Run code formatter"
	@echo "  test        - Run tests"
	@echo "  train       - Run training"
	@echo "  evaluate    - Run evaluation"
	@echo "  infer       - Run inference"
	@echo "  skyssh      - Launch a cloud VM with SSH config"
	@echo "  skycode     - Launch a vscode remote session on a cloud VM"
	@echo "  docs        - Build Sphinx documentation"
	@echo "  docs-help   - Show Sphinx documentation help"
	@echo "  docs-serve  - Serve docs locally and open in browser"

setup:
	@if conda env list | grep -q $(CONDA_ENV); then \
		echo "Conda environment '$(CONDA_ENV)' already exists. Skipping creation."; \
	else \
		conda create -n $(CONDA_ENV) python=3.11 -y; \
		$(CONDA_RUN) pip install -e ".[dev,train]"; \
		$(CONDA_RUN) pre-commit install; \
	fi

upgrade:
	$(CONDA_RUN) pip install --upgrade -e ".[dev,train]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf $(DOCS_BUILDDIR)

check:
	$(CONDA_RUN) pre-commit run --all-files

torchfix:
	$(CONDA_RUN) torchfix --select ALL .

format:
	$(CONDA_RUN) ruff format $(SRC_DIR) $(TEST_DIR)

test:
	$(CONDA_RUN) pytest $(TEST_DIR)

train:
	$(CONDA_RUN) python -m lema.train $(ARGS)

evaluate:
	$(CONDA_RUN) python -m lema.evaluate $(ARGS)

infer:
	$(CONDA_RUN) python -m lema.infer $(ARGS)

skyssh:
	$(CONDA_RUN) sky launch $(ARGS) -y --no-setup -c "${USERNAME}-dev" --cloud gcp configs/skypilot/sky_ssh.yaml
	ssh "${USERNAME}-dev"

skycode:
	$(CONDA_RUN) sky launch $(ARGS) -y --no-setup -c "${USERNAME}-dev" --cloud gcp configs/skypilot/sky_ssh.yaml
	code --new-window --folder-uri=vscode-remote://ssh-remote+"${USERNAME}-dev/home/gcpuser/sky_workdir/"

docs:
	$(CONDA_RUN) $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(DOCS_BUILDDIR)" $(SPHINXOPTS) $(O)

docs-help:
	$(CONDA_RUN) $(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(DOCS_BUILDDIR)" $(SPHINXOPTS) $(O)

docs-serve: docs
	@echo "Serving documentation at http://localhost:8000"
	@$(CONDA_RUN) python -c "import webbrowser; webbrowser.open('http://localhost:8000')" &
	@$(CONDA_RUN) python -m http.server 8000 --directory $(DOCS_BUILDDIR)/html

.PHONY: help setup upgrade clean check format test train evaluate infer skyssh skycode docs docs-help docs-serve
