SHELL := /bin/bash

# General makefile
# Conda environment name
CONDA_ENV := oumi
CONDA_ACTIVE := $(shell conda info --envs | grep -q "*" && echo "true" || echo "false")
CONDA_RUN := conda run -n $(CONDA_ENV)

# Source directory
SRC_DIR := .
TEST_DIR := tests
DOCS_DIR := docs/.sphinx
OUMI_SRC_DIR := src/oumi

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
	@echo "  coverage    - Run tests with coverage"
	@echo "  train       - Run training"
	@echo "  evaluate    - Run evaluation"
	@echo "  infer       - Run inference"
	@echo "  skyssh      - Launch a cloud VM with SSH config"
	@echo "  skycode     - Launch a vscode remote session on a cloud VM"
	@echo "  docs        - Build Sphinx documentation"
	@echo "  docs-help   - Show Sphinx documentation help"
	@echo "  docs-serve  - Serve docs locally and open in browser"
	@echo "  docs-rebuild  - Fully rebuild the docs: (a) Regenerate apidoc RST and (b) build html docs from source"

setup:
	@if command -v conda >/dev/null 2>&1; then \
		if conda env list | grep -q "^$(CONDA_ENV) "; then \
			echo "Conda environment '$(CONDA_ENV)' already exists. Updating dependencies..."; \
			$(CONDA_RUN) pip install -U -e ".[train,dev]"; \
		else \
			echo "Creating new conda environment '$(CONDA_ENV)'..."; \
			CONDA_BASE=$$(conda info --base); \
			source "$${CONDA_BASE}/etc/profile.d/conda.sh"; \
			conda create -n $(CONDA_ENV) python=3.11 -y; \
			$(CONDA_RUN) pip install -e ".[train,dev]"; \
			$(CONDA_RUN) pre-commit install; \
		fi; \
	else \
		echo "Error: Conda is not installed or not in PATH."; \
		echo "Please install Miniconda and initialize it for your shell:"; \
		echo "1. Download Miniconda from https://docs.conda.io/en/latest/miniconda.html"; \
		echo "2. Install Miniconda by running the downloaded script"; \
		echo "3. Initialize Conda for your shell:"; \
		echo "   - For bash: Run 'conda init bash' and restart your terminal"; \
		echo "   - For zsh: Run 'conda init zsh' and restart your terminal"; \
		echo "4. After initialization, run 'make setup' again"; \
		exit 1; \
	fi
	@echo "Setup completed successfully."

upgrade:
	$(CONDA_RUN) pip install --upgrade -e ".[train,dev]"

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

coverage:
	$(CONDA_RUN) pytest --cov=$(OUMI_SRC_DIR) --cov-report=term-missing --cov-report=html:coverage_html $(TEST_DIR)

train:
	$(CONDA_RUN) python -m oumi.train $(ARGS)

evaluate:
	$(CONDA_RUN) python -m oumi.evaluate $(ARGS)

infer:
	$(CONDA_RUN) python -m oumi.infer $(ARGS)

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

docs-rebuild:
	rm -rf $(DOCS_BUILDDIR) "$(SOURCEDIR)/apidoc"
	$(CONDA_RUN) sphinx-apidoc "$(SRC_DIR)/src/oumi" --output-dir "$(SOURCEDIR)/apidoc" --remove-old --force --module-first --implicit-namespaces  --maxdepth 2 --templatedir  "$(SOURCEDIR)/_templates/apidoc"
	$(CONDA_RUN) $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(DOCS_BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help setup upgrade clean check format test coverage train evaluate infer skyssh skycode docs docs-help docs-serve docs-rebuild
