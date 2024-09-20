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
	@if conda env list | grep -q $(CONDA_ENV); then \
		echo "Conda environment '$(CONDA_ENV)' already exists. Skipping creation."; \
	else \
		conda create -n $(CONDA_ENV) python=3.11 -y; \
		source ~/.bashrc 2>dev/null \
		source ~/.zshrc 2>dev/null \		
		conda activate $(CONDA_ENV); \
		pip install -e ".[all]"; \
		pre-commit install; \
	fi

upgrade:
	$(CONDA_RUN) pip install --upgrade -e ".[all]"

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
