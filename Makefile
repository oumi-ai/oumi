SHELL := /bin/bash

# General makefile
# Conda environment name
CONDA_ENV := oumi
CONDA_ACTIVE := $(shell conda info --envs | grep -q "*" && echo "true" || echo "false")
CONDA_RUN := conda run -n $(CONDA_ENV)
CONDA_INSTALL_PATH := $(HOME)/miniconda3

# Source directory
SRC_DIR := .
TEST_DIR := tests
DOCS_DIR := docs
OUMI_SRC_DIR := src/oumi

# Sphinx documentation variables
SPHINXOPTS    ?= -v
SPHINXBUILD   ?= sphinx-build
DOCS_SOURCEDIR     = $(DOCS_DIR)
DOCS_BUILDDIR      = $(DOCS_DIR)/_build

# Default target
ARGS :=
USERNAME := $(shell whoami)
.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  setup             - Set up the project (create conda env if not exists, install dependencies)"
	@echo "  install-miniconda - Install Miniconda"
	@echo "  upgrade           - Upgrade project dependencies"
	@echo "  clean             - Remove generated files and directories"
	@echo "  check             - Run pre-commit hooks"
	@echo "  torchfix          - Run TorchFix static analysis"
	@echo "  format            - Run code formatter"
	@echo "  test              - Run tests"
	@echo "  coverage          - Run tests with coverage"
	@echo "  gcpssh            - Launch a GCP VM and ssh into it"
	@echo "  gcpcode           - Launch a VS Code remote session on a GCP VM"
	@echo "  docs              - Build Sphinx documentation"
	@echo "  docs-help         - Show Sphinx documentation help"
	@echo "  docs-serve        - Serve docs locally and open in browser"
	@echo "  docs-rebuild      - Fully rebuild the docs: (a) Regenerate apidoc RST and (b) build html docs from source"
	@echo "  doctest           - Run doctests on documentation files"
	@echo "  doctest-file      - Run doctests on a specific documentation file"

# If we detect the system is an Intel Mac, print an error message and exit.
setup:
	@bash -c '\
		if [ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "x64_64" ]; then \
			echo "Cannot install Oumi on Intel Macs because PyTorch dropped support for that platform!"; \
			exit 1; \
		fi; \
		if command -v conda >/dev/null 2>&1; then \
			if conda env list | grep -q "^$(CONDA_ENV) "; then \
				echo "Conda environment $(CONDA_ENV) already exists. Updating dependencies..."; \
				$(CONDA_RUN) pip install -U uv; \
				$(CONDA_RUN) uv pip install -U -e ".[dev]"; \
			else \
				echo "Creating new conda environment $(CONDA_ENV)..."; \
				conda create -n $(CONDA_ENV) python=3.11 -y; \
				$(CONDA_RUN) pip install uv; \
				$(CONDA_RUN) uv pip install -e ".[dev]"; \
				$(CONDA_RUN) pre-commit install; \
				$(CONDA_RUN) python -m ipykernel install --user --name $(CONDA_ENV); \
			fi; \
		else \
			echo "Error: Conda is not installed or not in PATH."; \
			echo "Please run `make install-miniconda` to install Miniconda, then run `make setup` again."; \
			exit 1; \
		fi; \
		echo "Installation complete. Testing if oumi package can be imported..."; \
		if $(CONDA_RUN) python -c "import oumi" >/dev/null 2>&1; then \
			echo "oumi package imported successfully!"; \
			echo "To start using your new environment, run: \"conda activate $(CONDA_ENV)\"."; \
			echo "Happy fine-tuning!"; \
		else \
			echo "Error: Failed to import oumi package. Please check your installation."; \
			echo "Please open an issue at https://github.com/oumi-ai/oumi/issues if the problem persists."; \
			exit 1; \
		fi'

install-miniconda:
	@bash -c '\
		echo "1. Configuring miniconda installer..."; \
		if [ "$$(uname)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
			MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"; \
		elif [ "$$(uname)" = "Darwin" ]; then \
			MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"; \
		elif [ "$$(uname)" = "Linux" ]; then \
			MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
		else \
			echo "Unsupported operating system. Please install Miniconda manually."; \
			echo "1. Download Miniconda from https://docs.conda.io/en/latest/miniconda.html"; \
			echo "2. Run the installer and follow the instructions."; \
			exit 1; \
		fi; \
		CONDA_INSTALL_PATH="$(CONDA_INSTALL_PATH)"; \
		echo "- Installer URL: $$MINICONDA_URL"; \
		echo "- Conda will be installed to: $$CONDA_INSTALL_PATH"; \
		echo "2. Downloading Miniconda installer..."; \
		curl -o miniconda.sh $$MINICONDA_URL; \
		echo "3. Installing Miniconda..."; \
		bash miniconda.sh -u -b -p $$CONDA_INSTALL_PATH; \
		rm miniconda.sh; \
		echo "4. Initializing Conda..."; \
		$$CONDA_INSTALL_PATH/bin/conda init $$(basename $$SHELL); \
		echo "Conda has been initialized for $$(basename $$SHELL) shell."; \
		echo "Please restart your terminal for this to take effect."'

upgrade:
	@if $(CONDA_RUN) command -v uv >/dev/null 2>&1; then \
		$(CONDA_RUN) uv pip install --upgrade -e ".[dev]"; \
	else \
		echo "uv is not installed, using pip instead."; \
		echo "To install uv, run: 'pip install uv'"; \
		$(CONDA_RUN) pip install --upgrade -e ".[dev]"; \
	fi

clean: docs-clean
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

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

# To adjust the accelerators: `make gcpcode ARGS="--resources.accelerators A100:4"`
# To run on a different cloud: `make gcpssh ARGS="--resources.cloud aws"`
gcpssh:
	$(CONDA_RUN) oumi launch up $(ARGS) --cluster "${USERNAME}-dev" -c configs/examples/misc/dev_gcp_job.yaml
	ssh "${USERNAME}-dev"

# To adjust the accelerators: `make gcpcode ARGS="--resources.accelerators A100:4"`
# To run on a different cloud: `make gcpcode ARGS="--resources.cloud aws"`
gcpcode:
	$(CONDA_RUN) oumi launch up $(ARGS) --cluster "${USERNAME}-dev" -c configs/examples/misc/dev_gcp_job.yaml
	code --new-window --folder-uri=vscode-remote://ssh-remote+"${USERNAME}-dev/home/gcpuser/sky_workdir/"

docs:
	$(CONDA_RUN) $(SPHINXBUILD) -M html "$(DOCS_SOURCEDIR)" "$(DOCS_BUILDDIR)" $(SPHINXOPTS) $(O)

docs-rebuild: docs-clean docs-copy-files docs-update-summaries
	$(CONDA_RUN) sphinx-apidoc "$(SRC_DIR)/src/oumi" --output-dir "$(DOCS_SOURCEDIR)/api" --remove-old --force --module-first --implicit-namespaces  --maxdepth 2 --templatedir  "$(DOCS_SOURCEDIR)/_templates/apidoc"
	$(CONDA_RUN) $(SPHINXBUILD) -M html "$(DOCS_SOURCEDIR)" "$(DOCS_BUILDDIR)" $(SPHINXOPTS) $(O)

docs-help:
	$(CONDA_RUN) $(SPHINXBUILD) -M help "$(DOCS_SOURCEDIR)" "$(DOCS_BUILDDIR)" $(SPHINXOPTS) $(O)

docs-serve: docs
	@echo "Serving documentation at http://localhost:8000"
	@$(CONDA_RUN) python -c "import webbrowser; webbrowser.open('http://localhost:8000')" &
	@$(CONDA_RUN) python -m http.server 8000 --directory $(DOCS_BUILDDIR)/html

docs-copy-files:
	$(CONDA_RUN) python $(DOCS_SOURCEDIR)/_manage_doclinks.py copy "$(DOCS_SOURCEDIR)/_doclinks.config"

docs-update-summaries:
	$(CONDA_RUN) bash $(DOCS_SOURCEDIR)/_docsummaries.sh

docs-clean:
	rm -rf $(DOCS_BUILDDIR) "$(DOCS_SOURCEDIR)/api"
	$(CONDA_RUN) python $(DOCS_SOURCEDIR)/_manage_doclinks.py clean "$(DOCS_SOURCEDIR)/_doclinks.config"

doctest:
	$(CONDA_RUN) $(SPHINXBUILD) -b doctest "$(DOCS_SOURCEDIR)" "$(DOCS_BUILDDIR)"

doctest-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Please specify a file using FILE=docs/path/to/file"; \
		exit 1; \
	fi
	$(CONDA_RUN) $(SPHINXBUILD) -b doctest "$(DOCS_SOURCEDIR)" "$(DOCS_BUILDDIR)" $(FILE)

.PHONY: help setup upgrade clean check format test coverage gcpssh gcpcode docs docs-help docs-serve docs-rebuild copy-doc-files clean-docs doctest doctest-file
