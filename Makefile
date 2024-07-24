# Conda environment name
CONDA_ENV := lema
CONDA_ACTIVE := $(shell conda info --envs | grep -q "*" && echo "true" || echo "false")
CONDA_RUN := conda run -n $(CONDA_ENV)

# Source directory
SRC_DIR := .
TEST_DIR := tests

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
	@echo "  format      - Run code formatter"
	@echo "  test        - Run tests"
	@echo "  train       - Run training"
	@echo "  evaluate    - Run evaluation"
	@echo "  infer       - Run inference"
	@echo "  skyssh      - Launch a cloud VM with SSH config"
	@echo "  skyssh      - Launch a vscode remote session on a cloud VM"

setup:
	@if conda env list | grep -q $(CONDA_ENV); then \
		echo "Conda environment '$(CONDA_ENV)' already exists. Skipping creation."; \
	else \
		conda create -n $(CONDA_ENV) python=3.11 -y; \
		$(CONDA_RUN) pip install -e ".[dev,train]"; \
		$(CONDA_RUN) pre-commit install; \
	fi

upgrade:
	$(CONDA_RUN) pip install --upgrade -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

check:
	$(CONDA_RUN) pre-commit run --all-files

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

.PHONY: help setup upgrade clean check format test train evaluate infer skyssh skycode
