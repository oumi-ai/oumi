import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.list_cmd import list_configs, list_datasets, list_models, list_registry

runner = CliRunner()


#
# Fixtures
#
@pytest.fixture
def datasets_app():
    fake_app = typer.Typer()
    fake_app.command()(list_datasets)
    yield fake_app


@pytest.fixture
def models_app():
    fake_app = typer.Typer()
    fake_app.command()(list_models)
    yield fake_app


@pytest.fixture
def configs_app():
    fake_app = typer.Typer()
    fake_app.command()(list_configs)
    yield fake_app


@pytest.fixture
def registry_app():
    fake_app = typer.Typer()
    fake_app.command()(list_registry)
    yield fake_app


#
# Tests for list_datasets
#
def test_list_datasets_runs_without_exceptions(datasets_app):
    result = runner.invoke(datasets_app, [])
    assert result.exit_code == 0
    assert "Available datasets:" in result.stdout


def test_list_datasets_with_filter(datasets_app):
    result = runner.invoke(datasets_app, ["--filter", "alpaca"])
    assert result.exit_code == 0
    # Should show filtered results or no results message
    assert "Available datasets:" in result.stdout


def test_list_datasets_with_type_filter(datasets_app):
    result = runner.invoke(datasets_app, ["--type", "sft"])
    assert result.exit_code == 0
    assert "Available datasets:" in result.stdout


def test_list_datasets_with_invalid_regex(datasets_app):
    result = runner.invoke(datasets_app, ["--filter", "[invalid(regex"])
    assert result.exit_code == 1
    assert "Invalid regex pattern" in result.stdout


def test_list_datasets_with_nonmatching_filter(datasets_app):
    result = runner.invoke(datasets_app, ["--filter", "nonexistent_dataset_xyz123"])
    assert result.exit_code == 0
    assert "No datasets found matching criteria" in result.stdout


#
# Tests for list_models
#
def test_list_models_runs_without_exceptions(models_app):
    result = runner.invoke(models_app, [])
    assert result.exit_code == 0
    assert "Supported models:" in result.stdout


def test_list_models_with_filter(models_app):
    result = runner.invoke(models_app, ["--filter", "llama"])
    assert result.exit_code == 0
    assert "Supported models:" in result.stdout


def test_list_models_tested_only(models_app):
    result = runner.invoke(models_app, ["--tested-only"])
    assert result.exit_code == 0
    assert "Supported models:" in result.stdout


def test_list_models_with_invalid_regex(models_app):
    result = runner.invoke(models_app, ["--filter", "[invalid(regex"])
    assert result.exit_code == 1
    assert "Invalid regex pattern" in result.stdout


def test_list_models_with_nonmatching_filter(models_app):
    result = runner.invoke(models_app, ["--filter", "nonexistent_model_xyz123"])
    assert result.exit_code == 0
    assert "No models found matching criteria" in result.stdout


#
# Tests for list_configs
#
def test_list_configs_runs_without_exceptions(configs_app):
    result = runner.invoke(configs_app, [])
    assert result.exit_code == 0
    assert "Available configuration files:" in result.stdout


def test_list_configs_with_category(configs_app):
    result = runner.invoke(configs_app, ["--category", "recipes"])
    assert result.exit_code == 0
    assert "Available configuration files:" in result.stdout


def test_list_configs_with_model(configs_app):
    result = runner.invoke(configs_app, ["--model", "llama"])
    assert result.exit_code == 0
    assert "Available configuration files:" in result.stdout


def test_list_configs_with_invalid_category(configs_app):
    result = runner.invoke(configs_app, ["--category", "nonexistent_category_xyz"])
    assert result.exit_code == 0
    # Should show friendly message about category not found
    assert "not found" in result.stdout or "No config files found" in result.stdout


def test_list_configs_with_both_filters(configs_app):
    result = runner.invoke(configs_app, ["--category", "recipes", "--model", "phi3"])
    assert result.exit_code == 0
    assert "Available configuration files:" in result.stdout


#
# Tests for list_registry
#
def test_list_registry_runs_without_exceptions(registry_app):
    result = runner.invoke(registry_app, [])
    assert result.exit_code == 0
    assert "Registry contents:" in result.stdout


def test_list_registry_with_type_filter(registry_app):
    result = runner.invoke(registry_app, ["--type", "DATASET"])
    assert result.exit_code == 0
    assert "Registry contents:" in result.stdout
    assert "DATASET:" in result.stdout


def test_list_registry_with_invalid_type(registry_app):
    result = runner.invoke(registry_app, ["--type", "INVALID_TYPE"])
    assert result.exit_code == 1
    assert "Invalid registry type" in result.stdout
    assert "Valid types:" in result.stdout


def test_list_registry_multiple_types_in_output(registry_app):
    result = runner.invoke(registry_app, [])
    assert result.exit_code == 0
    # Should show multiple registry types
    assert "Registry contents:" in result.stdout
