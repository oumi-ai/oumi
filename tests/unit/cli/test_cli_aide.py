# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the AIDE CLI command."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.aide_cmd import aide
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.core.agentic.base_agentic_optimizer import AideResult
from oumi.core.configs import AideConfig, AideParams, ModelParams
from oumi.utils.logging import logger

runner = CliRunner()


def _create_aide_config() -> AideConfig:
    return AideConfig(
        model=ModelParams(
            model_name="test-model",
            trust_remote_code=True,
        ),
        goal="Test optimization goal",
        aide=AideParams(
            steps=2,
            target_metric="eval_loss",
            target_direction="minimize",
            output_dir="/tmp/aide_test",
        ),
    )


def _create_aide_result() -> AideResult:
    return AideResult(
        best_code="print('hello')",
        best_metric=0.5,
        total_steps=2,
        good_solutions=1,
        buggy_solutions=1,
        journal_path="/tmp/journal.json",
        best_solution_path="/tmp/best.py",
    )


@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(aide)
    yield fake_app


@pytest.fixture
def mock_aide():
    with patch("oumi.aide.aide") as m_aide:
        yield m_aide


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.cli_utils.resolve_and_fetch_config") as m_fetch:
        yield m_fetch


@pytest.fixture
def mock_parse_extra_cli_args():
    with patch("oumi.cli.cli_utils.parse_extra_cli_args") as m_parse:
        m_parse.return_value = []
        yield m_parse


@pytest.fixture
def mock_aide_config_from_yaml():
    with patch(
        "oumi.core.configs.aide_config.AideConfig.from_yaml_and_arg_list"
    ) as m_config:
        yield m_config


def test_aide_runs(
    app,
    mock_aide,
    mock_parse_extra_cli_args,
    mock_aide_config_from_yaml,
):
    """Test that aide command runs successfully with basic configuration."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "aide.yaml")
        config = _create_aide_config()
        config.to_yaml(yaml_path)

        mock_aide_config_from_yaml.return_value = config
        mock_aide.return_value = _create_aide_result()

        result = runner.invoke(app, ["--config", yaml_path])

        assert result.exit_code == 0
        mock_parse_extra_cli_args.assert_called_once()
        mock_aide_config_from_yaml.assert_called_once_with(yaml_path, [], logger=logger)
        mock_aide.assert_called_once()


def test_aide_with_overrides(
    app,
    mock_aide,
    mock_parse_extra_cli_args,
    mock_aide_config_from_yaml,
):
    """Test aide command with CLI argument overrides."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "aide.yaml")
        config = _create_aide_config()
        config.to_yaml(yaml_path)

        expected_config = _create_aide_config()
        expected_config.aide.steps = 10

        mock_aide_config_from_yaml.return_value = expected_config
        mock_aide.return_value = _create_aide_result()

        result = runner.invoke(
            app,
            ["--config", yaml_path, "--aide.steps", "10"],
        )

        assert result.exit_code == 0
        mock_aide.assert_called_once()


def test_aide_displays_results(
    app,
    mock_aide,
    mock_parse_extra_cli_args,
    mock_aide_config_from_yaml,
):
    """Test that aide command displays results correctly."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "aide.yaml")
        config = _create_aide_config()
        config.to_yaml(yaml_path)

        mock_aide_config_from_yaml.return_value = config
        mock_aide.return_value = _create_aide_result()

        result = runner.invoke(app, ["--config", yaml_path])

        assert result.exit_code == 0
        assert "AIDE optimization complete" in result.output
        assert "0.5" in result.output  # best_metric
        assert "1 good" in result.output
        assert "1 buggy" in result.output


def test_aide_missing_config(app):
    """Test that aide command fails gracefully with missing config."""
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_aide_with_oumi_prefix(
    app,
    mock_aide,
    mock_aide_config_from_yaml,
    mock_fetch,
):
    """Test aide command with oumi:// prefixed config path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/aide/135m/aide.yaml"
        expected_path = output_dir / "configs/recipes/smollm/aide/135m/aide.yaml"

        mock_fetch.return_value = expected_path

        config = _create_aide_config()
        mock_aide_config_from_yaml.return_value = config
        mock_aide.return_value = _create_aide_result()

        runner.invoke(app, ["--config", config_path])

        mock_fetch.assert_called_once()
