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

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

import oumi
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.prompt import prompt
from oumi.core.configs import ModelParams
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.prompt_optimization_params import (
    PromptOptimizationParams,
)
from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.utils.logging import logger

runner = CliRunner()


def _create_prompt_optimization_config(temp_dir: str) -> PromptOptimizationConfig:
    """Create a basic prompt optimization configuration for testing."""
    return PromptOptimizationConfig(
        model=ModelParams(
            model_name="HuggingFaceTB/SmolLM2-135M",
            trust_remote_code=True,
        ),
        generation=GenerationParams(
            max_new_tokens=128,
            temperature=0.7,
        ),
        optimization=PromptOptimizationParams(
            optimizer="mipro",
            num_trials=10,
            verbose=False,
        ),
        train_dataset_path=str(Path(temp_dir) / "train.jsonl"),
        output_dir=str(Path(temp_dir) / "output"),
        metric="accuracy",
    )


@pytest.fixture
def app():
    """Create a test Typer app with the prompt command."""
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(prompt)
    yield fake_app


@pytest.fixture
def mock_optimize_prompt():
    """Mock the optimize_prompt function."""
    with patch.object(oumi, "optimize_prompt") as m_optimize:
        yield m_optimize


@pytest.fixture
def mock_parse_extra_cli_args():
    """Mock CLI argument parsing."""
    with patch("oumi.cli.cli_utils.parse_extra_cli_args") as m_parse:
        m_parse.return_value = []
        yield m_parse


@pytest.fixture
def mock_prompt_config_from_yaml():
    """Mock configuration loading from YAML."""
    with patch(
        "oumi.core.configs.prompt_config.PromptOptimizationConfig.from_yaml_and_arg_list"
    ) as m_config:
        yield m_config


@pytest.fixture
def sample_optimization_results():
    """Sample optimization results."""
    return {
        "final_score": 0.85,
        "num_trials": 10,
        "output_dir": "./prompt_optimization_output",
    }


def test_prompt_runs(
    app,
    mock_optimize_prompt,
    mock_parse_extra_cli_args,
    mock_prompt_config_from_yaml,
    sample_optimization_results,
):
    """Test that prompt command runs successfully with basic configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = str(Path(temp_dir) / "prompt.yaml")
        config = _create_prompt_optimization_config(temp_dir)
        config.to_yaml(yaml_path)

        mock_prompt_config_from_yaml.return_value = config
        mock_optimize_prompt.return_value = sample_optimization_results

        result = runner.invoke(app, ["--config", yaml_path])

        assert result.exit_code == 0
        mock_parse_extra_cli_args.assert_called_once()
        mock_prompt_config_from_yaml.assert_called_once_with(
            yaml_path, [], logger=logger
        )
        mock_optimize_prompt.assert_called_once_with(config)


def test_prompt_with_verbose(
    app,
    mock_optimize_prompt,
    mock_parse_extra_cli_args,
    mock_prompt_config_from_yaml,
    sample_optimization_results,
):
    """Test prompt command with verbose flag."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = str(Path(temp_dir) / "prompt.yaml")
        config = _create_prompt_optimization_config(temp_dir)
        config.to_yaml(yaml_path)

        mock_prompt_config_from_yaml.return_value = config
        mock_optimize_prompt.return_value = sample_optimization_results

        result = runner.invoke(app, ["--config", yaml_path, "--verbose"])

        assert result.exit_code == 0
        mock_optimize_prompt.assert_called_once_with(config)


def test_prompt_with_cli_overrides(
    app,
    mock_optimize_prompt,
    mock_parse_extra_cli_args,
    mock_prompt_config_from_yaml,
    sample_optimization_results,
):
    """Test prompt command with CLI argument overrides."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = str(Path(temp_dir) / "prompt.yaml")
        config = _create_prompt_optimization_config(temp_dir)
        config.to_yaml(yaml_path)

        # Simulate CLI overrides
        cli_overrides = ["optimization.num_trials=50", "optimization.optimizer=gepa"]
        mock_parse_extra_cli_args.return_value = cli_overrides

        # Update config to reflect overrides
        overridden_config = _create_prompt_optimization_config(temp_dir)
        overridden_config.optimization.num_trials = 50
        overridden_config.optimization.optimizer = "gepa"
        mock_prompt_config_from_yaml.return_value = overridden_config
        mock_optimize_prompt.return_value = sample_optimization_results

        result = runner.invoke(
            app,
            [
                "--config",
                yaml_path,
                "--optimization.num_trials=50",
                "--optimization.optimizer=gepa",
            ],
        )

        assert result.exit_code == 0
        mock_parse_extra_cli_args.assert_called_once()
        mock_prompt_config_from_yaml.assert_called_once_with(
            yaml_path, cli_overrides, logger=logger
        )
        mock_optimize_prompt.assert_called_once()


def test_prompt_displays_results(
    app,
    mock_optimize_prompt,
    mock_parse_extra_cli_args,
    mock_prompt_config_from_yaml,
    sample_optimization_results,
):
    """Test that prompt command displays optimization results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = str(Path(temp_dir) / "prompt.yaml")
        config = _create_prompt_optimization_config(temp_dir)
        config.to_yaml(yaml_path)

        mock_prompt_config_from_yaml.return_value = config
        mock_optimize_prompt.return_value = sample_optimization_results

        result = runner.invoke(app, ["--config", yaml_path])

        assert result.exit_code == 0
        # Check that results are displayed in output
        assert "0.85" in result.stdout or "0.8500" in result.stdout  # final_score
        assert "10" in result.stdout  # num_trials
        assert "output" in result.stdout.lower()  # output directory
