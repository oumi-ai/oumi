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

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

import oumi
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.synth import synth
from oumi.core.configs import (
    InferenceConfig,
    ModelParams,
    SynthesisConfig,
)
from oumi.core.configs.params.synthesis_params import GeneralSynthesisParams
from oumi.utils.logging import logger

runner = CliRunner()


def _create_synthesis_config() -> SynthesisConfig:
    return SynthesisConfig(
        strategy_params=GeneralSynthesisParams(),
        inference_config=InferenceConfig(
            model=ModelParams(
                model_name="MlpEncoder",
                trust_remote_code=True,
                tokenizer_name="openai-community/gpt2",
            ),
        ),
        num_samples=5,
        output_path="test_output.jsonl",
    )


@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(synth)
    yield fake_app


@pytest.fixture
def mock_synthesize():
    with patch.object(oumi, "synthesize") as m_synthesize:
        yield m_synthesize


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.cli_utils.resolve_and_fetch_config") as m_fetch:
        yield m_fetch


@pytest.fixture
def mock_parse_extra_cli_args():
    with patch("oumi.cli.cli_utils.parse_extra_cli_args") as m_parse:
        m_parse.return_value = {}
        yield m_parse


@pytest.fixture
def mock_synthesis_config_from_yaml():
    with patch(
        "oumi.core.configs.synthesis_config.SynthesisConfig.from_yaml_and_arg_list"
    ) as m_config:
        yield m_config


@pytest.fixture
def sample_synthesis_results():
    return [
        {"prompt": "What is AI?", "response": "Artificial Intelligence..."},
        {"prompt": "What is ML?", "response": "Machine Learning..."},
        {"prompt": "What is DL?", "response": "Deep Learning..."},
    ]


def test_synth_runs(
    app,
    mock_synthesize,
    mock_parse_extra_cli_args,
    mock_synthesis_config_from_yaml,
    sample_synthesis_results,
):
    """Test that synth command runs successfully with basic configuration."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.to_yaml(yaml_path)

        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.return_value = sample_synthesis_results

        result = runner.invoke(app, ["--config", yaml_path])

        assert result.exit_code == 0
        mock_parse_extra_cli_args.assert_called_once()
        mock_synthesis_config_from_yaml.assert_called_once_with(
            yaml_path, {}, logger=logger
        )
        mock_synthesize.assert_called_once_with(config)


def test_synth_with_overrides(
    app,
    mock_synthesize,
    mock_parse_extra_cli_args,
    mock_synthesis_config_from_yaml,
    sample_synthesis_results,
):
    """Test synth command with CLI argument overrides."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.to_yaml(yaml_path)

        expected_config = _create_synthesis_config()
        expected_config.num_samples = 10
        expected_config.output_path = "custom_output.jsonl"

        mock_synthesis_config_from_yaml.return_value = expected_config
        mock_synthesize.return_value = sample_synthesis_results

        result = runner.invoke(
            app,
            [
                "--config",
                yaml_path,
                "--num_samples",
                "10",
                "--output_path",
                "custom_output.jsonl",
            ],
        )

        assert result.exit_code == 0
        mock_synthesize.assert_called_once_with(expected_config)


def test_synth_with_oumi_prefix(
    app,
    mock_synthesize,
    mock_synthesis_config_from_yaml,
    mock_fetch,
    sample_synthesis_results,
):
    """Test synth command with oumi:// prefixed config path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/examples/synthesis/basic.yaml"
        expected_path = output_dir / "configs/examples/synthesis/basic.yaml"

        config = _create_synthesis_config()
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(expected_path)

        mock_fetch.return_value = expected_path
        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.return_value = sample_synthesis_results

        with patch.dict("os.environ", {"OUMI_DIR": str(output_dir)}):
            result = runner.invoke(app, ["--config", config_path])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(config_path)
        mock_synthesize.assert_called_once_with(config)


def test_synth_auto_generates_output_path(
    app,
    mock_synthesize,
    mock_synthesis_config_from_yaml,
    sample_synthesis_results,
):
    """Test that synth auto-generates output path when not specified."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.output_path = None  # No output path specified
        config.to_yaml(yaml_path)

        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.return_value = sample_synthesis_results

        with (
            patch("oumi.cli.synth.Path.cwd") as mock_cwd,
            patch("oumi.cli.synth.datetime") as mock_datetime,
        ):
            mock_cwd.return_value = Path(output_temp_dir)
            mock_datetime.now.return_value.strftime.return_value = "20250124_120000"

            result = runner.invoke(app, ["--config", yaml_path])

            assert result.exit_code == 0
            # Verify that output_path was set on the config
            args, _ = mock_synthesize.call_args
            actual_config = args[0]
            assert actual_config.output_path.endswith(
                "oumi_synth_results_20250124_120000.jsonl"
            )


def test_synth_logging_levels(
    app,
    mock_synthesize,
    mock_synthesis_config_from_yaml,
    sample_synthesis_results,
):
    """Test synth command with different logging levels."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.to_yaml(yaml_path)

        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.return_value = sample_synthesis_results

        result = runner.invoke(app, ["--config", yaml_path, "--log-level", "WARNING"])
        assert result.exit_code == 0
        assert logger.level == logging.WARNING

        result = runner.invoke(app, ["--config", yaml_path, "-log", "CRITICAL"])
        assert result.exit_code == 0
        assert logger.level == logging.CRITICAL


def test_synth_fails_no_config(app):
    """Test that synth command fails when no config is provided."""
    result = runner.invoke(app, [])
    assert result.exit_code == 2


def test_synth_with_empty_results(
    app,
    mock_synthesize,
    mock_synthesis_config_from_yaml,
):
    """Test synth command handles empty synthesis results."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.to_yaml(yaml_path)

        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.return_value = []

        result = runner.invoke(app, ["--config", yaml_path])

        assert "No results found" in result.stdout
        assert result.exit_code == 0
        mock_synthesize.assert_called_once_with(config)


def test_synth_with_large_results(
    app,
    mock_synthesize,
    mock_synthesis_config_from_yaml,
):
    """Test synth command with more than 5 results (shows truncation message)."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.to_yaml(yaml_path)

        # Create 7 sample results
        large_results = [
            {"prompt": f"Question {i}?", "response": f"Answer {i}"} for i in range(7)
        ]

        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.return_value = large_results

        result = runner.invoke(app, ["--config", yaml_path])

        assert result.exit_code == 0
        mock_synthesize.assert_called_once_with(config)
        assert "and 6 more samples" in result.stdout


def test_synth_config_validation_error(
    app,
    mock_synthesis_config_from_yaml,
):
    """Test synth command handles config validation errors."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")

        # Mock config validation to raise an error
        mock_synthesis_config_from_yaml.side_effect = ValueError("Invalid config")

        result = runner.invoke(app, ["--config", yaml_path])

        # Should fail due to config validation error
        assert result.exit_code != 0


def test_synth_synthesis_error(
    app,
    mock_synthesize,
    mock_synthesis_config_from_yaml,
):
    """Test synth command handles synthesis errors."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.to_yaml(yaml_path)

        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.side_effect = RuntimeError("Synthesis failed")

        result = runner.invoke(app, ["--config", yaml_path])

        # Should fail due to synthesis error
        assert result.exit_code != 0
        mock_synthesize.assert_called_once_with(config)


def test_synth_output_path_collision_handling(
    app,
    mock_synthesize,
    mock_synthesis_config_from_yaml,
    sample_synthesis_results,
):
    """Test that synth handles output path collisions by incrementing filename."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.output_path = None  # No output path specified
        config.to_yaml(yaml_path)

        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.return_value = sample_synthesis_results

        # Create a file that would collide with the auto-generated name
        with (
            patch("oumi.cli.synth.Path.cwd") as mock_cwd,
            patch("oumi.cli.synth.datetime") as mock_datetime,
        ):
            mock_cwd.return_value = Path(output_temp_dir)
            mock_datetime.now.return_value.strftime.return_value = "20250124_120000"

            # Create the file that would be auto-generated
            collision_file = (
                Path(output_temp_dir) / "oumi_synth_results_20250124_120000.jsonl"
            )
            collision_file.touch()

            result = runner.invoke(app, ["--config", yaml_path])

            assert result.exit_code == 0
            # Verify that output_path was incremented to avoid collision
            args, _ = mock_synthesize.call_args
            actual_config = args[0]
            assert actual_config.output_path.endswith(
                "oumi_synth_results_20250124_120000_1.jsonl"
            )


def test_synth_config_finalization(
    app,
    mock_synthesize,
    mock_synthesis_config_from_yaml,
    sample_synthesis_results,
):
    """Test that synth calls config finalization and validation."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.to_yaml(yaml_path)

        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.return_value = sample_synthesis_results

        with patch.object(config, "finalize_and_validate") as mock_finalize:
            result = runner.invoke(app, ["--config", yaml_path])

            assert result.exit_code == 0
            mock_finalize.assert_called_once()


def test_synth_short_flag_syntax(
    app,
    mock_synthesize,
    mock_synthesis_config_from_yaml,
    sample_synthesis_results,
):
    """Test synth command with short flag syntax."""
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "synth.yaml")
        config: SynthesisConfig = _create_synthesis_config()
        config.to_yaml(yaml_path)

        mock_synthesis_config_from_yaml.return_value = config
        mock_synthesize.return_value = sample_synthesis_results

        result = runner.invoke(app, ["-c", yaml_path])

        assert result.exit_code == 0
        mock_synthesize.assert_called_once_with(config)
