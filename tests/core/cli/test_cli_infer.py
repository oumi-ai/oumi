import tempfile
from pathlib import Path
from unittest.mock import call, patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.core.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.core.cli.infer import infer
from oumi.core.configs import (
    GenerationConfig,
    InferenceConfig,
    ModelParams,
)

runner = CliRunner()


def _create_inference_config() -> InferenceConfig:
    return InferenceConfig(
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
        ),
        generation=GenerationConfig(
            max_new_tokens=5,
        ),
    )


#
# Fixtures
#
@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(infer)
    yield fake_app


@pytest.fixture
def mock_infer():
    with patch("oumi.core.cli.infer.oumi.infer") as m_infer:
        yield m_infer


def test_infer_runs(app, mock_infer):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["--config", yaml_path])
        mock_infer.infer_interactive.assert_has_calls([call(config)])


def test_infer_with_overrides(app, mock_infer):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(
            app,
            [
                "--config",
                yaml_path,
                "--model.model_name",
                "new_name",
                "--generation.max_new_tokens",
                "5",
            ],
        )
        expected_config = _create_inference_config()
        expected_config.model.model_name = "new_name"
        expected_config.generation.max_new_tokens = 5
        mock_infer.infer_interactive.assert_has_calls([call(expected_config)])


def test_infer_not_interactive_runs(app, mock_infer):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.generation.input_filepath = "some/path"
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["--config", yaml_path, "--detach"])
        mock_infer.infer.assert_has_calls(
            [
                call(
                    model_params=config.model,
                    generation_config=config.generation,
                    input=[],
                )
            ]
        )


def test_infer_not_interactive_with_overrides(app, mock_infer):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.generation.input_filepath = "some/path"
        config.to_yaml(yaml_path)
        _ = runner.invoke(
            app,
            [
                "--config",
                yaml_path,
                "--detach",
                "--model.model_name",
                "new_name",
                "--generation.max_new_tokens",
                "5",
            ],
        )
        expected_config = _create_inference_config()
        expected_config.model.model_name = "new_name"
        expected_config.generation.max_new_tokens = 5
        expected_config.generation.input_filepath = "some/path"
        mock_infer.infer.assert_has_calls(
            [
                call(
                    model_params=expected_config.model,
                    generation_config=expected_config.generation,
                    input=[],
                )
            ]
        )
