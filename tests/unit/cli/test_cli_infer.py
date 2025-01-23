import io
import logging
import tempfile
from pathlib import Path
from unittest.mock import call, patch

import PIL.Image
import pytest
import typer
from typer.testing import CliRunner

import oumi
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.infer import infer
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
)
from oumi.utils.logging import logger

runner = CliRunner()


def _create_inference_config() -> InferenceConfig:
    return InferenceConfig(
        model=ModelParams(
            model_name="MlpEncoder",
            trust_remote_code=True,
            tokenizer_name="gpt2",
        ),
        generation=GenerationParams(
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
    with patch.object(oumi, "infer") as m_infer:
        yield m_infer


@pytest.fixture
def mock_infer_interactive():
    with patch.object(oumi, "infer_interactive") as m_infer:
        yield m_infer


def test_infer_runs(app, mock_infer, mock_infer_interactive):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["-i", "--config", yaml_path])
        mock_infer_interactive.assert_has_calls([call(config, input_image_bytes=None)])


def test_infer_with_overrides(app, mock_infer, mock_infer_interactive):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(
            app,
            [
                "--interactive",
                "--config",
                yaml_path,
                "--model.tokenizer_name",
                "new_name",
                "--generation.max_new_tokens",
                "5",
                "--engine",
                "VLLM",
            ],
        )
        expected_config = _create_inference_config()
        expected_config.model.tokenizer_name = "new_name"
        expected_config.generation.max_new_tokens = 5
        expected_config.engine = InferenceEngineType.VLLM
        mock_infer_interactive.assert_has_calls(
            [call(expected_config, input_image_bytes=None)]
        )


def test_infer_runs_with_image(app, mock_infer, mock_infer_interactive):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)

        test_image = PIL.Image.new(mode="RGB", size=(32, 16))
        temp_io_output = io.BytesIO()
        test_image.save(temp_io_output, format="PNG")
        image_bytes = temp_io_output.getvalue()

        image_path = Path(output_temp_dir) / "test_image.png"
        with image_path.open(mode="wb") as f:
            f.write(image_bytes)

        _ = runner.invoke(
            app, ["-i", "--config", yaml_path, "--image", str(image_path)]
        )
        mock_infer_interactive.assert_has_calls(
            [call(config, input_image_bytes=image_bytes)]
        )


def test_infer_not_interactive_runs(app, mock_infer, mock_infer_interactive):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.input_path = "some/path"
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["--config", yaml_path])
        mock_infer.assert_has_calls([call(config)])


def test_infer_not_interactive_with_overrides(app, mock_infer, mock_infer_interactive):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.input_path = "some/path"
        config.to_yaml(yaml_path)
        _ = runner.invoke(
            app,
            [
                "--config",
                yaml_path,
                "--model.tokenizer_name",
                "new_name",
                "--generation.max_new_tokens",
                "5",
            ],
        )
        expected_config = _create_inference_config()
        expected_config.model.tokenizer_name = "new_name"
        expected_config.generation.max_new_tokens = 5
        expected_config.input_path = "some/path"
        mock_infer.assert_has_calls([call(expected_config)])


def test_infer_logging_levels(app, mock_infer, mock_infer_interactive):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["-i", "--config", yaml_path, "--log-level", "WARNING"])
        assert logger.level == logging.WARNING
        _ = runner.invoke(app, ["-i", "--config", yaml_path, "-log", "CRITICAL"])
        assert logger.level == logging.CRITICAL
