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
from oumi.cli.alias import AliasType
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


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.cli_utils.resolve_and_fetch_config") as m_fetch:
        yield m_fetch


@pytest.fixture
def mock_alias():
    with patch("oumi.cli.infer.try_get_config_name_for_alias") as try_alias:
        yield try_alias


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
        mock_infer_interactive.assert_has_calls(
            [call(config, input_image_bytes=None, system_prompt=None)]
        )


def test_infer_with_alias_runs(app, mock_infer, mock_infer_interactive, mock_alias):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        mock_alias.return_value = yaml_path
        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["-i", "--config", "random_alias"])
        mock_alias.assert_called_once_with("random_alias", AliasType.INFER)
        mock_infer_interactive.assert_has_calls(
            [call(config, input_image_bytes=None, system_prompt=None)]
        )


def test_infer_runs_interactive_by_default(app, mock_infer, mock_infer_interactive):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["--config", yaml_path])
        mock_infer_interactive.assert_has_calls(
            [call(config, input_image_bytes=None, system_prompt=None)]
        )


def test_infer_fails_no_args(app, mock_infer, mock_infer_interactive):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")
        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)
        result = runner.invoke(app, [])
        mock_infer_interactive.assert_not_called()
        assert result.exit_code == 2


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
            [call(expected_config, input_image_bytes=None, system_prompt=None)]
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
            [call(config, input_image_bytes=[image_bytes], system_prompt=None)]
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


def test_infer_with_system_prompt(app, mock_infer_interactive):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "infer.yaml")

        config: InferenceConfig = _create_inference_config()
        config.to_yaml(yaml_path)

        # Test with interactive mode and system prompt
        result = runner.invoke(
            app,
            [
                "-i",
                "--config",
                yaml_path,
                "--system-prompt",
                "You are a mighty assistant",
            ],
        )
        assert result.exit_code == 0
        mock_infer_interactive.assert_called_once_with(
            config, system_prompt="You are a mighty assistant", input_image_bytes=None
        )
        mock_infer_interactive.reset_mock()


def test_infer_with_system_prompt_and_image(app, mock_infer_interactive):
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

        result = runner.invoke(
            app,
            [
                "-i",
                "--config",
                yaml_path,
                "--system-prompt",
                "You are not an average assistant",
                "--image",
                str(image_path),
            ],
        )
        assert result.exit_code == 0
        mock_infer_interactive.assert_called_once_with(
            config,
            system_prompt="You are not an average assistant",
            input_image_bytes=[image_bytes],
        )


def test_infer_with_oumi_prefix_and_explicit_output_dir(
    app, mock_fetch, mock_infer_interactive
):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"

        config = _create_inference_config()
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(expected_path)
        mock_fetch.return_value = expected_path

        with patch.dict("os.environ", {"OUMI_DIR": str(output_dir)}):
            result = runner.invoke(app, ["-i", "--config", config_path])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(config_path)
        mock_infer_interactive.assert_called_once_with(
            config, input_image_bytes=None, system_prompt=None
        )
