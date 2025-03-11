import logging
import tempfile
from pathlib import Path
from unittest.mock import call, patch

import pytest
import typer
from typer.testing import CliRunner

import oumi
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.evaluate import evaluate
from oumi.core.configs import (
    EvaluationConfig,
    EvaluationTaskParams,
    ModelParams,
)
from oumi.utils.logging import logger

runner = CliRunner()


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.cli_utils.resolve_and_fetch_config") as m_fetch:
        yield m_fetch


def _create_eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        output_dir="output/dir",
        tasks=[
            EvaluationTaskParams(
                evaluation_backend="lm_harness",
                task_name="mmlu",
                num_samples=4,
            )
        ],
        model=ModelParams(
            model_name="MlpEncoder",
            trust_remote_code=True,
            tokenizer_name="gpt2",
        ),
    )


#
# Fixtures
#
@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(evaluate)
    yield fake_app


@pytest.fixture
def mock_evaluate():
    with patch.object(oumi, "evaluate", autospec=True) as m_evaluate:
        yield m_evaluate


def test_evaluate_runs(app, mock_evaluate):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "eval.yaml")
        config: EvaluationConfig = _create_eval_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["--config", yaml_path])
        mock_evaluate.assert_has_calls([call(config)])


def test_evaluate_with_overrides(app, mock_evaluate):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "eval.yaml")
        config: EvaluationConfig = _create_eval_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(
            app,
            [
                "--config",
                yaml_path,
                "--model.tokenizer_name",
                "new_name",
                "--tasks",
                "[{evaluation_backend: lm_harness, num_samples: 5, task_name: mmlu}]",
            ],
        )
        expected_config = _create_eval_config()
        expected_config.model.tokenizer_name = "new_name"
        if expected_config.tasks:
            if expected_config.tasks[0]:
                expected_config.tasks[0].num_samples = 5
        mock_evaluate.assert_has_calls([call(expected_config)])


def test_evaluate_logging_levels(app, mock_evaluate):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        yaml_path = str(Path(output_temp_dir) / "eval.yaml")
        config: EvaluationConfig = _create_eval_config()
        config.to_yaml(yaml_path)
        _ = runner.invoke(app, ["--config", yaml_path, "--log-level", "DEBUG"])
        assert logger.level == logging.DEBUG
        _ = runner.invoke(app, ["--config", yaml_path, "-log", "WARNING"])
        assert logger.level == logging.WARNING


def test_evaluate_with_oumi_prefix(app, mock_evaluate, mock_fetch):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        output_dir = Path(output_temp_dir)
        yaml_path = "oumi://configs/recipes/smollm/evaluation/135m/eval.yaml"
        expected_path = output_dir / "configs/recipes/smollm/evaluation/135m/eval.yaml"

        config: EvaluationConfig = _create_eval_config()
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(expected_path)
        mock_fetch.return_value = expected_path

        with patch.dict("os.environ", {"OUMI_DIR": str(output_dir)}):
            result = runner.invoke(app, ["--config", yaml_path])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(yaml_path)
        mock_evaluate.assert_has_calls([call(config)])
