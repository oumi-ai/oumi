import logging
import tempfile
from pathlib import Path
from unittest.mock import call, patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.core.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.core.cli.evaluate import evaluate
from oumi.core.configs import (
    EvaluationConfig,
    LMHarnessParams,
    ModelParams,
)
from oumi.utils.logging import logger

runner = CliRunner()


def _create_eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        output_dir="output/dir",
        lm_harness_params=LMHarnessParams(
            tasks=["mmlu"],
            num_samples=4,
        ),
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
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
    with patch("oumi.core.cli.evaluate.oumi_evaluate") as m_evaluate:
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
                "--model.model_name",
                "new_name",
                "--lm_harness_params.num_samples",
                "5",
            ],
        )
        expected_config = _create_eval_config()
        expected_config.model.model_name = "new_name"
        if expected_config.lm_harness_params:
            expected_config.lm_harness_params.num_samples = 5
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
