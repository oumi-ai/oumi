import logging
import tempfile
from pathlib import Path
from unittest.mock import call, patch

import pytest
import typer
from typer.testing import CliRunner

import oumi
import oumi.core.distributed
import oumi.utils.torch_utils
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.train import train
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from oumi.utils.logging import logger

runner = CliRunner()


def _create_training_config() -> TrainingConfig:
    return TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="debug_sft",
                    )
                ],
            ),
        ),
        model=ModelParams(
            model_name="MlpEncoder",
            model_max_length=1024,
            trust_remote_code=True,
            tokenizer_name="gpt2",
        ),
        training=TrainingParams(
            trainer_type=TrainerType.TRL_SFT,
            max_steps=3,
            logging_steps=3,
            log_model_summary=True,
            enable_wandb=False,
            enable_tensorboard=False,
            try_resume_from_last_checkpoint=True,
            save_final_model=True,
        ),
    )


#
# Fixtures
#
@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(train)
    yield fake_app


@pytest.fixture
def mock_train():
    with patch.object(oumi, "train", autospec=True) as m_train:
        yield m_train


@pytest.fixture
def mock_limit_per_process_memory():
    with patch.object(
        oumi.utils.torch_utils, "limit_per_process_memory", autospec=True
    ) as m_memory:
        yield m_memory


@pytest.fixture
def mock_device_cleanup():
    with patch.object(
        oumi.utils.torch_utils, "device_cleanup", autospec=True
    ) as m_cleanup:
        yield m_cleanup


@pytest.fixture
def mock_set_random_seeds():
    with patch.object(
        oumi.core.distributed, "set_random_seeds", autospec=True
    ) as m_seeds:
        yield m_seeds


def test_train_runs(
    app,
    mock_train,
    mock_limit_per_process_memory,
    mock_device_cleanup,
    mock_set_random_seeds,
):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        _ = runner.invoke(app, ["--config", train_yaml_path, "--log-level", "ERROR"])
        mock_limit_per_process_memory.assert_called_once()
        mock_train.assert_has_calls([call(config)])
        mock_device_cleanup.assert_has_calls([call(), call()])
        mock_set_random_seeds.assert_called_once()
        assert logger.level == logging.ERROR


def test_train_with_overrides(
    app,
    mock_train,
    mock_limit_per_process_memory,
    mock_device_cleanup,
    mock_set_random_seeds,
):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        _ = runner.invoke(
            app,
            [
                "--config",
                train_yaml_path,
                "--model.tokenizer_name",
                "new_name",
                "--training.max_steps",
                "5",
            ],
        )
        mock_limit_per_process_memory.assert_called_once()
        expected_config = _create_training_config()
        expected_config.model.tokenizer_name = "new_name"
        expected_config.training.max_steps = 5
        mock_train.assert_has_calls([call(expected_config)])
        mock_device_cleanup.assert_has_calls([call(), call()])
        mock_set_random_seeds.assert_called_once()
