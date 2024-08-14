import pathlib
import tempfile
from unittest.mock import ANY, Mock, patch

import pytest

from lema.core.types import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from lema.core.types.base_cluster import JobStatus
from lema.launch import _LaunchArgs, _LauncherAction, launch
from lema.launcher import JobConfig, JobResources


#
# Fixtures
#
@pytest.fixture
def mock_launcher():
    with patch("lema.launch.launcher") as launcher_mock:
        yield launcher_mock


@pytest.fixture
def mock_printer():
    with patch("lema.launch._print_and_wait") as printer_mock:
        yield printer_mock


def _create_training_config() -> TrainingConfig:
    return TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="yahma/alpaca-cleaned",
                        preprocessing_function_name="alpaca",
                    )
                ],
                target_col="text",
            ),
        ),
        model=ModelParams(
            model_name="openai-community/gpt2",
            model_max_length=1024,
            trust_remote_code=True,
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


def _create_job_config(training_config_path: str) -> JobConfig:
    return JobConfig(
        name="foo",
        user="bar",
        working_dir=".",
        resources=JobResources(
            cloud="aws",
            region="us-west-1",
            zone=None,
            accelerators="A100-80GB",
            cpus="4",
            memory="64",
            instance_type=None,
            use_spot=True,
            disk_size=512,
            disk_tier="low",
        ),
        run=f"python -m lema.launch {training_config_path}",
    )


def test_launch_launch_job(mock_launcher, mock_printer):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(pathlib.Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        job_yaml_path = str(pathlib.Path(output_temp_dir) / "job.yaml")
        job_config = _create_job_config(train_yaml_path)
        job_config.to_yaml(job_yaml_path)
        mock_launcher.JobConfig = JobConfig
        mock_cluster = Mock()
        job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="running",
            metadata="",
            done=False,
        )
        mock_launcher.up.return_value = (mock_cluster, job_status)
        mock_cluster.get_job.return_value = job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        launch(_LaunchArgs(job=job_yaml_path, action=_LauncherAction.UP))
        mock_printer.assert_called_once_with("Running job job_id", ANY)
        mock_cluster.get_job.assert_called_once_with("job_id")


def test_launch_launch_job_not_found(mock_launcher):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(pathlib.Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        job_yaml_path = str(pathlib.Path(output_temp_dir) / "job.yaml")
        job_config = _create_job_config(train_yaml_path)
        job_config.to_yaml(job_yaml_path)
        mock_launcher.JobConfig = JobConfig
        mock_cluster = Mock()
        job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="running",
            metadata="",
            done=False,
        )
        mock_launcher.up.return_value = (mock_cluster, job_status)
        mock_cluster.get_job.return_value = job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        with pytest.raises(FileNotFoundError) as exception_info:
            launch(
                _LaunchArgs(
                    job=str(pathlib.Path(output_temp_dir) / "fake_path.yaml"),
                    action=_LauncherAction.UP,
                )
            )
        assert "No such file or directory" in str(exception_info.value)
