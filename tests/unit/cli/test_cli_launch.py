import logging
import pathlib
import tempfile
from unittest.mock import Mock, call, patch

import pytest
import typer
from typer.testing import CliRunner

import oumi
from oumi.cli.alias import AliasType
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.launch import cancel, down, status, stop, up, which
from oumi.cli.launch import run as launcher_run
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from oumi.core.launcher import JobStatus
from oumi.launcher import JobConfig, JobResources
from oumi.utils.logging import logger


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.cli_utils.resolve_and_fetch_config") as m_fetch:
        m_fetch.side_effect = lambda x: x
        yield m_fetch


@pytest.fixture
def mock_alias():
    with patch("oumi.cli.launch.try_get_config_name_for_alias") as try_alias:
        yield try_alias


runner = CliRunner()


class MockPool:
    def __init__(self):
        self.mock_result = Mock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def apply_async(self, fn, kwds):
        fn(**kwds)
        return self.mock_result


#
# Fixtures
#
@pytest.fixture
def app():
    launch_app = typer.Typer()
    launch_app.command()(down)
    launch_app.command(name="run", context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(
        launcher_run
    )
    launch_app.command()(status)
    launch_app.command()(stop)
    launch_app.command()(cancel)
    launch_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(up)
    launch_app.command()(which)
    yield launch_app


@pytest.fixture
def mock_launcher():
    with patch.object(oumi, "launcher", autospec=True) as launcher_mock:
        yield launcher_mock


@pytest.fixture
def mock_pool():
    with patch("oumi.cli.launch.Pool") as pool_mock:
        mock_pool = MockPool()
        pool_mock.return_value = mock_pool
        yield pool_mock


@pytest.fixture
def mock_confirm():
    with patch("typer.confirm") as confirm_mock:
        yield confirm_mock


@pytest.fixture
def mock_version():
    with patch("oumi.utils.version_utils.version") as version_mock:
        version_mock.return_value = ""
        yield version_mock


@pytest.fixture
def mock_git_root():
    with patch("oumi.cli.launch.get_git_root_dir") as root_mock:
        root_mock.return_value = _oumi_root()
        yield root_mock


def _oumi_root() -> str:
    return "fake/oumi/root"


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
            enable_mlflow=False,
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
        run=f"oumi launch up --config {training_config_path}",
    )


def test_launch_up_job(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
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
        _ = runner.invoke(
            app,
            [
                "up",
                "--config",
                job_yaml_path,
                "--log-level",
                "DEBUG",
            ],
        )
        mock_fetch.assert_called_once_with(job_yaml_path)
        mock_cluster.get_job.assert_has_calls([call("job_id")])
        assert logger.level == logging.DEBUG


def test_launch_up_job_with_alias(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch, mock_alias
):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(pathlib.Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        job_yaml_path = str(pathlib.Path(output_temp_dir) / "job.yaml")
        mock_alias.return_value = job_yaml_path
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
        _ = runner.invoke(
            app,
            [
                "up",
                "--config",
                "some_alias",
                "--log-level",
                "DEBUG",
            ],
        )
        mock_fetch.assert_called_once_with(job_yaml_path)
        mock_cluster.get_job.assert_has_calls([call("job_id")])
        mock_alias.assert_called_once_with("some_alias", AliasType.JOB)
        assert logger.level == logging.DEBUG


def test_launch_up_job_dev_confirm(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_git_root, mock_fetch
):
    mock_version.return_value = "0.1.0.dev0"
    mock_confirm.return_value = True
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
        _ = runner.invoke(
            app,
            [
                "up",
                "--config",
                job_yaml_path,
            ],
        )
        mock_cluster.get_job.assert_has_calls([call("job_id")])
        job_config.working_dir = _oumi_root()
        mock_launcher.up.assert_called_once_with(job_config, None)


def test_launch_up_job_dev_no_confirm(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
    mock_version.return_value = "0.1.0.dev0"
    mock_confirm.return_value = False
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
        _ = runner.invoke(
            app,
            [
                "up",
                "--config",
                job_yaml_path,
            ],
        )
        mock_cluster.get_job.assert_has_calls([call("job_id")])
        mock_launcher.up.assert_called_once_with(job_config, None)


def test_launch_up_job_dev_no_confirm_same_path(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
    working_dir = "/foo/dir/"
    mock_version.return_value = working_dir
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(pathlib.Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        job_yaml_path = str(pathlib.Path(output_temp_dir) / "job.yaml")
        job_config = _create_job_config(train_yaml_path)
        job_config.working_dir = working_dir
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
        _ = runner.invoke(
            app,
            [
                "up",
                "--config",
                job_yaml_path,
            ],
        )
        mock_cluster.get_job.assert_has_calls([call("job_id")])
        mock_launcher.up.assert_called_once_with(job_config, None)


def test_launch_up_job_existing_cluster(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(pathlib.Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        job_yaml_path = str(pathlib.Path(output_temp_dir) / "job.yaml")
        job_config = _create_job_config(train_yaml_path)
        job_config.to_yaml(job_yaml_path)
        mock_launcher.JobConfig = JobConfig
        mock_cluster = Mock()
        mock_cluster.name.return_value = "cluster_id"
        job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="running",
            metadata="",
            done=False,
        )
        mock_launcher.run.return_value = job_status
        mock_cluster.get_job.return_value = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        mock_cloud = Mock()
        mock_launcher.get_cloud.return_value = mock_cloud
        mock_cloud.get_cluster.return_value = mock_cluster
        _ = runner.invoke(
            app,
            [
                "up",
                "--config",
                job_yaml_path,
                "--cluster",
                "cluster_id",
            ],
        )
        mock_launcher.run.assert_called_once_with(job_config, "cluster_id")
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])


def test_launch_up_job_detach(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
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
        _ = runner.invoke(
            app,
            [
                "up",
                "--config",
                job_yaml_path,
                "--detach",
            ],
        )
        mock_cluster.get_job.assert_not_called()


def test_launch_up_job_detached_local(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(pathlib.Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        job_yaml_path = str(pathlib.Path(output_temp_dir) / "job.yaml")
        job_config = _create_job_config(train_yaml_path)
        job_config.resources.cloud = "local"
        job_config.to_yaml(job_yaml_path)
        mock_launcher.JobConfig = JobConfig
        mock_cluster = Mock()
        job_status = JobStatus(
            id="job_id",
            cluster="local",
            name="job_name",
            status="running",
            metadata="",
            done=False,
        )
        mock_launcher.up.return_value = (mock_cluster, job_status)
        mock_cluster.get_job.return_value = job_status = JobStatus(
            id="job_id",
            cluster="local",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        _ = runner.invoke(
            app,
            [
                "up",
                "--config",
                job_yaml_path,
                "--detach",
            ],
        )
        mock_cluster.get_job.assert_has_calls([call("job_id")])


def test_launch_up_job_not_found(
    app, mock_launcher, mock_version, mock_confirm, mock_fetch
):
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
            res = runner.invoke(
                app,
                [
                    "up",
                    "--config",
                    str(pathlib.Path(output_temp_dir) / "fake_path.yaml"),
                ],
            )
            if res.exception:
                raise res.exception
        assert "No such file or directory" in str(exception_info.value)


def test_launch_run_job(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
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
        mock_cloud = Mock()
        mock_launcher.run.return_value = job_status
        mock_launcher.get_cloud.side_effect = [mock_cloud, mock_cloud]
        mock_cloud.get_cluster.side_effect = [mock_cluster, mock_cluster]
        mock_cluster.get_job.return_value = job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        _ = runner.invoke(
            app,
            [
                "run",
                "--config",
                job_yaml_path,
                "--cluster",
                "cluster_id",
                "-log",
                "CRITICAL",
            ],
        )
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])
        mock_launcher.run.assert_called_once_with(job_config, "cluster_id")
        mock_launcher.get_cloud.assert_has_calls([call("aws"), call("aws")])
        mock_cloud.get_cluster.assert_has_calls(
            [call("cluster_id"), call("cluster_id")]
        )
        mock_fetch.assert_called_once_with(job_yaml_path)
        assert logger.level == logging.CRITICAL


def test_launch_run_job_with_alias(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch, mock_alias
):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(pathlib.Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        job_yaml_path = str(pathlib.Path(output_temp_dir) / "job.yaml")
        mock_alias.return_value = job_yaml_path
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
        mock_cloud = Mock()
        mock_launcher.run.return_value = job_status
        mock_launcher.get_cloud.side_effect = [mock_cloud, mock_cloud]
        mock_cloud.get_cluster.side_effect = [mock_cluster, mock_cluster]
        mock_cluster.get_job.return_value = job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        _ = runner.invoke(
            app,
            [
                "run",
                "--config",
                "some_alias",
                "--cluster",
                "cluster_id",
                "-log",
                "CRITICAL",
            ],
        )
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])
        mock_launcher.run.assert_called_once_with(job_config, "cluster_id")
        mock_launcher.get_cloud.assert_has_calls([call("aws"), call("aws")])
        mock_cloud.get_cluster.assert_has_calls(
            [call("cluster_id"), call("cluster_id")]
        )
        mock_fetch.assert_called_once_with(job_yaml_path)
        mock_alias.assert_called_once_with("some_alias", AliasType.JOB)
        assert logger.level == logging.CRITICAL


def test_launch_run_job_dev_confirm(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_git_root, mock_fetch
):
    mock_version.return_value = "0.1.0.dev0"
    mock_confirm.return_value = True
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
        mock_cloud = Mock()
        mock_launcher.run.return_value = job_status
        mock_launcher.get_cloud.side_effect = [mock_cloud, mock_cloud]
        mock_cloud.get_cluster.side_effect = [mock_cluster, mock_cluster]
        mock_cluster.get_job.return_value = job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        _ = runner.invoke(
            app,
            [
                "run",
                "--config",
                job_yaml_path,
                "--cluster",
                "cluster_id",
            ],
        )
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])
        job_config.working_dir = _oumi_root()
        mock_launcher.run.assert_called_once_with(job_config, "cluster_id")
        mock_launcher.get_cloud.assert_has_calls([call("aws"), call("aws")])
        mock_cloud.get_cluster.assert_has_calls(
            [call("cluster_id"), call("cluster_id")]
        )


def test_launch_run_job_dev_no_confirm(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
    mock_version.return_value = "0.1.0.dev0"
    mock_confirm.return_value = False
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
        mock_cloud = Mock()
        mock_launcher.run.return_value = job_status
        mock_launcher.get_cloud.side_effect = [mock_cloud, mock_cloud]
        mock_cloud.get_cluster.side_effect = [mock_cluster, mock_cluster]
        mock_cluster.get_job.return_value = job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        _ = runner.invoke(
            app,
            [
                "run",
                "--config",
                job_yaml_path,
                "--cluster",
                "cluster_id",
            ],
        )
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])
        mock_launcher.run.assert_called_once_with(job_config, "cluster_id")
        mock_launcher.get_cloud.assert_has_calls([call("aws"), call("aws")])
        mock_cloud.get_cluster.assert_has_calls(
            [call("cluster_id"), call("cluster_id")]
        )


def test_launch_run_job_detached(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
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
        mock_cloud = Mock()
        mock_launcher.run.return_value = job_status
        mock_launcher.get_cloud.return_value = mock_cloud
        mock_cloud.get_cluster.return_value = mock_cluster
        mock_cluster.get_job.return_value = job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        _ = runner.invoke(
            app,
            [
                "run",
                "--config",
                job_yaml_path,
                "--cluster",
                "cluster_id",
                "--detach",
            ],
        )
        mock_cluster.get_job.assert_not_called()
        mock_launcher.run.assert_called_once_with(job_config, "cluster_id")
        mock_launcher.get_cloud.assert_not_called()
        mock_cloud.get_cluster.assert_not_called()


def test_launch_run_job_detached_local(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        train_yaml_path = str(pathlib.Path(output_temp_dir) / "train.yaml")
        config: TrainingConfig = _create_training_config()
        config.to_yaml(train_yaml_path)
        job_yaml_path = str(pathlib.Path(output_temp_dir) / "job.yaml")
        job_config = _create_job_config(train_yaml_path)
        job_config.resources.cloud = "local"
        job_config.to_yaml(job_yaml_path)
        mock_launcher.JobConfig = JobConfig
        mock_cluster = Mock()
        job_status = JobStatus(
            id="job_id",
            cluster="local",
            name="job_name",
            status="running",
            metadata="",
            done=False,
        )
        mock_cloud = Mock()
        mock_launcher.run.return_value = job_status
        mock_launcher.get_cloud.side_effect = [mock_cloud, mock_cloud]
        mock_cloud.get_cluster.return_value = mock_cluster
        mock_cluster.get_job.return_value = JobStatus(
            id="job_id",
            cluster="local",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        _ = runner.invoke(
            app,
            [
                "run",
                "--config",
                job_yaml_path,
                "--cluster",
                "local",
            ],
        )
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])
        mock_launcher.run.assert_called_once_with(job_config, "local")
        mock_cloud.get_cluster.assert_has_calls([call("local"), call("local")])
        mock_launcher.get_cloud.assert_has_calls([call("local"), call("local")])


def test_launch_run_job_no_cluster(
    app, mock_launcher, mock_pool, mock_version, mock_confirm, mock_fetch
):
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
        mock_cloud = Mock()
        mock_launcher.run.return_value = job_status
        mock_launcher.get_cloud.return_value = mock_cloud
        mock_cloud.get_cluster.return_value = mock_cluster
        mock_cluster.get_job.return_value = job_status = JobStatus(
            id="job_id",
            cluster="cluster_id",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        with pytest.raises(
            ValueError, match="No cluster specified for the `run` action."
        ):
            res = runner.invoke(
                app,
                [
                    "run",
                    "--config",
                    job_yaml_path,
                ],
            )
            if res.exception:
                raise res.exception


def test_launch_down_with_cloud(app, mock_launcher, mock_pool):
    mock_cloud = Mock()
    mock_cluster = Mock()
    mock_launcher.get_cloud.return_value = mock_cloud
    mock_cloud.get_cluster.return_value = mock_cluster
    _ = runner.invoke(
        app,
        [
            "down",
            "--cluster",
            "cluster_id",
            "--cloud",
            "aws",
            "--log-level",
            "INFO",
        ],
    )
    mock_launcher.get_cloud.assert_called_once_with("aws")
    mock_cloud.get_cluster.assert_called_once_with("cluster_id")
    mock_cluster.down.assert_called_once()
    assert logger.level == logging.INFO


def test_launch_down_no_cloud(app, mock_launcher, mock_pool):
    mock_cloud1 = Mock()
    mock_cluster1 = Mock()
    mock_cloud2 = Mock()
    mock_launcher.get_cloud.side_effect = [mock_cloud1, mock_cloud2]
    mock_cloud1.get_cluster.return_value = mock_cluster1
    mock_cloud2.get_cluster.return_value = None
    mock_launcher.which_clouds.return_value = ["aws", "foo"]
    _ = runner.invoke(
        app,
        [
            "down",
            "--cluster",
            "cluster_id",
        ],
    )
    mock_launcher.get_cloud.assert_has_calls([call("aws"), call("foo")])
    mock_cloud1.get_cluster.assert_called_once_with("cluster_id")
    mock_cloud2.get_cluster.assert_called_once_with("cluster_id")
    mock_cluster1.down.assert_called_once()


def test_launch_down_multiple_clusters(app, mock_launcher, mock_pool):
    mock_cloud1 = Mock()
    mock_cluster1 = Mock()
    mock_cloud2 = Mock()
    mock_cluster2 = Mock()
    mock_launcher.get_cloud.side_effect = [mock_cloud1, mock_cloud2]
    mock_cloud1.get_cluster.return_value = mock_cluster1
    mock_cloud2.get_cluster.return_value = mock_cluster2
    mock_launcher.which_clouds.return_value = ["aws", "foo"]
    _ = runner.invoke(
        app,
        [
            "down",
            "--cluster",
            "cluster_id",
        ],
    )
    mock_launcher.get_cloud.assert_has_calls([call("aws"), call("foo")])
    mock_cloud1.get_cluster.assert_called_once_with("cluster_id")
    mock_cloud2.get_cluster.assert_called_once_with("cluster_id")
    mock_cluster1.down.assert_not_called()
    mock_cluster2.down.assert_not_called()


def test_launch_down_no_clusters(app, mock_launcher, mock_pool):
    mock_cloud1 = Mock()
    mock_cloud2 = Mock()
    mock_launcher.get_cloud.side_effect = [mock_cloud1, mock_cloud2]
    mock_cloud1.get_cluster.return_value = None
    mock_cloud2.get_cluster.return_value = None
    mock_launcher.which_clouds.return_value = ["aws", "foo"]
    _ = runner.invoke(
        app,
        [
            "down",
            "--cluster",
            "cluster_id",
        ],
    )
    mock_launcher.get_cloud.assert_has_calls([call("aws"), call("foo")])
    mock_cloud1.get_cluster.assert_called_once_with("cluster_id")
    mock_cloud2.get_cluster.assert_called_once_with("cluster_id")


def test_launch_stop_with_cloud(app, mock_launcher, mock_pool):
    mock_cloud = Mock()
    mock_cluster = Mock()
    mock_launcher.get_cloud.return_value = mock_cloud
    mock_cloud.get_cluster.return_value = mock_cluster
    _ = runner.invoke(
        app,
        [
            "stop",
            "--cluster",
            "cluster_id",
            "--cloud",
            "aws",
            "--log-level",
            "ERROR",
        ],
    )
    mock_launcher.get_cloud.assert_called_once_with("aws")
    mock_cloud.get_cluster.assert_called_once_with("cluster_id")
    mock_cluster.stop.assert_called_once()
    assert logger.level == logging.ERROR


def test_launch_stop_no_cloud(app, mock_launcher, mock_pool):
    mock_cloud1 = Mock()
    mock_cluster1 = Mock()
    mock_cloud2 = Mock()
    mock_launcher.get_cloud.side_effect = [mock_cloud1, mock_cloud2]
    mock_cloud1.get_cluster.return_value = mock_cluster1
    mock_cloud2.get_cluster.return_value = None
    mock_launcher.which_clouds.return_value = ["aws", "foo"]
    _ = runner.invoke(
        app,
        [
            "stop",
            "--cluster",
            "cluster_id",
        ],
    )
    mock_launcher.get_cloud.assert_has_calls([call("aws"), call("foo")])
    mock_cloud1.get_cluster.assert_called_once_with("cluster_id")
    mock_cloud2.get_cluster.assert_called_once_with("cluster_id")
    mock_cluster1.stop.assert_called_once()


def test_launch_stop_multiple_clusters(app, mock_launcher, mock_pool):
    mock_cloud1 = Mock()
    mock_cluster1 = Mock()
    mock_cloud2 = Mock()
    mock_cluster2 = Mock()
    mock_launcher.get_cloud.side_effect = [mock_cloud1, mock_cloud2]
    mock_cloud1.get_cluster.return_value = mock_cluster1
    mock_cloud2.get_cluster.return_value = mock_cluster2
    mock_launcher.which_clouds.return_value = ["aws", "foo"]
    _ = runner.invoke(
        app,
        [
            "stop",
            "--cluster",
            "cluster_id",
        ],
    )
    mock_launcher.get_cloud.assert_has_calls([call("aws"), call("foo")])
    mock_cloud1.get_cluster.assert_called_once_with("cluster_id")
    mock_cloud2.get_cluster.assert_called_once_with("cluster_id")
    mock_cluster1.stop.assert_not_called()
    mock_cluster2.stop.assert_not_called()


def test_launch_stop_no_clusters(app, mock_launcher, mock_pool):
    mock_cloud1 = Mock()
    mock_cloud2 = Mock()
    mock_launcher.get_cloud.side_effect = [mock_cloud1, mock_cloud2]
    mock_cloud1.get_cluster.return_value = None
    mock_cloud2.get_cluster.return_value = None
    mock_launcher.which_clouds.return_value = ["aws", "foo"]
    _ = runner.invoke(
        app,
        [
            "stop",
            "--cluster",
            "cluster_id",
        ],
    )
    mock_launcher.get_cloud.assert_has_calls([call("aws"), call("foo")])
    mock_cloud1.get_cluster.assert_called_once_with("cluster_id")
    mock_cloud2.get_cluster.assert_called_once_with("cluster_id")


def test_launch_cancel_success(app, mock_launcher, mock_pool):
    _ = runner.invoke(
        app,
        [
            "cancel",
            "--cloud",
            "cloud",
            "--cluster",
            "cluster",
            "--id",
            "job",
            "--log-level",
            "DEBUG",
        ],
    )
    mock_launcher.cancel.assert_called_once()
    assert logger.level == logging.DEBUG


def test_launch_which_success(app, mock_launcher, mock_pool):
    _ = runner.invoke(
        app,
        [
            "which",
            "-log",
            "INFO",
        ],
    )
    mock_launcher.which_clouds.assert_called_once()
    assert logger.level == logging.INFO


def test_launch_status_success(app, mock_launcher, mock_pool):
    _ = runner.invoke(
        app,
        [
            "status",
            "--cloud",
            "cloud",
            "--cluster",
            "cluster",
            "--id",
            "job",
            "-log",
            "DEBUG",
        ],
    )
    mock_launcher.status.assert_has_calls(
        [call(cloud="cloud", cluster="cluster", id="job")]
    )
    assert logger.level == logging.DEBUG


def test_launch_status_cluster_no_jobs(app, mock_launcher, mock_pool):
    mock_cluster = Mock()
    mock_cluster.name.return_value = "cluster_id"
    mock_cluster.get_jobs.return_value = []
    mock_cloud = Mock()
    mock_launcher.get_cloud.return_value = mock_cloud
    mock_launcher.status.return_value = {"cloud_id": []}
    mock_cloud.list_clusters.return_value = [mock_cluster]
    result = runner.invoke(
        app,
        [
            "status",
        ],
    )
    mock_launcher.status.assert_has_calls([call(cloud=None, cluster=None, id=None)])
    assert "Cloud: cloud_id" in result.stdout
    assert "Cluster: cluster_id" in result.stdout
    assert "No matching jobs found." in result.stdout
