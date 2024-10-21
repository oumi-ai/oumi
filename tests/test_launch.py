import pathlib
import tempfile
from unittest.mock import Mock, call, patch

import pytest

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
from oumi.launch import _LaunchArgs, _LauncherAction, down, launch, run, stop, which
from oumi.launcher import JobConfig, JobResources


class MockThreadPool:
    def __init__(self):
        self.mock_result = Mock()

    def apply_async(self, fn):
        fn()
        return self.mock_result


#
# Fixtures
#
@pytest.fixture
def mock_launcher():
    with patch("oumi.launch.launcher") as launcher_mock:
        yield launcher_mock


@pytest.fixture
def mock_threadpool():
    with patch("oumi.launch.ThreadPool") as threadpool_mock:
        mock_pool = MockThreadPool()
        threadpool_mock.return_value = mock_pool
        yield threadpool_mock


def _create_training_config() -> TrainingConfig:
    return TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="yahma/alpaca-cleaned",
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
        run=f"python -m oumi.launch {training_config_path}",
    )


def test_launch_launch_job(mock_launcher, mock_threadpool):
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
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])


def test_launch_launch_job_existing_cluster(mock_launcher, mock_threadpool):
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
        launch(
            _LaunchArgs(
                job=job_yaml_path, action=_LauncherAction.UP, cluster="cluster_id"
            )
        )
        mock_launcher.run.assert_called_once_with(job_config, "cluster_id")
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])


def test_launch_launch_job_detach(mock_launcher, mock_threadpool):
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
        launch(_LaunchArgs(job=job_yaml_path, action=_LauncherAction.UP, detach=True))
        mock_cluster.get_job.assert_not_called()


def test_launch_launch_job_detached_local(mock_launcher, mock_threadpool):
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
        launch(_LaunchArgs(job=job_yaml_path, action=_LauncherAction.UP, detach=True))
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])


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


def test_launch_run_job(mock_launcher, mock_threadpool):
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
        run(
            _LaunchArgs(
                job=job_yaml_path, action=_LauncherAction.RUN, cluster="cluster_id"
            )
        )
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])
        mock_launcher.run.assert_called_once_with(job_config, "cluster_id")
        mock_launcher.get_cloud.assert_called_once_with("aws")
        mock_cloud.get_cluster.assert_called_once_with("cluster_id")


def test_launch_run_job_detached(mock_launcher, mock_threadpool):
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
        run(
            _LaunchArgs(
                job=job_yaml_path,
                action=_LauncherAction.RUN,
                cluster="cluster_id",
                detach=True,
            )
        )
        mock_cluster.get_job.assert_not_called()
        mock_launcher.run.assert_called_once_with(job_config, "cluster_id")
        mock_launcher.get_cloud.assert_not_called()
        mock_cloud.get_cluster.assert_not_called()


def test_launch_run_job_detached_local(mock_launcher, mock_threadpool):
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
            cluster="local",
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
            cluster="local",
            name="job_name",
            status="done",
            metadata="",
            done=True,
        )
        run(_LaunchArgs(job=job_yaml_path, action=_LauncherAction.RUN, cluster="local"))
        mock_cluster.get_job.assert_has_calls([call("job_id"), call("job_id")])
        mock_launcher.run.assert_called_once_with(job_config, "local")
        mock_launcher.get_cloud.assert_called_once_with("aws")
        mock_cloud.get_cluster.assert_called_once_with("local")


def test_launch_run_job_no_cluster(mock_launcher, mock_threadpool):
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
            run(_LaunchArgs(job=job_yaml_path, action=_LauncherAction.RUN))


def test_launch_down_with_cloud(mock_launcher, mock_threadpool):
    mock_cloud = Mock()
    mock_cluster = Mock()
    mock_launcher.get_cloud.return_value = mock_cloud
    mock_cloud.get_cluster.return_value = mock_cluster
    down(_LaunchArgs(action=_LauncherAction.DOWN, cluster="cluster_id", cloud="aws"))
    mock_launcher.get_cloud.assert_called_once_with("aws")
    mock_cloud.get_cluster.assert_called_once_with("cluster_id")
    mock_cluster.down.assert_called_once()


def test_launch_down_no_cloud(mock_launcher, mock_threadpool):
    mock_cloud1 = Mock()
    mock_cluster1 = Mock()
    mock_cloud2 = Mock()
    mock_launcher.get_cloud.side_effect = [mock_cloud1, mock_cloud2]
    mock_cloud1.get_cluster.return_value = mock_cluster1
    mock_cloud2.get_cluster.return_value = None
    mock_launcher.which_clouds.return_value = ["aws", "foo"]
    down(
        _LaunchArgs(
            action=_LauncherAction.DOWN,
            cluster="cluster_id",
        )
    )
    mock_launcher.get_cloud.assert_has_calls([call("aws"), call("foo")])
    mock_cloud1.get_cluster.assert_called_once_with("cluster_id")
    mock_cloud2.get_cluster.assert_called_once_with("cluster_id")
    mock_cluster1.down.assert_called_once()


def test_launch_down_multiple_clusters(mock_launcher, mock_threadpool):
    mock_cloud1 = Mock()
    mock_cluster1 = Mock()
    mock_cloud2 = Mock()
    mock_cluster2 = Mock()
    mock_launcher.get_cloud.side_effect = [mock_cloud1, mock_cloud2]
    mock_cloud1.get_cluster.return_value = mock_cluster1
    mock_cloud2.get_cluster.return_value = mock_cluster2
    mock_launcher.which_clouds.return_value = ["aws", "foo"]
    down(
        _LaunchArgs(
            action=_LauncherAction.DOWN,
            cluster="cluster_id",
        )
    )
    mock_launcher.get_cloud.assert_has_calls([call("aws"), call("foo")])
    mock_cloud1.get_cluster.assert_called_once_with("cluster_id")
    mock_cloud2.get_cluster.assert_called_once_with("cluster_id")
    mock_cluster1.down.assert_not_called()
    mock_cluster2.down.assert_not_called()


def test_launch_down_no_clusters(mock_launcher, mock_threadpool):
    mock_cloud1 = Mock()
    mock_cloud2 = Mock()
    mock_launcher.get_cloud.side_effect = [mock_cloud1, mock_cloud2]
    mock_cloud1.get_cluster.return_value = None
    mock_cloud2.get_cluster.return_value = None
    mock_launcher.which_clouds.return_value = ["aws", "foo"]
    down(
        _LaunchArgs(
            action=_LauncherAction.DOWN,
            cluster="cluster_id",
        )
    )
    mock_launcher.get_cloud.assert_has_calls([call("aws"), call("foo")])
    mock_cloud1.get_cluster.assert_called_once_with("cluster_id")
    mock_cloud2.get_cluster.assert_called_once_with("cluster_id")


def test_launch_down_no_cluster_arg(mock_launcher, mock_threadpool):
    with pytest.raises(ValueError, match="No cluster specified for `down` action."):
        down(
            _LaunchArgs(
                action=_LauncherAction.DOWN,
            )
        )


def test_launch_stop_no_cluster_arg(mock_launcher, mock_threadpool):
    with pytest.raises(ValueError, match="No cluster specified for `stop` action."):
        stop(
            _LaunchArgs(
                cloud="aws",
                job_id="bar",
                action=_LauncherAction.STOP,
            )
        )


def test_launch_stop_no_cloud_arg(mock_launcher, mock_threadpool):
    with pytest.raises(ValueError, match="No cloud specified for `stop` action."):
        stop(
            _LaunchArgs(
                cluster="aws",
                job_id="bar",
                action=_LauncherAction.STOP,
            )
        )


def test_launch_stop_no_job_id_arg(mock_launcher, mock_threadpool):
    with pytest.raises(ValueError, match="No job ID specified for `stop` action."):
        stop(
            _LaunchArgs(
                cloud="aws",
                cluster="cluster",
                action=_LauncherAction.STOP,
            )
        )


def test_launch_stop_success(mock_launcher, mock_threadpool):
    stop(
        _LaunchArgs(
            cloud="cloud",
            cluster="cluster",
            job_id="job",
            action=_LauncherAction.STOP,
        )
    )
    mock_launcher.stop.assert_called_once_with("job", "cloud", "cluster")


def test_launch_which_success(mock_launcher, mock_threadpool):
    which()
    mock_launcher.which_clouds.assert_called_once()
