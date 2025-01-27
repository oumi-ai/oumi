from unittest.mock import ANY, Mock, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobStatus
from oumi.launcher.clients.sky_client import SkyClient


#
# Fixtures
#
@pytest.fixture
def mock_sky_data_storage():
    with patch("sky.data.Storage") as mock_storage:
        yield mock_storage


def _get_default_job(cloud: str) -> JobConfig:
    resources = JobResources(
        cloud=cloud,
        region="us-central1",
        zone=None,
        accelerators="A100-80",
        cpus="4",
        memory="64",
        instance_type=None,
        use_spot=True,
        disk_size=512,
        disk_tier="low",
    )
    return JobConfig(
        name="myjob",
        user="user",
        working_dir="./",
        num_nodes=2,
        resources=resources,
        envs={"var1": "val1"},
        file_mounts={},
        storage_mounts={
            "~/home/remote/path/gcs/": StorageMount(
                source="gs://mybucket/", store="gcs"
            )
        },
        setup="pip install -r requirements.txt",
        run="./hello_world.sh",
    )


#
# Tests
#
def test_sky_client_gcp_name():
    client = SkyClient()
    assert client.SupportedClouds.GCP.value == "gcp"


def test_sky_client_runpod_name():
    client = SkyClient()
    assert client.SupportedClouds.RUNPOD.value == "runpod"


def test_sky_client_lambda_name():
    client = SkyClient()
    assert client.SupportedClouds.LAMBDA.value == "lambda"


def test_sky_client_aws_name():
    client = SkyClient()
    assert client.SupportedClouds.AWS.value == "aws"


def test_sky_client_azure_name():
    client = SkyClient()
    assert client.SupportedClouds.AZURE.value == "azure"


def test_sky_client_launch(mock_sky_data_storage):
    with patch("sky.launch") as mock_launch:
        job = _get_default_job("gcp")
        mock_resource_handle = Mock()
        mock_resource_handle.cluster_name = "mycluster"
        mock_launch.return_value = (1, mock_resource_handle)
        client = SkyClient()
        job_status = client.launch(job)
        expected_job_status = JobStatus(
            name="",
            id="1",
            cluster="mycluster",
            status="",
            metadata="",
            done=False,
        )
        assert job_status == expected_job_status
        mock_launch.assert_called_once_with(
            ANY,
            cluster_name=None,
            detach_run=True,
            idle_minutes_to_autostop=60,
        )


def test_sky_client_launch_kwarg(mock_sky_data_storage):
    with patch("sky.launch") as mock_launch:
        job = _get_default_job("gcp")
        mock_resource_handle = Mock()
        mock_resource_handle.cluster_name = "mycluster"
        mock_launch.return_value = (1, mock_resource_handle)
        client = SkyClient()
        job_status = client.launch(job, idle_minutes_to_autostop=None)
        expected_job_status = JobStatus(
            name="",
            id="1",
            cluster="mycluster",
            status="",
            metadata="",
            done=False,
        )
        assert job_status == expected_job_status
        mock_launch.assert_called_once_with(
            ANY,
            cluster_name=None,
            detach_run=True,
            idle_minutes_to_autostop=None,
        )


def test_sky_client_launch_kwarg_value(mock_sky_data_storage):
    with patch("sky.launch") as mock_launch:
        job = _get_default_job("gcp")
        mock_resource_handle = Mock()
        mock_resource_handle.cluster_name = "mycluster"
        mock_launch.return_value = (1, mock_resource_handle)
        client = SkyClient()
        job_status = client.launch(job, idle_minutes_to_autostop=45)
        expected_job_status = JobStatus(
            name="",
            id="1",
            cluster="mycluster",
            status="",
            metadata="",
            done=False,
        )
        assert job_status == expected_job_status
        mock_launch.assert_called_once_with(
            ANY,
            cluster_name=None,
            detach_run=True,
            idle_minutes_to_autostop=45,
        )


def test_sky_client_launch_unused_kwarg(mock_sky_data_storage):
    with patch("sky.launch") as mock_launch:
        job = _get_default_job("gcp")
        mock_resource_handle = Mock()
        mock_resource_handle.cluster_name = "mycluster"
        mock_launch.return_value = (1, mock_resource_handle)
        client = SkyClient()
        job_status = client.launch(job, foo=None)
        expected_job_status = JobStatus(
            name="",
            id="1",
            cluster="mycluster",
            status="",
            metadata="",
            done=False,
        )
        assert job_status == expected_job_status
        mock_launch.assert_called_once_with(
            ANY,
            cluster_name=None,
            detach_run=True,
            idle_minutes_to_autostop=60,
        )


def test_sky_client_launch_with_cluster_name(mock_sky_data_storage):
    with patch("sky.launch") as mock_launch:
        job = _get_default_job("gcp")
        mock_resource_handle = Mock()
        mock_resource_handle.cluster_name = "cluster_name"
        mock_launch.return_value = (1, mock_resource_handle)
        client = SkyClient()
        job_status = client.launch(job, "cluster_name")
        expected_job_status = JobStatus(
            name="",
            id="1",
            cluster="cluster_name",
            status="",
            metadata="",
            done=False,
        )
        assert job_status == expected_job_status
        mock_launch.assert_called_once_with(
            ANY,
            cluster_name="cluster_name",
            detach_run=True,
            idle_minutes_to_autostop=60,
        )


def test_sky_client_status():
    with patch("sky.status") as mock_status:
        mock_status.return_value = [{"name": "mycluster"}]
        client = SkyClient()
        status = client.status()
        mock_status.assert_called_once()
        assert status == [{"name": "mycluster"}]


def test_sky_client_queue():
    with patch("sky.queue") as mock_queue:
        mock_queue.return_value = [{"name": "myjob"}]
        client = SkyClient()
        queue = client.queue("mycluster")
        mock_queue.assert_called_once_with("mycluster")
        assert queue == [{"name": "myjob"}]


def test_sky_client_cancel():
    with patch("sky.cancel") as mock_cancel:
        client = SkyClient()
        client.cancel("mycluster", "1")
        mock_cancel.assert_called_once_with("mycluster", 1)


def test_sky_client_exec(mock_sky_data_storage):
    with patch("sky.exec") as mock_exec:
        mock_resource_handle = Mock()
        mock_exec.return_value = (1, mock_resource_handle)
        client = SkyClient()
        job = _get_default_job("gcp")
        job_id = client.exec(job, "mycluster")
        mock_exec.assert_called_once_with(ANY, "mycluster", detach_run=True)
        assert job_id == "1"


def test_sky_client_exec_runtime_error(mock_sky_data_storage):
    with pytest.raises(RuntimeError):
        with patch("sky.exec") as mock_exec:
            mock_resource_handle = Mock()
            mock_exec.return_value = (None, mock_resource_handle)
            client = SkyClient()
            job = _get_default_job("gcp")
            _ = client.exec(job, "mycluster")


def test_sky_client_down():
    with patch("sky.down") as mock_down:
        client = SkyClient()
        client.down("mycluster")
        mock_down.assert_called_once_with("mycluster")


def test_sky_client_stop():
    with patch("sky.stop") as mock_stop:
        client = SkyClient()
        client.stop("mycluster")
        mock_stop.assert_called_once_with("mycluster")
