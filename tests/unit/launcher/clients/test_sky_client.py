import os
from typing import Optional
from unittest.mock import ANY, Mock, call, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobStatus
from oumi.launcher.clients.sky_client import (
    SkyClient,
    _convert_job_to_task,
    _get_use_spot_vm_override,
)


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
        accelerators="A100-80GB",
        cpus="4",
        memory="64",
        instance_type=None,
        use_spot=True,
        disk_size=512,
        disk_tier="low",
        image_id="docker://ubuntu:latest",
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


def test_convert_job_to_task(
    mock_sky_data_storage,
):
    with patch.dict(os.environ, {"OUMI_USE_SPOT_VM": "nonspot"}, clear=True):
        with patch("sky.Resources") as mock_resources:
            with patch("sky.clouds.GCP") as mock_cloud:
                mock_gcp = Mock()
                mock_cloud.return_value = mock_gcp
                with patch("sky.Task") as mock_task_cls:
                    mock_task = Mock()
                    mock_task_cls.return_value = mock_task
                    job = _get_default_job("gcp")
                    _ = _convert_job_to_task(job)
                    mock_resources.assert_has_calls(
                        [
                            call(
                                cloud=mock_gcp,
                                instance_type=job.resources.instance_type,
                                cpus=job.resources.cpus,
                                memory=job.resources.memory,
                                accelerators=job.resources.accelerators,
                                use_spot=False,
                                region=job.resources.region,
                                zone=job.resources.zone,
                                disk_size=job.resources.disk_size,
                                disk_tier=job.resources.disk_tier,
                                image_id=job.resources.image_id,
                            )
                        ]
                    )
                    mock_task_cls.assert_has_calls(
                        [
                            call(
                                name=job.name,
                                setup=job.setup,
                                run=job.run,
                                envs=job.envs,
                                workdir=job.working_dir,
                                num_nodes=job.num_nodes,
                            )
                        ]
                    )
                    mock_task.set_file_mounts.assert_called_once()
                    mock_task.set_storage_mounts.assert_called_once()
                    mock_task.set_resources.assert_called_once()


@pytest.mark.parametrize(
    "env_var_use_spot_vm,expected_use_spot_vm",
    [
        (None, None),
        ("spot", True),
        ("preemptable", True),
        ("nonspot", False),
        ("non-preemptible", False),
    ],
)
def test_get_use_spot_vm_override(
    env_var_use_spot_vm: Optional[str], expected_use_spot_vm: bool
):
    if env_var_use_spot_vm is not None:
        with patch.dict(
            os.environ, {"OUMI_USE_SPOT_VM": env_var_use_spot_vm}, clear=True
        ):
            assert _get_use_spot_vm_override() == expected_use_spot_vm
    else:
        with patch.dict(os.environ, {}, clear=True):
            assert _get_use_spot_vm_override() is None


def test_sky_client_launch(
    mock_sky_data_storage,
):
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


def test_sky_client_launch_no_stop(
    mock_sky_data_storage,
):
    with patch("sky.launch") as mock_launch:
        job = _get_default_job("runpod")
        job.resources.region = "ca"
        job.resources.disk_tier = "best"
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
            idle_minutes_to_autostop=None,
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
