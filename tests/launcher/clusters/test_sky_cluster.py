from unittest.mock import ANY, Mock

import pytest

from lema.core.types.base_cluster import JobStatus
from lema.core.types.configs import JobConfig
from lema.core.types.params.job_resources import JobResources, StorageMount
from lema.launcher.clients.sky_client import SkyClient
from lema.launcher.clusters.sky_cluster import SkyCluster


#
# Fixtures
#
@pytest.fixture
def mock_sky_client():
    yield Mock(spec=SkyClient)


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
def test_sky_cluster_name(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    assert cluster.name() == "mycluster"


def test_sky_cluster_get_job_valid_id(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.return_value = [
        {
            "job_id": "myjob2",
            "job_name": "some name",
            "status": "running",
        },
        {
            "job_id": "myjob",
            "job_name": "some name",
            "status": "running",
        },
        {
            "job_id": "myjob3",
            "job_name": "some name",
            "status": "running",
        },
    ]
    job = cluster.get_job("myjob")
    mock_sky_client.queue.assert_called_once_with("mycluster")
    assert job is not None
    assert job.id == "myjob"


def test_sky_cluster_get_job_invalid_id_empty(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.return_value = []
    job = cluster.get_job("myjob")
    mock_sky_client.queue.assert_called_once_with("mycluster")
    assert job is None


def test_sky_cluster_get_job_invalid_id_nonempty(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.return_value = [
        {
            "job_id": "wrong_id",
            "job_name": "some name",
            "status": "running",
        }
    ]
    job = cluster.get_job("myjob")
    mock_sky_client.queue.assert_called_once_with("mycluster")
    assert job is None


def test_sky_cluster_get_jobs_nonempty(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.return_value = [
        {
            "job_id": "myjob2",
            "job_name": "some name",
            "status": "running",
        },
        {
            "job_id": "myjob",
            "job_name": "r",
            "status": "stopped",
        },
        {
            "job_id": "myjob3",
            "job_name": "so",
            "status": "failed",
        },
    ]
    jobs = cluster.get_jobs()
    mock_sky_client.queue.assert_called_once_with("mycluster")
    expected_jobs = [
        JobStatus(
            id="myjob2",
            name="some name",
            status="running",
            metadata="",
            cluster="mycluster",
        ),
        JobStatus(
            id="myjob",
            name="r",
            status="stopped",
            metadata="",
            cluster="mycluster",
        ),
        JobStatus(
            id="myjob3",
            name="so",
            status="failed",
            metadata="",
            cluster="mycluster",
        ),
    ]
    assert jobs == expected_jobs


def test_sky_cluster_get_jobs_empty(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.return_value = []
    jobs = cluster.get_jobs()
    mock_sky_client.queue.assert_called_once_with("mycluster")
    expected_jobs = []
    assert jobs == expected_jobs


def test_sky_cluster_stop_job(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.return_value = [
        {
            "job_id": "myjobid",
            "job_name": "some name",
            "status": "failed",
        }
    ]
    job_status = cluster.stop_job("myjobid")
    expected_status = JobStatus(
        id="myjobid",
        name="some name",
        status="failed",
        metadata="",
        cluster="mycluster",
    )
    mock_sky_client.cancel.assert_called_once_with("mycluster", "myjobid")
    assert job_status == expected_status


def test_sky_cluster_stop_job_fails(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.return_value = [
        {
            "job_id": "wrong_job",
            "job_name": "some name",
            "status": "failed",
        }
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.stop_job("myjobid")


def test_sky_cluster_run_job(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.exec.return_value = "new_job_id"
    mock_sky_client.queue.return_value = [
        {
            "job_id": "new_job_id",
            "job_name": "some name",
            "status": "queued",
        }
    ]
    expected_status = JobStatus(
        id="new_job_id",
        name="some name",
        status="queued",
        metadata="",
        cluster="mycluster",
    )
    job_status = cluster.run_job(_get_default_job("gcp"))
    mock_sky_client.exec.assert_called_once_with(ANY, "mycluster")
    mock_sky_client.queue.assert_called_once_with("mycluster")
    assert job_status == expected_status


def test_sky_cluster_run_job_fails(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.exec.return_value = "new_job_id"
    mock_sky_client.queue.return_value = [
        {
            "job_id": "wrong_id",
            "job_name": "some name",
            "status": "queued",
        }
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.run_job(_get_default_job("gcp"))


def test_sky_cluster_down(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    cluster.down()
    mock_sky_client.down.assert_called_once_with("mycluster")
