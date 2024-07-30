from unittest.mock import ANY, Mock

import pytest

from lema.core.types.base_cluster import JobStatus
from lema.core.types.configs import JobConfig
from lema.core.types.params.node_params import DiskTier, NodeParams, StorageMount
from lema.launcher.clients.polaris_client import PolarisClient
from lema.launcher.clusters.polaris_cluster import PolarisCluster


#
# Fixtures
#
@pytest.fixture
def mock_polaris_client():
    yield Mock(spec=PolarisClient)


def _get_default_job(cloud: str) -> JobConfig:
    resources = NodeParams(
        cloud=cloud,
        region="us-central1",
        zone=None,
        accelerators="A100-80",
        cpus=4,
        memory=64,
        instance_type=None,
        use_spot=True,
        disk_size=512,
        disk_tier=DiskTier.LOW,
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
def test_sky_cluster_name(mock_polaris_client):
    cluster = PolarisCluster("demand.einstein", mock_polaris_client)
    assert cluster.name() == "demand.einstein"

    cluster = PolarisCluster("debug.einstein", mock_polaris_client)
    assert cluster.name() == "debug.einstein"

    cluster = PolarisCluster("debug-scaling.einstein", mock_polaris_client)
    assert cluster.name() == "debug-scaling.einstein"

    cluster = PolarisCluster("preemptable.einstein", mock_polaris_client)
    assert cluster.name() == "preemptable.einstein"

    cluster = PolarisCluster("prod.einstein", mock_polaris_client)
    assert cluster.name() == "prod.einstein"


def test_sky_cluster_invalid_name(mock_polaris_client):
    with pytest.raises(ValueError):
        PolarisCluster("einstein", mock_polaris_client)


def test_sky_cluster_invalid_queue(mock_polaris_client):
    with pytest.raises(ValueError):
        PolarisCluster("albert.einstein", mock_polaris_client)


def test_sky_cluster_get_job_valid_id(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    mock_polaris_client.queue.return_value = [
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
    mock_polaris_client.queue.assert_called_once_with("mycluster")
    assert job is not None
    assert job.id == "myjob"


def test_sky_cluster_get_job_invalid_id_empty(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    mock_polaris_client.queue.return_value = []
    job = cluster.get_job("myjob")
    mock_polaris_client.queue.assert_called_once_with("mycluster")
    assert job is None


def test_sky_cluster_get_job_invalid_id_nonempty(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    mock_polaris_client.queue.return_value = [
        {
            "job_id": "wrong_id",
            "job_name": "some name",
            "status": "running",
        }
    ]
    job = cluster.get_job("myjob")
    mock_polaris_client.queue.assert_called_once_with("mycluster")
    assert job is None


def test_sky_cluster_get_jobs_nonempty(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    mock_polaris_client.queue.return_value = [
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
    mock_polaris_client.queue.assert_called_once_with("mycluster")
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


def test_sky_cluster_get_jobs_empty(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    mock_polaris_client.queue.return_value = []
    jobs = cluster.get_jobs()
    mock_polaris_client.queue.assert_called_once_with("mycluster")
    expected_jobs = []
    assert jobs == expected_jobs


def test_sky_cluster_stop_job(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    mock_polaris_client.queue.return_value = [
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
    mock_polaris_client.cancel.assert_called_once_with("mycluster", "myjobid")
    assert job_status == expected_status


def test_sky_cluster_stop_job_fails(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    mock_polaris_client.queue.return_value = [
        {
            "job_id": "wrong_job",
            "job_name": "some name",
            "status": "failed",
        }
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.stop_job("myjobid")


def test_sky_cluster_run_job(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    mock_polaris_client.exec.return_value = "new_job_id"
    mock_polaris_client.queue.return_value = [
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
    mock_polaris_client.exec.assert_called_once_with(ANY, "mycluster")
    mock_polaris_client.queue.assert_called_once_with("mycluster")
    assert job_status == expected_status


def test_sky_cluster_run_job_fails(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    mock_polaris_client.exec.return_value = "new_job_id"
    mock_polaris_client.queue.return_value = [
        {
            "job_id": "wrong_id",
            "job_name": "some name",
            "status": "queued",
        }
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.run_job(_get_default_job("gcp"))


def test_sky_cluster_down(mock_polaris_client):
    cluster = PolarisCluster("mycluster", mock_polaris_client)
    cluster.down()
    mock_polaris_client.down.assert_called_once_with("mycluster")
