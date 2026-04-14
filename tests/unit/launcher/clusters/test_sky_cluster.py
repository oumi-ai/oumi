from unittest.mock import ANY, Mock

import pytest
import sky.exceptions

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobState, JobStatus
from oumi.launcher.clients.sky_client import SkyClient
from oumi.launcher.clusters.sky_cluster import SkyCluster


#
# Fixtures
#
@pytest.fixture
def mock_sky_client():
    client = Mock(spec=SkyClient)
    client.get_cluster_hourly_price.return_value = None
    yield client


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
            "status": "JobStatus.RUNNING",
        },
        {
            "job_id": "myjob",
            "job_name": "some name",
            "status": "JobStatus.RUNNING",
        },
        {
            "job_id": "myjob3",
            "job_name": "some name",
            "status": "JobStatus.RUNNING",
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
            "status": "JobStatus.RUNNING",
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
            "status": "JobStatus.RUNNING",
        },
        {
            "job_id": "myjob",
            "job_name": "r",
            "status": "JobStatus.CANCELLED",
        },
        {
            "job_id": "myjob3",
            "job_name": "so",
            "status": "JobStatus.FAILED",
        },
    ]
    jobs = cluster.get_jobs()
    mock_sky_client.queue.assert_called_once_with("mycluster")
    expected_jobs = [
        JobStatus(
            id="myjob2",
            name="some name",
            status="JobStatus.RUNNING",
            metadata="",
            cluster="mycluster",
            done=False,
            state=JobState.RUNNING,
        ),
        JobStatus(
            id="myjob",
            name="r",
            status="JobStatus.CANCELLED",
            metadata="",
            cluster="mycluster",
            done=True,
            state=JobState.CANCELLED,
        ),
        JobStatus(
            id="myjob3",
            name="so",
            status="JobStatus.FAILED",
            metadata="",
            cluster="mycluster",
            done=True,
            state=JobState.FAILED,
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


def test_sky_cluster_get_jobs_down_empty(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.side_effect = sky.exceptions.ClusterNotUpError(
        "foo", None, None
    )
    jobs = cluster.get_jobs()
    mock_sky_client.queue.assert_called_once_with("mycluster")
    expected_jobs = []
    assert jobs == expected_jobs


def test_sky_cluster_cancel_job(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.return_value = [
        {
            "job_id": "myjobid",
            "job_name": "some name",
            "status": "JobStatus.FAILED",
        }
    ]
    job_status = cluster.cancel_job("myjobid")
    expected_status = JobStatus(
        id="myjobid",
        name="some name",
        status="JobStatus.FAILED",
        metadata="",
        cluster="mycluster",
        done=True,
        state=JobState.FAILED,
    )
    mock_sky_client.cancel.assert_called_once_with("mycluster", "myjobid")
    assert job_status == expected_status


def test_sky_cluster_cancel_job_fails(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.queue.return_value = [
        {
            "job_id": "wrong_job",
            "job_name": "some name",
            "status": "JobStatus.FAILED",
        }
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.cancel_job("myjobid")


def test_sky_cluster_run_job(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    mock_sky_client.exec.return_value = "new_job_id"
    mock_sky_client.queue.return_value = [
        {
            "job_id": "new_job_id",
            "job_name": "some name",
            "status": "JobStatus.SUBMITTED",
        }
    ]
    expected_status = JobStatus(
        id="new_job_id",
        name="some name",
        status="JobStatus.SUBMITTED",
        metadata="",
        cluster="mycluster",
        done=False,
        state=JobState.PENDING,
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
            "status": "JobStatus.SUBMITTED",
        }
    ]
    with pytest.raises(RuntimeError):
        _ = cluster.run_job(_get_default_job("gcp"))


def test_sky_cluster_down(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    cluster.down()
    mock_sky_client.down.assert_called_once_with("mycluster")


def test_sky_cluster_stop(mock_sky_client):
    cluster = SkyCluster("mycluster", mock_sky_client)
    cluster.stop()
    mock_sky_client.stop.assert_called_once_with("mycluster")


def test_convert_sky_job_populates_cost_per_hour(mock_sky_client):
    """Test that cost_per_hour is populated from cluster status handle."""
    cluster = SkyCluster("test-cluster", mock_sky_client)
    mock_sky_client.get_cluster_hourly_price.return_value = 3.14
    sky_job = {
        "job_id": 1,
        "job_name": "test-job",
        "status": "JobStatus.RUNNING",
    }
    mock_sky_client.queue.return_value = [sky_job]
    jobs = cluster.get_jobs()

    assert len(jobs) == 1
    assert jobs[0].cost_per_hour == 3.14
    mock_sky_client.get_cluster_hourly_price.assert_called_once_with("test-cluster")


def test_convert_sky_job_populates_start_at_and_end_at(mock_sky_client):
    """Test that start_at and end_at are extracted from sky job dict."""
    cluster = SkyCluster("test-cluster", mock_sky_client)
    sky_job = {
        "job_id": 1,
        "job_name": "test-job",
        "status": "JobStatus.SUCCEEDED",
        "start_at": 1700000000.0,
        "end_at": 1700003600.0,
    }
    mock_sky_client.queue.return_value = [sky_job]
    jobs = cluster.get_jobs()

    assert len(jobs) == 1
    assert jobs[0].start_at == 1700000000.0
    assert jobs[0].end_at == 1700003600.0


def test_convert_sky_job_handles_missing_timestamps(mock_sky_client):
    """Test that missing start_at/end_at default to None."""
    cluster = SkyCluster("test-cluster", mock_sky_client)
    sky_job = {
        "job_id": 1,
        "job_name": "test-job",
        "status": "JobStatus.RUNNING",
    }
    mock_sky_client.queue.return_value = [sky_job]
    jobs = cluster.get_jobs()

    assert len(jobs) == 1
    assert jobs[0].start_at is None
    assert jobs[0].end_at is None
