from unittest.mock import Mock, call, patch

import pytest

from lema.core.types.base_cluster import JobStatus
from lema.core.types.configs import JobConfig
from lema.core.types.params.job_resources import JobResources, StorageMount
from lema.launcher.clients.local_client import LocalClient
from lema.launcher.clusters.local_cluster import LocalCluster


#
# Fixtures
#
@pytest.fixture
def mock_local_client():
    yield Mock(spec=LocalClient)


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
        file_mounts={
            "~/home/remote/path.bar": "~/local/path.bar",
            "~/home/remote/path2.txt": "~/local/path2.txt",
        },
        storage_mounts={
            "~/home/remote/path/gcs/": StorageMount(
                source="gs://mybucket/", store="gcs"
            )
        },
        setup=(
            "#PBS -o some/log \n#PBE -l wow\n#PBS -e run/log\n"
            "pip install -r requirements.txt"
        ),
        run="./hello_world.sh",
    )


#
# Tests
#
def test_local_cluster_name(mock_local_client):
    cluster = LocalCluster("local name", mock_local_client)
    assert cluster.name() == "local name"

    cluster = LocalCluster("", mock_local_client)
    assert cluster.name() == ""


def test_local_cluster_get_job_valid_id(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="",
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="",
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="",
        ),
    ]
    job = cluster.get_job("myjob")
    mock_local_client.list_jobs.assert_called_once()
    assert job == JobStatus(
        id="myjob",
        name="some name",
        status="running",
        metadata="",
        cluster="name",
    )


def test_local_cluster_get_job_invalid_id_empty(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.return_value = []
    job = cluster.get_job("myjob")
    mock_local_client.list_jobs.assert_called_once()
    assert job is None


def test_local_cluster_get_job_invalid_id_nonempty(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="mycluster",
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
        ),
    ]
    job = cluster.get_job("wrong job")
    mock_local_client.list_jobs.assert_called_once()
    assert job is None


def test_local_cluster_get_jobs_nonempty(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="mycluster",
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
        ),
    ]
    jobs = cluster.get_jobs()
    mock_local_client.list_jobs.assert_called_once()
    expected_jobs = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="name",
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="name",
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="name",
        ),
    ]
    assert jobs == expected_jobs


def test_local_cluster_get_jobs_empty(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.return_value = []
    jobs = cluster.get_jobs()
    mock_local_client.list_jobs.assert_called_once()
    expected_jobs = []
    assert jobs == expected_jobs


def test_local_cluster_stop_job(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="debug.name",
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug.name",
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="debug.name",
        ),
    ]
    job_status = cluster.stop_job("job2")
    expected_status = JobStatus(
        id="job2",
        name="some",
        status="running",
        metadata="",
        cluster="name",
    )
    mock_local_client.cancel.assert_called_once_with(
        "job2",
    )
    assert job_status == expected_status


def test_local_cluster_stop_job_fails(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.return_value = [
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug.name",
        ),
    ]
    with pytest.raises(RuntimeError, match="Job myjobid not found."):
        _ = cluster.stop_job("myjobid")


def test_local_cluster_run_job(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.submit_job.return_value = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="mycluster",
    )
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="name",
    )
    expected_job = _get_default_job("local")
    job_status = cluster.run_job(expected_job)
    mock_local_client.submit_job.assert_called_once_with(
        expected_job,
    )
    assert job_status == expected_status


def test_local_cluster_run_job_no_name(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.submit_job.return_value = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="mycluster",
    )
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="name",
    )
    job = _get_default_job("local")
    job.name = None
    with patch("lema.launcher.clusters.local_cluster.uuid") as mock_uuid:
        mock_hex = Mock()
        mock_hex.hex = "1-2-3"
        mock_uuid.uuid1.return_value = mock_hex
        job_status = cluster.run_job(job)
    expected_job = _get_default_job("local")
    expected_job.name = "1-2-3"
    mock_local_client.submit_job.assert_called_once_with(
        expected_job,
    )
    assert job_status == expected_status


def test_local_cluster_down(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.side_effect = [
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="running",
                metadata="",
                cluster="",
            ),
            JobStatus(
                id="job2",
                name="some",
                status="running",
                metadata="",
                cluster="",
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="running",
                metadata="",
                cluster="",
            ),
        ],
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="CANCELED",
                metadata="",
                cluster="",
            ),
            JobStatus(
                id="job2",
                name="some",
                status="running",
                metadata="",
                cluster="",
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="running",
                metadata="",
                cluster="",
            ),
        ],
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="CANCELED",
                metadata="",
                cluster="",
            ),
            JobStatus(
                id="job2",
                name="some",
                status="CANCELED",
                metadata="",
                cluster="",
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="running",
                metadata="",
                cluster="",
            ),
        ],
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="CANCELED",
                metadata="",
                cluster="",
            ),
            JobStatus(
                id="job2",
                name="some",
                status="CANCELED",
                metadata="",
                cluster="",
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="CANCELED",
                metadata="",
                cluster="",
            ),
        ],
    ]
    cluster.down()
    mock_local_client.cancel.assert_has_calls(
        [call("myjob"), call("job2"), call("final job")]
    )
    mock_local_client.list_jobs.assert_has_calls([call(), call(), call(), call()])
    # Nothing to assert, this method is a no-op.
