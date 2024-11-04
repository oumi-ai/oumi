from unittest.mock import Mock, call, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobStatus
from oumi.launcher.clients.local_client import LocalClient
from oumi.launcher.clusters.local_cluster import LocalCluster


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
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="",
            done=False,
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
        done=False,
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
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
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
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="mycluster",
            done=False,
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
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="name",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="name",
            done=False,
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


def test_local_cluster_cancel_job(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.return_value = [
        JobStatus(
            id="myjob",
            name="some name",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
        JobStatus(
            id="final job",
            name="name3",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
    ]
    job_status = cluster.cancel_job("job2")
    expected_status = JobStatus(
        id="job2",
        name="some",
        status="running",
        metadata="",
        cluster="name",
        done=False,
    )
    mock_local_client.cancel.assert_called_once_with(
        "job2",
    )
    assert job_status == expected_status


def test_local_cluster_cancel_job_fails(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.return_value = [
        JobStatus(
            id="job2",
            name="some",
            status="running",
            metadata="",
            cluster="debug.name",
            done=False,
        ),
    ]
    with pytest.raises(RuntimeError, match="Job myjobid not found."):
        _ = cluster.cancel_job("myjobid")


def test_local_cluster_run_job(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.submit_job.return_value = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="mycluster",
        done=False,
    )
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="name",
        done=False,
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
        done=False,
    )
    expected_status = JobStatus(
        id="1234",
        name="some name",
        status="queued",
        metadata="",
        cluster="name",
        done=False,
    )
    job = _get_default_job("local")
    job.name = None
    with patch("oumi.launcher.clusters.local_cluster.uuid") as mock_uuid:
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
                done=False,
            ),
            JobStatus(
                id="job2",
                name="some",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
        ],
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="job2",
                name="some",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
        ],
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="job2",
                name="some",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
        ],
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="job2",
                name="some",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
        ],
    ]
    cluster.down()
    mock_local_client.cancel.assert_has_calls(
        [call("myjob"), call("job2"), call("final job")]
    )
    mock_local_client.list_jobs.assert_has_calls([call(), call(), call(), call()])
    # Nothing to assert, this method is a no-op.


def test_local_cluster_stop(mock_local_client):
    cluster = LocalCluster("name", mock_local_client)
    mock_local_client.list_jobs.side_effect = [
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="job2",
                name="some",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
        ],
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="job2",
                name="some",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
        ],
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="job2",
                name="some",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="running",
                metadata="",
                cluster="",
                done=False,
            ),
        ],
        [
            JobStatus(
                id="myjob",
                name="some name",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="job2",
                name="some",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
            JobStatus(
                id="final job",
                name="name3",
                status="CANCELED",
                metadata="",
                cluster="",
                done=False,
            ),
        ],
    ]
    cluster.stop()
    mock_local_client.cancel.assert_has_calls(
        [call("myjob"), call("job2"), call("final job")]
    )
    mock_local_client.list_jobs.assert_has_calls([call(), call(), call(), call()])
    # Nothing to assert, this method is a no-op.
