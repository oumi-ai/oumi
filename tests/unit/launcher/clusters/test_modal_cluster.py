"""Unit tests for ModalCluster."""

import io
from unittest.mock import Mock

import pytest

from oumi.core.configs import JobConfig, JobResources
from oumi.core.launcher import JobState, JobStatus
from oumi.launcher.clients.modal_client import ModalClient
from oumi.launcher.clusters.modal_cluster import ModalCluster, _validate_job_config


#
# Fixtures
#
@pytest.fixture
def mock_modal_client():
    """Create a mock ModalClient."""
    client = Mock(spec=ModalClient)
    return client


def _get_default_job(cloud: str = "modal") -> JobConfig:
    """Create a default JobConfig for testing."""
    resources = JobResources(
        cloud=cloud,
        region=None,
        zone=None,
        accelerators="A100:4",
        cpus="4",
        memory="64GB",
        instance_type=None,
        use_spot=False,
        disk_size=None,
    )
    return JobConfig(
        name="test-modal-job",
        user="user",
        working_dir="./",
        num_nodes=1,
        resources=resources,
        envs={"VAR1": "val1"},
        file_mounts={},
        storage_mounts={},
        setup="pip install torch",
        run="python train.py",
    )


#
# Tests for job validation
#
def test_validate_job_config_valid():
    """Test validation passes for valid config."""
    job = _get_default_job("modal")
    # Should not raise
    _validate_job_config(job)


def test_validate_job_config_missing_run():
    """Test validation fails when run script is missing."""
    job = _get_default_job("modal")
    job.run = ""

    with pytest.raises(ValueError, match="Run script must be provided"):
        _validate_job_config(job)


def test_validate_job_config_wrong_cloud():
    """Test validation fails for wrong cloud."""
    job = _get_default_job("gcp")

    with pytest.raises(ValueError, match="must be `modal`"):
        _validate_job_config(job)


def test_validate_job_config_multi_node_warning(caplog):
    """Test validation warns about multi-node."""
    job = _get_default_job("modal")
    job.num_nodes = 4

    _validate_job_config(job)
    assert "Multi-node jobs are in early preview" in caplog.text


def test_validate_job_config_region_warning(caplog):
    """Test validation warns about region specification."""
    job = _get_default_job("modal")
    job.resources.region = "us-central1"

    _validate_job_config(job)
    assert "Region specification is not supported" in caplog.text


def test_validate_job_config_spot_warning(caplog):
    """Test validation warns about spot instances."""
    job = _get_default_job("modal")
    job.resources.use_spot = True

    _validate_job_config(job)
    assert "Spot instances are not a concept" in caplog.text


#
# Tests for ModalCluster
#
def test_modal_cluster_name(mock_modal_client):
    """Test cluster name retrieval."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    assert cluster.name() == "test-cluster"


def test_modal_cluster_equality(mock_modal_client):
    """Test cluster equality."""
    cluster1 = ModalCluster("test-cluster", mock_modal_client)
    cluster2 = ModalCluster("test-cluster", mock_modal_client)
    cluster3 = ModalCluster("other-cluster", mock_modal_client)

    assert cluster1 == cluster2
    assert cluster1 != cluster3
    assert cluster1 != "not-a-cluster"


def test_modal_cluster_get_job(mock_modal_client):
    """Test getting a job from the cluster."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    expected_status = JobStatus(
        id="job-123",
        name="test-job",
        cluster="",
        status="RUNNING",
        metadata="",
        done=False,
        state=JobState.RUNNING,
    )
    mock_modal_client.get_job.return_value = expected_status

    status = cluster.get_job("job-123")

    assert status is not None
    assert status.id == "job-123"
    assert status.cluster == "test-cluster"
    mock_modal_client.get_job.assert_called_once_with("job-123")


def test_modal_cluster_get_job_not_found(mock_modal_client):
    """Test getting a non-existent job."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    mock_modal_client.get_job.return_value = None

    status = cluster.get_job("nonexistent")

    assert status is None


def test_modal_cluster_get_jobs(mock_modal_client):
    """Test listing jobs from the cluster."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    expected_jobs = [
        JobStatus(
            id="job-1",
            name="job1",
            cluster="",
            status="RUNNING",
            metadata="",
            done=False,
            state=JobState.RUNNING,
        ),
        JobStatus(
            id="job-2",
            name="job2",
            cluster="",
            status="COMPLETED",
            metadata="",
            done=True,
            state=JobState.SUCCEEDED,
        ),
    ]
    mock_modal_client.list_jobs.return_value = expected_jobs

    jobs = cluster.get_jobs()

    assert len(jobs) == 2
    assert all(j.cluster == "test-cluster" for j in jobs)


def test_modal_cluster_run_job(mock_modal_client):
    """Test running a job on the cluster."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    job = _get_default_job("modal")
    expected_status = JobStatus(
        id="modal-0-abc123",
        name="test-modal-job",
        cluster="",
        status="SUBMITTING",
        metadata="",
        done=False,
        state=JobState.PENDING,
    )
    mock_modal_client.submit_job.return_value = expected_status

    status = cluster.run_job(job)

    assert status is not None
    assert status.cluster == "test-cluster"
    mock_modal_client.submit_job.assert_called_once()


def test_modal_cluster_run_job_generates_name(mock_modal_client):
    """Test that run_job generates a name if not provided."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    job = _get_default_job("modal")
    job.name = None
    mock_modal_client.submit_job.return_value = JobStatus(
        id="modal-0-abc123",
        name="generated",
        cluster="",
        status="SUBMITTING",
        metadata="",
        done=False,
        state=JobState.PENDING,
    )

    status = cluster.run_job(job)

    # The job passed to submit_job should have a generated name
    call_args = mock_modal_client.submit_job.call_args
    submitted_job = call_args[0][0]
    assert submitted_job.name is not None
    assert "oumi-modal-" in submitted_job.name
    assert status


def test_modal_cluster_cancel_job(mock_modal_client):
    """Test cancelling a job."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    expected_status = JobStatus(
        id="job-123",
        name="test-job",
        cluster="",
        status="CANCELLED",
        metadata="",
        done=True,
        state=JobState.CANCELLED,
    )
    mock_modal_client.cancel.return_value = expected_status

    status = cluster.cancel_job("job-123")

    assert status.state == JobState.CANCELLED
    assert status.cluster == "test-cluster"
    mock_modal_client.cancel.assert_called_once_with("job-123")


def test_modal_cluster_cancel_job_not_found(mock_modal_client):
    """Test cancelling a non-existent job."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    mock_modal_client.cancel.return_value = None

    with pytest.raises(RuntimeError, match="not found"):
        cluster.cancel_job("nonexistent")


def test_modal_cluster_stop(mock_modal_client):
    """Test stopping the cluster cancels all jobs."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    jobs = [
        JobStatus(
            id="job-1",
            name="job1",
            cluster="test-cluster",
            status="RUNNING",
            metadata="",
            done=False,
            state=JobState.RUNNING,
        ),
        JobStatus(
            id="job-2",
            name="job2",
            cluster="test-cluster",
            status="PENDING",
            metadata="",
            done=False,
            state=JobState.PENDING,
        ),
    ]
    mock_modal_client.list_jobs.return_value = jobs
    mock_modal_client.cancel.return_value = JobStatus(
        id="",
        name="",
        cluster="",
        status="CANCELLED",
        metadata="",
        done=True,
        state=JobState.CANCELLED,
    )

    cluster.stop()

    # Both jobs should have been cancelled
    assert mock_modal_client.cancel.call_count == 2


def test_modal_cluster_down(mock_modal_client):
    """Test tearing down the cluster is same as stop."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    mock_modal_client.list_jobs.return_value = []

    cluster.down()

    mock_modal_client.list_jobs.assert_called()


def test_modal_cluster_get_logs_stream(mock_modal_client):
    """Test getting log stream."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    mock_modal_client.get_logs.return_value = "Log line 1\nLog line 2"
    mock_modal_client.list_jobs.return_value = [
        JobStatus(
            id="job-1",
            name="job1",
            cluster="test-cluster",
            status="RUNNING",
            metadata="",
            done=False,
            state=JobState.RUNNING,
        )
    ]

    stream = cluster.get_logs_stream("test-cluster", "job-1")

    assert isinstance(stream, io.StringIO)
    content = stream.read()
    assert "Log line 1" in content


def test_modal_cluster_get_logs_stream_default_job(mock_modal_client):
    """Test getting log stream for most recent job."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    mock_modal_client.get_logs.return_value = "Recent log"
    mock_modal_client.list_jobs.return_value = [
        JobStatus(
            id="job-1",
            name="job1",
            cluster="test-cluster",
            status="COMPLETED",
            metadata="",
            done=True,
            state=JobState.SUCCEEDED,
        ),
        JobStatus(
            id="job-2",
            name="job2",
            cluster="test-cluster",
            status="RUNNING",
            metadata="",
            done=False,
            state=JobState.RUNNING,
        ),
    ]

    stream = cluster.get_logs_stream("test-cluster")
    assert stream

    # Should get logs for the last job (job-2)
    mock_modal_client.get_logs.assert_called_with("job-2")


def test_modal_cluster_get_logs_stream_no_jobs(mock_modal_client):
    """Test getting log stream when no jobs exist."""
    cluster = ModalCluster("test-cluster", mock_modal_client)
    mock_modal_client.list_jobs.return_value = []

    with pytest.raises(RuntimeError, match="No jobs found"):
        cluster.get_logs_stream("test-cluster")
