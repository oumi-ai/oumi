"""Unit tests for ModalCloud."""

from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import JobConfig, JobResources
from oumi.core.launcher import JobState, JobStatus
from oumi.core.registry import REGISTRY, RegistryType
from oumi.launcher.clients.modal_client import ModalClient
from oumi.launcher.clouds.modal_cloud import ModalCloud
from oumi.launcher.clusters.modal_cluster import ModalCluster


#
# Fixtures
#
@pytest.fixture
def mock_modal_client():
    with patch("oumi.launcher.clouds.modal_cloud.ModalClient") as client:
        yield client


@pytest.fixture
def mock_modal_cluster():
    with patch("oumi.launcher.clouds.modal_cloud.ModalCluster") as cluster:
        yield cluster


def _get_default_job(cloud: str = "modal") -> JobConfig:
    """Create a default JobConfig for testing."""
    resources = JobResources(
        cloud=cloud,
        region=None,
        zone=None,
        accelerators="H100:4",
        cpus="8",
        memory="128GB",
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
        setup="pip install torch transformers",
        run="python train.py",
    )


#
# Tests
#
def test_modal_cloud_up_cluster(mock_modal_client, mock_modal_cluster):
    """Test creating a cluster and running a job."""
    cloud = ModalCloud()
    mock_client = Mock(spec=ModalClient)
    mock_modal_client.return_value = mock_client
    mock_cluster = Mock(spec=ModalCluster)
    mock_modal_cluster.return_value = mock_cluster

    expected_job_status = JobStatus(
        id="modal-0-abc123",
        cluster="test-cluster",
        name="test-modal-job",
        status="SUBMITTING",
        metadata="",
        done=False,
        state=JobState.PENDING,
    )
    mock_cluster.run_job.return_value = expected_job_status

    job = _get_default_job("modal")
    job_status = cloud.up_cluster(job, "test-cluster")

    mock_modal_client.assert_called_once()
    mock_modal_cluster.assert_called_once_with("test-cluster", mock_client)
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_modal_cloud_up_cluster_no_name(mock_modal_client, mock_modal_cluster):
    """Test creating a cluster with default name."""
    cloud = ModalCloud()
    mock_client = Mock(spec=ModalClient)
    mock_modal_client.return_value = mock_client
    mock_cluster = Mock(spec=ModalCluster)
    mock_modal_cluster.return_value = mock_cluster

    expected_job_status = JobStatus(
        id="modal-0-abc123",
        cluster="modal",
        name="test-modal-job",
        status="SUBMITTING",
        metadata="",
        done=False,
        state=JobState.PENDING,
    )
    mock_cluster.run_job.return_value = expected_job_status

    job = _get_default_job("modal")
    job_status = cloud.up_cluster(job, None)

    mock_modal_cluster.assert_called_once_with("modal", mock_client)
    assert job_status == expected_job_status


def test_modal_cloud_up_cluster_reuses_existing(mock_modal_client, mock_modal_cluster):
    """Test that up_cluster reuses existing cluster."""
    cloud = ModalCloud()
    mock_client = Mock(spec=ModalClient)
    mock_modal_client.return_value = mock_client
    mock_cluster = Mock(spec=ModalCluster)
    mock_modal_cluster.return_value = mock_cluster

    mock_cluster.run_job.return_value = JobStatus(
        id="job-1",
        cluster="modal",
        name="job",
        status="RUNNING",
        metadata="",
        done=False,
        state=JobState.RUNNING,
    )

    job = _get_default_job("modal")

    # First call creates cluster
    cloud.up_cluster(job, "modal")
    # Second call should reuse
    cloud.up_cluster(job, "modal")

    # Client and cluster should only be created once
    assert mock_modal_client.call_count == 1
    assert mock_modal_cluster.call_count == 1
    # But run_job should be called twice
    assert mock_cluster.run_job.call_count == 2


def test_modal_cloud_up_cluster_fails(mock_modal_client, mock_modal_cluster):
    """Test up_cluster raises error when job fails to submit."""
    cloud = ModalCloud()
    mock_client = Mock(spec=ModalClient)
    mock_modal_client.return_value = mock_client
    mock_cluster = Mock(spec=ModalCluster)
    mock_modal_cluster.return_value = mock_cluster
    mock_cluster.run_job.return_value = None

    job = _get_default_job("modal")

    with pytest.raises(RuntimeError, match="Failed to submit job"):
        cloud.up_cluster(job, "test-cluster")


def test_modal_cloud_list_clusters_empty(mock_modal_client):
    """Test listing clusters when none exist."""
    cloud = ModalCloud()
    assert cloud.list_clusters() == []


def test_modal_cloud_list_clusters(mock_modal_client, mock_modal_cluster):
    """Test listing clusters after creating some."""
    cloud = ModalCloud()
    mock_client = Mock(spec=ModalClient)
    mock_modal_client.return_value = mock_client
    mock_cluster = Mock(spec=ModalCluster)
    mock_cluster.name.return_value = "modal"
    mock_modal_cluster.return_value = mock_cluster
    mock_cluster.run_job.return_value = JobStatus(
        id="job-1",
        cluster="modal",
        name="job",
        status="RUNNING",
        metadata="",
        done=False,
        state=JobState.RUNNING,
    )

    job = _get_default_job("modal")
    cloud.up_cluster(job, None)
    cloud.up_cluster(job, "another-cluster")

    clusters = cloud.list_clusters()
    assert len(clusters) == 2


def test_modal_cloud_get_cluster_empty(mock_modal_client):
    """Test getting cluster when none exist."""
    cloud = ModalCloud()
    assert cloud.get_cluster("modal") is None


def test_modal_cloud_get_cluster_success(mock_modal_client, mock_modal_cluster):
    """Test getting an existing cluster."""
    cloud = ModalCloud()
    mock_client = Mock(spec=ModalClient)
    mock_modal_client.return_value = mock_client
    mock_cluster = Mock(spec=ModalCluster)
    mock_cluster.name.return_value = "test-cluster"
    mock_modal_cluster.return_value = mock_cluster
    mock_cluster.run_job.return_value = JobStatus(
        id="job-1",
        cluster="test-cluster",
        name="job",
        status="RUNNING",
        metadata="",
        done=False,
        state=JobState.RUNNING,
    )

    job = _get_default_job("modal")
    cloud.up_cluster(job, "test-cluster")

    cluster = cloud.get_cluster("test-cluster")
    assert cluster is not None


def test_modal_cloud_get_cluster_not_found(mock_modal_client, mock_modal_cluster):
    """Test getting a non-existent cluster."""
    cloud = ModalCloud()
    mock_client = Mock(spec=ModalClient)
    mock_modal_client.return_value = mock_client
    mock_cluster = Mock(spec=ModalCluster)
    mock_cluster.name.return_value = "existing-cluster"
    mock_modal_cluster.return_value = mock_cluster
    mock_cluster.run_job.return_value = JobStatus(
        id="job-1",
        cluster="existing-cluster",
        name="job",
        status="RUNNING",
        metadata="",
        done=False,
        state=JobState.RUNNING,
    )

    job = _get_default_job("modal")
    cloud.up_cluster(job, "existing-cluster")

    assert cloud.get_cluster("nonexistent") is None


def test_modal_cloud_builder_registered():
    """Test that modal cloud builder is registered."""
    assert REGISTRY.contains("modal", RegistryType.CLOUD)


def test_modal_cloud_builder_creates_instance():
    """Test that modal cloud builder creates ModalCloud instance."""
    builder = REGISTRY.get("modal", RegistryType.CLOUD)
    cloud = builder()
    assert isinstance(cloud, ModalCloud)
