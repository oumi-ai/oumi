from unittest.mock import Mock, call, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobStatus
from oumi.core.registry import REGISTRY, RegistryType
from oumi.launcher.clients.slurm_client import SlurmClient
from oumi.launcher.clouds.frontier_cloud import FrontierCloud
from oumi.launcher.clusters.frontier_cluster import FrontierCluster


#
# Fixtures
#
@pytest.fixture
def mock_slurm_client():
    with patch("oumi.launcher.clouds.frontier_cloud.SlurmClient") as client:
        yield client


@pytest.fixture
def mock_frontier_cluster():
    with patch("oumi.launcher.clouds.frontier_cloud.FrontierCluster") as cluster:
        cluster.SupportedQueues = FrontierCluster.SupportedQueues
        yield cluster


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
def test_frontier_cloud_up_cluster_extended(mock_slurm_client, mock_frontier_cluster):
    cloud = FrontierCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=FrontierCluster)
    mock_frontier_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="extended.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("frontier")
    job_status = cloud.up_cluster(job, "extended.user")
    mock_slurm_client.assert_called_once_with(
        "user", "frontier.olcf.ornl.gov", "extended.user"
    )
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_frontier_cloud_up_cluster_batch(mock_slurm_client, mock_frontier_cluster):
    cloud = FrontierCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=FrontierCluster)
    mock_frontier_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="batch.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("frontier")
    job_status = cloud.up_cluster(job, "batch.user")
    mock_slurm_client.assert_called_once_with(
        "user", "frontier.olcf.ornl.gov", "batch.user"
    )
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_frontier_cloud_up_cluster_fails_mismatched_user(
    mock_slurm_client, mock_frontier_cluster
):
    cloud = FrontierCloud()
    with pytest.raises(ValueError, match="User must match the provided job user"):
        _ = cloud.up_cluster(_get_default_job("frontier"), "batch.user1")
    with pytest.raises(ValueError, match="User must match the provided job user"):
        _ = cloud.up_cluster(_get_default_job("frontier"), "extended.user1")


def test_frontier_cloud_up_cluster_default_queue(
    mock_slurm_client, mock_frontier_cluster
):
    cloud = FrontierCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=FrontierCluster)
    mock_frontier_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="batch.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("frontier")
    job_status = cloud.up_cluster(job, None)
    mock_slurm_client.assert_called_once_with(
        "user", "frontier.olcf.ornl.gov", "batch.user"
    )
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_frontier_cloud_init_with_users(mock_slurm_client):
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client]
    mock_slurm_client.get_active_users.return_value = ["user1", "user2"]
    cloud = FrontierCloud()
    cluster_names = [cluster.name() for cluster in cloud.list_clusters()]
    cluster_names.sort()
    assert cluster_names == [
        "batch.user1",
        "batch.user2",
        "extended.user1",
        "extended.user2",
    ]
    mock_slurm_client.assert_has_calls(
        [
            call("user1", "frontier.olcf.ornl.gov", "batch.user1"),
            call("user2", "frontier.olcf.ornl.gov", "batch.user2"),
        ]
    )


def test_frontier_cloud_initialize_cluster(mock_slurm_client):
    cloud = FrontierCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client]
    clusters = cloud.initialize_clusters("me")
    clusters2 = cloud.initialize_clusters("me")
    mock_slurm_client.assert_called_once_with(
        "me", "frontier.olcf.ornl.gov", "batch.me"
    )
    cluster_names = [cluster.name() for cluster in clusters]
    cluster_names.sort()
    assert cluster_names == [
        "batch.me",
        "extended.me",
    ]
    # Verify that the second initialization returns the same clusters.
    assert clusters == clusters2


def test_frontier_cloud_list_clusters(mock_slurm_client):
    cloud = FrontierCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client]
    # Check that there are no initial clusters.
    assert [] == cloud.list_clusters()
    cloud.initialize_clusters("me")
    clusters = cloud.list_clusters()
    expected_clusters = [
        "batch.me",
        "extended.me",
    ]
    cluster_names = [cluster.name() for cluster in clusters]
    cluster_names.sort()
    assert cluster_names == expected_clusters


def test_frontier_cloud_get_cluster_empty(mock_slurm_client):
    cloud = FrontierCloud()
    # Check that there are no initial clusters.
    assert cloud.get_cluster("batch.user") is None


def test_frontier_cloud_get_cluster_success(mock_slurm_client):
    cloud = FrontierCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client]
    cloud.initialize_clusters("me")
    expected_clusters = [
        "batch.me",
        "extended.me",
    ]
    for name in expected_clusters:
        cluster = cloud.get_cluster(name)
        assert cluster is not None
        assert cluster.name() == name


def test_frontier_cloud_get_cluster_fails(mock_slurm_client):
    cloud = FrontierCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client]
    cloud.initialize_clusters("me")
    assert cloud.get_cluster("nonexistent") is None


def test_frontier_cloud_builder_registered():
    assert REGISTRY.contains("frontier", RegistryType.CLOUD)
