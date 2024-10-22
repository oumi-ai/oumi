from unittest.mock import Mock, call, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobStatus
from oumi.core.registry import REGISTRY, RegistryType
from oumi.launcher.clients.polaris_client import PolarisClient
from oumi.launcher.clouds.polaris_cloud import PolarisCloud
from oumi.launcher.clusters.polaris_cluster import PolarisCluster


#
# Fixtures
#
@pytest.fixture
def mock_polaris_client():
    with patch("oumi.launcher.clouds.polaris_cloud.PolarisClient") as client:
        client.SupportedQueues = PolarisClient.SupportedQueues
        yield client


@pytest.fixture
def mock_polaris_cluster():
    with patch("oumi.launcher.clouds.polaris_cloud.PolarisCluster") as cluster:
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
def test_polaris_cloud_up_cluster_debug(mock_polaris_client, mock_polaris_cluster):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=PolarisCluster)
    mock_polaris_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="debug.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("polaris")
    job_status = cloud.up_cluster(job, "debug.user")
    mock_polaris_client.assert_called_once_with("user")
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_polaris_cloud_up_cluster_demand(mock_polaris_client, mock_polaris_cluster):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=PolarisCluster)
    mock_polaris_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="demand.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("polaris")
    job_status = cloud.up_cluster(job, "demand.user")
    mock_polaris_client.assert_called_once_with("user")
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_polaris_cloud_up_cluster_debug_scaling(
    mock_polaris_client, mock_polaris_cluster
):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=PolarisCluster)
    mock_polaris_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="debug-scaling.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("polaris")
    job_status = cloud.up_cluster(job, "debug-scaling.user")
    mock_polaris_client.assert_called_once_with("user")
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_polaris_cloud_up_cluster_preemptable(
    mock_polaris_client, mock_polaris_cluster
):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=PolarisCluster)
    mock_polaris_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="preemptable.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("polaris")
    job_status = cloud.up_cluster(job, "preemptable.user")
    mock_polaris_client.assert_called_once_with("user")
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_polaris_cloud_up_cluster_prod(mock_polaris_client, mock_polaris_cluster):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=PolarisCluster)
    mock_polaris_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="prod.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("polaris")
    job_status = cloud.up_cluster(job, "prod.user")
    mock_polaris_client.assert_called_once_with("user")
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_polaris_cloud_up_cluster_fails_mismatched_user(
    mock_polaris_client, mock_polaris_cluster
):
    cloud = PolarisCloud()
    with pytest.raises(ValueError):
        _ = cloud.up_cluster(_get_default_job("polaris"), "debug.user1")


def test_polaris_cloud_up_cluster_default_queue(
    mock_polaris_client, mock_polaris_cluster
):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=PolarisCluster)
    mock_polaris_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="prod.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("polaris")
    job_status = cloud.up_cluster(job, None)
    mock_polaris_client.assert_called_once_with("user")
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_polaris_cloud_init_with_users(mock_polaris_client):
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client, mock_client]
    mock_polaris_client.get_active_users.return_value = ["user1", "user2"]
    cloud = PolarisCloud()
    cluster_names = [cluster.name() for cluster in cloud.list_clusters()]
    cluster_names.sort()
    assert cluster_names == [
        "debug-scaling.user1",
        "debug-scaling.user2",
        "debug.user1",
        "debug.user2",
        "demand.user1",
        "demand.user2",
        "preemptable.user1",
        "preemptable.user2",
        "prod.user1",
        "prod.user2",
    ]
    mock_polaris_client.assert_has_calls([call("user1"), call("user2")])


def test_polaris_cloud_initialize_cluster(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    clusters = cloud.initialize_clusters("me")
    clusters2 = cloud.initialize_clusters("me")
    mock_polaris_client.assert_called_once_with("me")
    cluster_names = [cluster.name() for cluster in clusters]
    cluster_names.sort()
    assert cluster_names == [
        "debug-scaling.me",
        "debug.me",
        "demand.me",
        "preemptable.me",
        "prod.me",
    ]
    # Verify that the second initialization returns the same clusters.
    assert clusters == clusters2


def test_polaris_cloud_list_clusters(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client, mock_client]
    # Check that there are no initial clusters.
    assert [] == cloud.list_clusters()
    cloud.initialize_clusters("me")
    clusters = cloud.list_clusters()
    expected_clusters = [
        "debug-scaling.me",
        "debug.me",
        "demand.me",
        "preemptable.me",
        "prod.me",
    ]
    cluster_names = [cluster.name() for cluster in clusters]
    cluster_names.sort()
    assert cluster_names == expected_clusters


def test_polaris_cloud_get_cluster_empty(mock_polaris_client):
    cloud = PolarisCloud()
    # Check that there are no initial clusters.
    assert cloud.get_cluster("debug.user") is None


def test_polaris_cloud_get_cluster_success(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client, mock_client]
    cloud.initialize_clusters("me")
    expected_clusters = [
        "debug-scaling.me",
        "debug.me",
        "demand.me",
        "preemptable.me",
        "prod.me",
    ]
    for name in expected_clusters:
        cluster = cloud.get_cluster(name)
        assert cluster is not None
        assert cluster.name() == name


def test_polaris_cloud_get_cluster_fails(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client, mock_client]
    cloud.initialize_clusters("me")
    assert cloud.get_cluster("nonexistent") is None


def test_polaris_cloud_builder_registered():
    assert REGISTRY.contains("polaris", RegistryType.CLOUD)
