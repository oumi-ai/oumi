from unittest.mock import Mock, patch

import pytest

from lema.core.types.configs import JobConfig
from lema.core.types.params.node_params import DiskTier, NodeParams, StorageMount
from lema.launcher.clients.polaris_client import PolarisClient
from lema.launcher.clouds.polaris_cloud import PolarisCloud


#
# Fixtures
#
@pytest.fixture
def mock_polaris_client():
    with patch("lema.launcher.clouds.polaris_cloud.PolarisClient") as client:
        client.SupportedQueues = PolarisClient.SupportedQueues
        yield client


@pytest.fixture
def mock_polaris_cluster():
    with patch("lema.launcher.clouds.polaris_cloud.PolarisCluster") as cluster:
        yield cluster


def _get_default_job(cloud: str) -> JobConfig:
    resources = NodeParams(
        cloud=cloud,
        region="us-central1",
        zone=None,
        accelerators="A100-80GB",
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
def test_polaris_cloud_up_cluster_debug(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    cluster = cloud.up_cluster(_get_default_job("polaris"), "debug.user")
    mock_polaris_client.assert_called_once_with("user")
    assert cluster.name() == "debug.user"


def test_polaris_cloud_up_cluster_demand(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    cluster = cloud.up_cluster(_get_default_job("polaris"), "demand.user")
    mock_polaris_client.assert_called_once_with("user")
    assert cluster.name() == "demand.user"


def test_polaris_cloud_up_cluster_debug_scaling(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    cluster = cloud.up_cluster(_get_default_job("polaris"), "debug-scaling.user")
    mock_polaris_client.assert_called_once_with("user")
    assert cluster.name() == "debug-scaling.user"


def test_polaris_cloud_up_cluster_preemptable(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    cluster = cloud.up_cluster(_get_default_job("polaris"), "preemptable.user")
    mock_polaris_client.assert_called_once_with("user")
    assert cluster.name() == "preemptable.user"


def test_polaris_cloud_up_cluster_prod(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    cluster = cloud.up_cluster(_get_default_job("polaris"), "prod.user")
    mock_polaris_client.assert_called_once_with("user")
    assert cluster.name() == "prod.user"


def test_polaris_cloud_up_cluster_fails_mismatched_user(mock_polaris_client):
    cloud = PolarisCloud()
    with pytest.raises(ValueError):
        _ = cloud.up_cluster(_get_default_job("polaris"), "debug.user1")


def test_polaris_cloud_up_cluster_default_queue(mock_polaris_client):
    cloud = PolarisCloud()
    mock_client = Mock(spec=PolarisClient)
    mock_polaris_client.side_effect = [mock_client]
    cluster = cloud.up_cluster(_get_default_job("polaris"), name=None)
    mock_polaris_client.assert_called_once_with("user")
    assert cluster.name() == "prod.user"


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
    cloud.up_cluster(_get_default_job("polaris"), "debug.user")
    cloud.initialize_clusters("me")
    clusters = cloud.list_clusters()
    expected_clusters = [
        "debug-scaling.me",
        "debug.me",
        "debug.user",
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
    cloud.up_cluster(_get_default_job("polaris"), "debug.user")
    cloud.initialize_clusters("me")
    expected_clusters = [
        "debug-scaling.me",
        "debug.me",
        "debug.user",
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
    cloud.up_cluster(_get_default_job("polaris"), "debug.user")
    cloud.initialize_clusters("me")
    assert cloud.get_cluster("nonexistent") is None
