from unittest.mock import Mock

import pytest
import sky

from lema.core.registry import REGISTRY, RegistryType
from lema.core.types.configs import JobConfig
from lema.core.types.params.node_params import DiskTier, NodeParams, StorageMount
from lema.launcher.clients.sky_client import SkyClient
from lema.launcher.clouds.sky_cloud import SkyCloud
from lema.launcher.clusters.sky_cluster import SkyCluster


#
# Fixtures
#
@pytest.fixture
def mock_sky_client():
    yield Mock(spec=SkyClient)


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
def test_sky_cloud_up_cluster(mock_sky_client):
    cloud = SkyCloud("gcp", mock_sky_client)
    mock_sky_client.launch.return_value = "new_cluster_name"
    cluster = cloud.up_cluster(_get_default_job("gcp"), "new_cluster_name")
    mock_sky_client.launch.assert_called_once_with(
        _get_default_job("gcp"), "new_cluster_name"
    )
    assert cluster.name() == "new_cluster_name"


def test_sky_cloud_up_cluster_no_name(mock_sky_client):
    cloud = SkyCloud("gcp", mock_sky_client)
    mock_sky_client.launch.return_value = "new_cluster_name"
    cluster = cloud.up_cluster(_get_default_job("gcp"), None)
    mock_sky_client.launch.assert_called_once_with(_get_default_job("gcp"), None)
    assert cluster.name() == "new_cluster_name"


def test_sky_cloud_list_clusters_gcp(mock_sky_client):
    cloud = SkyCloud("gcp", mock_sky_client)
    mock_gcp_cluster = Mock(spec=sky.clouds.GCP)
    mock_gcp_handler = Mock()
    mock_gcp_handler.launched_resources = Mock()
    mock_gcp_handler.launched_resources.cloud = mock_gcp_cluster

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_lambda_cluster = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler = Mock()
    mock_lambda_handler.launched_resources = Mock()
    mock_lambda_handler.launched_resources.cloud = mock_lambda_cluster
    mock_sky_client.status.return_value = [
        {"name": "gcp_cluster", "status": "RUNNING", "handle": mock_gcp_handler},
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
        {"name": "lambda_cluster", "status": "RUNNING", "handle": mock_lambda_handler},
    ]
    clusters = cloud.list_clusters()
    mock_sky_client.status.assert_called_once()
    assert clusters == [SkyCluster("gcp_cluster", mock_sky_client)]


def test_sky_cloud_list_clusters_runpod(mock_sky_client):
    cloud = SkyCloud("runpod", mock_sky_client)
    mock_gcp_cluster = Mock(spec=sky.clouds.GCP)
    mock_gcp_handler = Mock()
    mock_gcp_handler.launched_resources = Mock()
    mock_gcp_handler.launched_resources.cloud = mock_gcp_cluster

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_lambda_cluster = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler = Mock()
    mock_lambda_handler.launched_resources = Mock()
    mock_lambda_handler.launched_resources.cloud = mock_lambda_cluster
    mock_sky_client.status.return_value = [
        {"name": "gcp_cluster", "status": "RUNNING", "handle": mock_gcp_handler},
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
        {"name": "lambda_cluster", "status": "RUNNING", "handle": mock_lambda_handler},
    ]
    clusters = cloud.list_clusters()
    mock_sky_client.status.assert_called_once()
    assert clusters == [SkyCluster("runpod_cluster", mock_sky_client)]


def test_sky_cloud_list_clusters_lambda(mock_sky_client):
    cloud = SkyCloud("lambda", mock_sky_client)
    mock_gcp_cluster = Mock(spec=sky.clouds.GCP)
    mock_gcp_handler = Mock()
    mock_gcp_handler.launched_resources = Mock()
    mock_gcp_handler.launched_resources.cloud = mock_gcp_cluster

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_lambda_cluster = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler = Mock()
    mock_lambda_handler.launched_resources = Mock()
    mock_lambda_handler.launched_resources.cloud = mock_lambda_cluster
    mock_sky_client.status.return_value = [
        {"name": "gcp_cluster", "status": "RUNNING", "handle": mock_gcp_handler},
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
        {"name": "lambda_cluster", "status": "RUNNING", "handle": mock_lambda_handler},
    ]
    clusters = cloud.list_clusters()
    mock_sky_client.status.assert_called_once()
    assert clusters == [SkyCluster("lambda_cluster", mock_sky_client)]


def test_sky_cloud_list_clusters_lambda_no_cluster(mock_sky_client):
    cloud = SkyCloud("lambda", mock_sky_client)
    mock_gcp_cluster = Mock(spec=sky.clouds.GCP)
    mock_gcp_handler = Mock()
    mock_gcp_handler.launched_resources = Mock()
    mock_gcp_handler.launched_resources.cloud = mock_gcp_cluster

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_sky_client.status.return_value = [
        {"name": "gcp_cluster", "status": "RUNNING", "handle": mock_gcp_handler},
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
    ]
    clusters = cloud.list_clusters()
    mock_sky_client.status.assert_called_once()
    assert clusters == []


def test_sky_cloud_list_clusters_lambda_multiple_cluster(mock_sky_client):
    cloud = SkyCloud("lambda", mock_sky_client)
    mock_gcp_cluster = Mock(spec=sky.clouds.GCP)
    mock_gcp_handler = Mock()
    mock_gcp_handler.launched_resources = Mock()
    mock_gcp_handler.launched_resources.cloud = mock_gcp_cluster

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_lambda_cluster = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler = Mock()
    mock_lambda_handler.launched_resources = Mock()
    mock_lambda_handler.launched_resources.cloud = mock_lambda_cluster

    mock_lambda_cluster2 = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler2 = Mock()
    mock_lambda_handler2.launched_resources = Mock()
    mock_lambda_handler2.launched_resources.cloud = mock_lambda_cluster2

    mock_sky_client.status.return_value = [
        {"name": "gcp_cluster", "status": "RUNNING", "handle": mock_gcp_handler},
        {"name": "another_one", "status": "RUNNING", "handle": mock_lambda_handler2},
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
        {"name": "lambda_cluster", "status": "RUNNING", "handle": mock_lambda_handler},
    ]
    clusters = cloud.list_clusters()
    mock_sky_client.status.assert_called_once()
    assert clusters == [
        SkyCluster("another_one", mock_sky_client),
        SkyCluster("lambda_cluster", mock_sky_client),
    ]


def test_sky_cloud_list_clusters_invalid_cloud(mock_sky_client):
    cloud = SkyCloud("fake_cloud", mock_sky_client)
    mock_gcp_cluster = Mock(spec=sky.clouds.GCP)
    mock_gcp_handler = Mock()
    mock_gcp_handler.launched_resources = Mock()
    mock_gcp_handler.launched_resources.cloud = mock_gcp_cluster

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_lambda_cluster = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler = Mock()
    mock_lambda_handler.launched_resources = Mock()
    mock_lambda_handler.launched_resources.cloud = mock_lambda_cluster

    mock_lambda_cluster2 = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler2 = Mock()
    mock_lambda_handler2.launched_resources = Mock()
    mock_lambda_handler2.launched_resources.cloud = mock_lambda_cluster2

    mock_sky_client.status.return_value = [
        {"name": "gcp_cluster", "status": "RUNNING", "handle": mock_gcp_handler},
        {"name": "another_one", "status": "RUNNING", "handle": mock_lambda_handler2},
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
        {"name": "lambda_cluster", "status": "RUNNING", "handle": mock_lambda_handler},
    ]
    with pytest.raises(ValueError):
        _ = cloud.list_clusters()


def test_sky_cloud_get_cluster_gcp_success(mock_sky_client):
    cloud = SkyCloud("gcp", mock_sky_client)
    mock_gcp_cluster = Mock(spec=sky.clouds.GCP)
    mock_gcp_handler = Mock()
    mock_gcp_handler.launched_resources = Mock()
    mock_gcp_handler.launched_resources.cloud = mock_gcp_cluster

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_lambda_cluster = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler = Mock()
    mock_lambda_handler.launched_resources = Mock()
    mock_lambda_handler.launched_resources.cloud = mock_lambda_cluster

    mock_sky_client.status.return_value = [
        {"name": "gcp_cluster", "status": "RUNNING", "handle": mock_gcp_handler},
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
        {"name": "lambda_cluster", "status": "RUNNING", "handle": mock_lambda_handler},
    ]
    cluster = cloud.get_cluster("gcp_cluster")
    mock_sky_client.status.assert_called_once()
    assert cluster == SkyCluster("gcp_cluster", mock_sky_client)


def test_sky_cloud_get_cluster_runpod_success(mock_sky_client):
    cloud = SkyCloud("runpod", mock_sky_client)
    mock_gcp_cluster = Mock(spec=sky.clouds.GCP)
    mock_gcp_handler = Mock()
    mock_gcp_handler.launched_resources = Mock()
    mock_gcp_handler.launched_resources.cloud = mock_gcp_cluster

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_lambda_cluster = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler = Mock()
    mock_lambda_handler.launched_resources = Mock()
    mock_lambda_handler.launched_resources.cloud = mock_lambda_cluster

    mock_sky_client.status.return_value = [
        {"name": "gcp_cluster", "status": "RUNNING", "handle": mock_gcp_handler},
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
        {"name": "lambda_cluster", "status": "RUNNING", "handle": mock_lambda_handler},
    ]
    cluster = cloud.get_cluster("runpod_cluster")
    mock_sky_client.status.assert_called_once()
    assert cluster == SkyCluster("runpod_cluster", mock_sky_client)


def test_sky_cloud_get_cluster_lambda_success(mock_sky_client):
    cloud = SkyCloud("lambda", mock_sky_client)
    mock_gcp_cluster = Mock(spec=sky.clouds.GCP)
    mock_gcp_handler = Mock()
    mock_gcp_handler.launched_resources = Mock()
    mock_gcp_handler.launched_resources.cloud = mock_gcp_cluster

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_lambda_cluster = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler = Mock()
    mock_lambda_handler.launched_resources = Mock()
    mock_lambda_handler.launched_resources.cloud = mock_lambda_cluster

    mock_sky_client.status.return_value = [
        {"name": "gcp_cluster", "status": "RUNNING", "handle": mock_gcp_handler},
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
        {"name": "lambda_cluster", "status": "RUNNING", "handle": mock_lambda_handler},
    ]
    cluster = cloud.get_cluster("lambda_cluster")
    mock_sky_client.status.assert_called_once()
    assert cluster == SkyCluster("lambda_cluster", mock_sky_client)


def test_sky_cloud_get_cluster_failure_wrong_cloud(mock_sky_client):
    cloud = SkyCloud("gcp", mock_sky_client)

    mock_runpod_cluster = Mock(spec=sky.clouds.RunPod)
    mock_runpod_handler = Mock()
    mock_runpod_handler.launched_resources = Mock()
    mock_runpod_handler.launched_resources.cloud = mock_runpod_cluster

    mock_lambda_cluster = Mock(spec=sky.clouds.Lambda)
    mock_lambda_handler = Mock()
    mock_lambda_handler.launched_resources = Mock()
    mock_lambda_handler.launched_resources.cloud = mock_lambda_cluster

    mock_sky_client.status.return_value = [
        {"name": "runpod_cluster", "status": "RUNNING", "handle": mock_runpod_handler},
        {"name": "lambda_cluster", "status": "RUNNING", "handle": mock_lambda_handler},
    ]
    cluster = cloud.get_cluster("runpod_cluster")
    mock_sky_client.status.assert_called_once()
    assert cluster is None


def test_sky_cloud_get_cluster_failure_empty(mock_sky_client):
    cloud = SkyCloud("gcp", mock_sky_client)
    mock_sky_client.status.return_value = []
    cluster = cloud.get_cluster("gcp_cluster")
    mock_sky_client.status.assert_called_once()
    assert cluster is None


def test_runpod_cloud_builder_registered():
    assert REGISTRY.contains("runpod", RegistryType.CLOUD)


def test_gcp_cloud_builder_registered():
    assert REGISTRY.contains("gcp", RegistryType.CLOUD)


def test_lambda_cloud_builder_registered():
    assert REGISTRY.contains("lambda", RegistryType.CLOUD)
