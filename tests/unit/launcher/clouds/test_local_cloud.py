from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobStatus
from oumi.core.registry import REGISTRY, RegistryType
from oumi.launcher.clients.local_client import LocalClient
from oumi.launcher.clouds.local_cloud import LocalCloud
from oumi.launcher.clusters.local_cluster import LocalCluster


#
# Fixtures
#
@pytest.fixture
def mock_local_client():
    with patch("oumi.launcher.clouds.local_cloud.LocalClient") as client:
        yield client


@pytest.fixture
def mock_local_cluster():
    with patch("oumi.launcher.clouds.local_cloud.LocalCluster") as cluster:
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
def test_local_cloud_up_cluster(mock_local_client, mock_local_cluster):
    cloud = LocalCloud()
    mock_client = Mock(spec=LocalClient)
    mock_local_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=LocalCluster)
    mock_local_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="cluster.name",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("local")
    job_status = cloud.up_cluster(job, "cluster.name")
    mock_local_client.assert_called_once()
    mock_local_cluster.assert_called_once_with("cluster.name", mock_client)
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_local_cloud_up_cluster_no_name(mock_local_client, mock_local_cluster):
    cloud = LocalCloud()
    mock_client = Mock(spec=LocalClient)
    mock_local_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=LocalCluster)
    mock_local_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="cluster.name",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("local")
    job_status = cloud.up_cluster(job, None)
    mock_local_client.assert_called_once()
    mock_local_cluster.assert_called_once_with("local", mock_client)
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_local_cloud_list_clusters(mock_local_client):
    cloud = LocalCloud()
    mock_client = Mock(spec=LocalClient)
    mock_local_client.side_effect = [mock_client, mock_client]
    # Check that there are no initial clusters.
    assert [] == cloud.list_clusters()
    job = _get_default_job("local")
    cloud.up_cluster(job, None)
    cloud.up_cluster(job, "new cluster")
    cloud.up_cluster(job, "local")
    clusters = cloud.list_clusters()
    expected_clusters = [
        "local",
        "new cluster",
    ]
    cluster_names = [cluster.name() for cluster in clusters]
    cluster_names.sort()
    assert cluster_names == expected_clusters


def test_local_cloud_get_cluster_empty(mock_local_client):
    cloud = LocalCloud()
    # Check that there are no initial clusters.
    assert cloud.get_cluster("local") is None


def test_local_cloud_get_cluster_success(mock_local_client):
    cloud = LocalCloud()
    mock_client = Mock(spec=LocalClient)
    mock_local_client.side_effect = [mock_client, mock_client]
    job = _get_default_job("local")
    cloud.up_cluster(job, None)
    cloud.up_cluster(job, "new cluster")
    cloud.up_cluster(job, "local")
    expected_clusters = [
        "local",
        "new cluster",
    ]
    for name in expected_clusters:
        cluster = cloud.get_cluster(name)
        assert cluster is not None
        assert cluster.name() == name


def test_local_cloud_get_cluster_fails(mock_local_client):
    cloud = LocalCloud()
    mock_client = Mock(spec=LocalClient)
    mock_local_client.side_effect = [mock_client, mock_client]
    job = _get_default_job("local")
    cloud.up_cluster(job, None)
    cloud.up_cluster(job, "new cluster")
    cloud.up_cluster(job, "local")
    assert cloud.get_cluster("nonexistent") is None


def test_local_cloud_builder_registered():
    assert REGISTRY.contains("local", RegistryType.CLOUD)
