from unittest.mock import Mock, call, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import JobStatus
from oumi.core.registry import REGISTRY, RegistryType
from oumi.launcher.clients.slurm_client import SlurmClient
from oumi.launcher.clouds.slurm_cloud import SlurmCloud
from oumi.launcher.clusters.slurm_cluster import SlurmCluster


#
# Fixtures
#
@pytest.fixture
def mock_slurm_client():
    with patch("oumi.launcher.clouds.slurm_cloud.SlurmClient") as client:
        yield client


@pytest.fixture
def mock_slurm_cluster():
    with patch("oumi.launcher.clouds.slurm_cloud.SlurmCluster") as cluster:
        cluster.get_slurm_connections.return_value = []
        cluster.parse_cluster_name = SlurmCluster.parse_cluster_name
        yield cluster


@pytest.fixture
def mock_get_slurm_connections():
    with patch(
        "oumi.launcher.clouds.slurm_cloud.SlurmCluster.get_slurm_connections"
    ) as get_conns:
        get_conns.return_value = []
        yield get_conns


@pytest.fixture
def mock_parse_cluster_name():
    with patch(
        "oumi.launcher.clouds.slurm_cloud.SlurmCluster.parse_cluster_name"
    ) as parse_name:
        yield parse_name


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
def test_slurm_cloud_up_cluster(mock_slurm_client, mock_slurm_cluster):
    cloud = SlurmCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=SlurmCluster)
    mock_slurm_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="user@somehost",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = _get_default_job("slurm")
    job_status = cloud.up_cluster(job, "user@somehost")
    mock_slurm_client.assert_called_once_with(
        user="user", slurm_host="somehost", cluster_name="user@somehost"
    )
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_slurm_cloud_up_cluster_fails_mismatched_user(
    mock_slurm_client, mock_slurm_cluster
):
    cloud = SlurmCloud()
    with pytest.raises(
        ValueError,
        match=(
            "Invalid cluster name: `user1@somehost`. "
            "User must match the provided job user: `user`."
        ),
    ):
        _ = cloud.up_cluster(_get_default_job("slurm"), "user1@somehost")


def test_slurm_cloud_init_with_connections(
    mock_slurm_client, mock_get_slurm_connections
):
    mock_client = Mock(spec=SlurmClient)
    mock_get_slurm_connections.return_value = [
        SlurmCluster.ConnectionInfo(user="user1", hostname="host1"),
        SlurmCluster.ConnectionInfo(user="user2", hostname="host2"),
    ]
    mock_slurm_client.side_effect = [mock_client, mock_client]
    cloud = SlurmCloud()
    cluster_names = [cluster.name() for cluster in cloud.list_clusters()]
    cluster_names.sort()
    assert cluster_names == [
        "user1@host1",
        "user2@host2",
    ]
    mock_slurm_client.assert_has_calls(
        [
            call(user="user1", slurm_host="host1", cluster_name="user1@host1"),
            call(user="user2", slurm_host="host2", cluster_name="user2@host2"),
        ]
    )


def test_slurm_cloud_init_skips_malformed_connections(
    mock_slurm_client, mock_get_slurm_connections
):
    mock_client = Mock(spec=SlurmClient)
    mock_get_slurm_connections.return_value = [
        SlurmCluster.ConnectionInfo(user="user1", hostname="host1"),
        SlurmCluster.ConnectionInfo(user="user2", hostname="host2"),
        SlurmCluster.ConnectionInfo(user="user3", hostname="host3"),
    ]
    mock_slurm_client.side_effect = [mock_client, mock_client, mock_client]
    cloud = SlurmCloud()
    cluster_names = [cluster.name() for cluster in cloud.list_clusters()]
    cluster_names.sort()
    assert cluster_names == [
        "user1@host1",
        "user2@host2",
        "user3@host3",
    ]
    mock_slurm_client.assert_has_calls(
        [
            call(user="user1", slurm_host="host1", cluster_name="user1@host1"),
            call(user="user2", slurm_host="host2", cluster_name="user2@host2"),
            call(user="user3", slurm_host="host3", cluster_name="user3@host3"),
        ]
    )


def test_slurm_cloud_initialize_cluster(mock_slurm_client, mock_get_slurm_connections):
    cloud = SlurmCloud()
    mock_get_slurm_connections.return_value = [
        SlurmCluster.ConnectionInfo(user="user1", hostname="host1"),
        SlurmCluster.ConnectionInfo(user="user2", hostname="host2"),
        SlurmCluster.ConnectionInfo(user="user3", hostname="host3"),
    ]
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client, mock_client]
    clusters = cloud.initialize_clusters()
    clusters2 = cloud.initialize_clusters()
    mock_slurm_client.assert_has_calls(
        [
            call(user="user1", slurm_host="host1", cluster_name="user1@host1"),
            call(user="user2", slurm_host="host2", cluster_name="user2@host2"),
            call(user="user3", slurm_host="host3", cluster_name="user3@host3"),
        ]
    )
    cluster_names = [cluster.name() for cluster in clusters]
    cluster_names.sort()
    assert cluster_names == [
        "user1@host1",
        "user2@host2",
        "user3@host3",
    ]
    # Verify that the second initialization returns the same clusters.
    assert clusters == clusters2


def test_slurm_cloud_list_clusters(mock_slurm_client, mock_get_slurm_connections):
    cloud = SlurmCloud()
    mock_get_slurm_connections.return_value = [
        SlurmCluster.ConnectionInfo(user="user1", hostname="host1"),
        SlurmCluster.ConnectionInfo(user="user2", hostname="host2"),
        SlurmCluster.ConnectionInfo(user="user3", hostname="host3"),
    ]
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client, mock_client]
    assert [] == cloud.list_clusters()
    clusters = cloud.initialize_clusters()
    mock_slurm_client.assert_has_calls(
        [
            call(user="user1", slurm_host="host1", cluster_name="user1@host1"),
            call(user="user2", slurm_host="host2", cluster_name="user2@host2"),
            call(user="user3", slurm_host="host3", cluster_name="user3@host3"),
        ]
    )
    clusters = cloud.list_clusters()
    expected_clusters = [
        "user1@host1",
        "user2@host2",
        "user3@host3",
    ]
    cluster_names = [cluster.name() for cluster in clusters]
    cluster_names.sort()
    assert cluster_names == expected_clusters


def test_slurm_cloud_get_cluster_empty(mock_slurm_client):
    cloud = SlurmCloud()
    # Check that there are no initial clusters.
    assert cloud.get_cluster("debug.user") is None


def test_slurm_cloud_get_cluster_success(mock_slurm_client, mock_get_slurm_connections):
    mock_client = Mock(spec=SlurmClient)
    mock_get_slurm_connections.side_effect = [
        [],
        [
            SlurmCluster.ConnectionInfo(user="user1", hostname="host1"),
            SlurmCluster.ConnectionInfo(user="user2", hostname="host2"),
            SlurmCluster.ConnectionInfo(user="user3", hostname="host3"),
        ],
    ]
    mock_slurm_client.side_effect = [mock_client, mock_client, mock_client]
    cloud = SlurmCloud()
    assert [] == cloud.list_clusters()
    _ = cloud.initialize_clusters()
    mock_slurm_client.assert_has_calls(
        [
            call(user="user1", slurm_host="host1", cluster_name="user1@host1"),
            call(user="user2", slurm_host="host2", cluster_name="user2@host2"),
            call(user="user3", slurm_host="host3", cluster_name="user3@host3"),
        ]
    )
    expected_clusters = [
        "user1@host1",
        "user2@host2",
        "user3@host3",
    ]
    for name in expected_clusters:
        cluster = cloud.get_cluster(name)
        assert cluster is not None
        assert cluster.name() == name


def test_slurm_cloud_get_cluster_fails(mock_slurm_client):
    cloud = SlurmCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client]
    cloud.initialize_clusters()
    assert cloud.get_cluster("nonexistent") is None


def test_slurm_cloud_builder_registered():
    assert REGISTRY.contains("slurm", RegistryType.CLOUD)
