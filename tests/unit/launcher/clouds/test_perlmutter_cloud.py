from unittest.mock import Mock, call, patch

import pytest

from oumi.core.configs import JobConfig
from oumi.core.launcher import JobState, JobStatus
from oumi.core.registry import REGISTRY, RegistryType
from oumi.launcher.clients.slurm_client import SlurmClient
from oumi.launcher.clouds.perlmutter_cloud import PerlmutterCloud
from oumi.launcher.clusters.perlmutter_cluster import PerlmutterCluster


#
# Fixtures
#
@pytest.fixture
def mock_slurm_client():
    with patch("oumi.launcher.clouds.perlmutter_cloud.SlurmClient") as client:
        yield client


@pytest.fixture
def mock_perlmutter_cluster():
    with patch("oumi.launcher.clouds.perlmutter_cloud.PerlmutterCluster") as cluster:
        cluster.SupportedQueues = PerlmutterCluster.SupportedQueues
        yield cluster


#
# Tests
#
def test_perlmutter_cloud_up_cluster_regular(
    mock_slurm_client, mock_perlmutter_cluster
):
    cloud = PerlmutterCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=PerlmutterCluster)
    mock_perlmutter_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="regular.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
        state=JobState.PENDING,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = JobConfig(user="user")
    job_status = cloud.up_cluster(job, "regular.user")
    mock_slurm_client.assert_called_once_with(
        "user", "perlmutter.nersc.gov", "regular.user"
    )
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_perlmutter_cloud_up_cluster_debug(mock_slurm_client, mock_perlmutter_cluster):
    cloud = PerlmutterCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=PerlmutterCluster)
    mock_perlmutter_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="debug.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
        state=JobState.PENDING,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = JobConfig(user="user")
    job_status = cloud.up_cluster(job, "debug.user")
    mock_slurm_client.assert_called_once_with(
        "user", "perlmutter.nersc.gov", "debug.user"
    )
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_perlmutter_cloud_up_cluster_fails_mismatched_user(
    mock_slurm_client, mock_perlmutter_cluster
):
    cloud = PerlmutterCloud()
    with pytest.raises(ValueError, match="User must match the provided job user"):
        _ = cloud.up_cluster(JobConfig(user="user"), "debug.user1")
    with pytest.raises(ValueError, match="User must match the provided job user"):
        _ = cloud.up_cluster(JobConfig(user="user"), "regular.user1")


def test_perlmutter_cloud_up_cluster_default_queue(
    mock_slurm_client, mock_perlmutter_cluster
):
    cloud = PerlmutterCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client]
    mock_cluster = Mock(spec=PerlmutterCluster)
    mock_perlmutter_cluster.side_effect = [mock_cluster]
    expected_job_status = JobStatus(
        id="job_id",
        cluster="regular.user",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
        state=JobState.PENDING,
    )
    mock_cluster.run_job.return_value = expected_job_status
    job = JobConfig(user="user")
    job_status = cloud.up_cluster(job, None)
    mock_slurm_client.assert_called_once_with(
        "user", "perlmutter.nersc.gov", "regular.user"
    )
    mock_cluster.run_job.assert_called_once_with(job)
    assert job_status == expected_job_status


def test_perlmutter_cloud_init_with_users(mock_slurm_client):
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client]
    mock_slurm_client.get_active_users.return_value = ["user1", "user2"]
    cloud = PerlmutterCloud()
    cluster_names = [cluster.name() for cluster in cloud.list_clusters()]
    cluster_names.sort()
    assert cluster_names == [
        "debug.user1",
        "debug.user2",
        "debug_preempt.user1",
        "debug_preempt.user2",
        "interactive.user1",
        "interactive.user2",
        "jupyter.user1",
        "jupyter.user2",
        "overrun.user1",
        "overrun.user2",
        "preempt.user1",
        "preempt.user2",
        "premium.user1",
        "premium.user2",
        "realtime.user1",
        "realtime.user2",
        "regular.user1",
        "regular.user2",
        "shared.user1",
        "shared.user2",
        "shared_interactive.user1",
        "shared_interactive.user2",
        "shared_overrun.user1",
        "shared_overrun.user2",
    ]
    mock_slurm_client.assert_has_calls(
        [
            call("user1", "perlmutter.nersc.gov", "debug.user1"),
            call("user2", "perlmutter.nersc.gov", "debug.user2"),
        ]
    )


def test_perlmutter_cloud_initialize_cluster(mock_slurm_client):
    cloud = PerlmutterCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client]
    clusters = cloud.initialize_clusters("me")
    clusters2 = cloud.initialize_clusters("me")
    mock_slurm_client.assert_called_once_with("me", "perlmutter.nersc.gov", "debug.me")
    cluster_names = [cluster.name() for cluster in clusters]
    cluster_names.sort()
    assert cluster_names == [
        "debug.me",
        "debug_preempt.me",
        "interactive.me",
        "jupyter.me",
        "overrun.me",
        "preempt.me",
        "premium.me",
        "realtime.me",
        "regular.me",
        "shared.me",
        "shared_interactive.me",
        "shared_overrun.me",
    ]
    # Verify that the second initialization returns the same clusters.
    assert clusters == clusters2


def test_perlmutter_cloud_list_clusters(mock_slurm_client):
    cloud = PerlmutterCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client]
    # Check that there are no initial clusters.
    assert [] == cloud.list_clusters()
    cloud.initialize_clusters("me")
    clusters = cloud.list_clusters()
    expected_clusters = [
        "debug.me",
        "debug_preempt.me",
        "interactive.me",
        "jupyter.me",
        "overrun.me",
        "preempt.me",
        "premium.me",
        "realtime.me",
        "regular.me",
        "shared.me",
        "shared_interactive.me",
        "shared_overrun.me",
    ]
    cluster_names = [cluster.name() for cluster in clusters]
    cluster_names.sort()
    assert cluster_names == expected_clusters


def test_perlmutter_cloud_get_cluster_empty(mock_slurm_client):
    cloud = PerlmutterCloud()
    # Check that there are no initial clusters.
    assert cloud.get_cluster("debug.user") is None


def test_perlmutter_cloud_get_cluster_success(mock_slurm_client):
    cloud = PerlmutterCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client]
    cloud.initialize_clusters("me")
    expected_clusters = [
        "debug.me",
        "regular.me",
    ]
    for name in expected_clusters:
        cluster = cloud.get_cluster(name)
        assert cluster is not None
        assert cluster.name() == name


def test_perlmutter_cloud_get_cluster_fails(mock_slurm_client):
    cloud = PerlmutterCloud()
    mock_client = Mock(spec=SlurmClient)
    mock_slurm_client.side_effect = [mock_client, mock_client]
    cloud.initialize_clusters("me")
    assert cloud.get_cluster("nonexistent") is None


def test_perlmutter_cloud_builder_registered():
    assert REGISTRY.contains("perlmutter", RegistryType.CLOUD)
