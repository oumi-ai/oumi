from multiprocessing import Process, set_start_method
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.core.launcher import BaseCloud, BaseCluster, JobStatus
from oumi.launcher.launcher import (
    LAUNCHER,
    Launcher,
    cancel,
    down,
    get_cloud,
    run,
    status,
    stop,
    up,
    which_clouds,
)


#
# Fixtures
#
@pytest.fixture
def mock_registry():
    with patch("oumi.launcher.launcher.REGISTRY") as registry:
        yield registry


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
def test_launcher_get_cloud(mock_registry):
    sky_mock = Mock(spec=BaseCloud)
    polaris_mock = Mock(spec=BaseCloud)

    def _sky_builder():
        return sky_mock

    def _polaris_builder():
        return polaris_mock

    mock_registry.get_all.return_value = {
        "sky": _sky_builder,
        "polaris": _polaris_builder,
    }
    launcher = Launcher()
    cloud = launcher.get_cloud(_get_default_job("sky"))
    assert cloud == sky_mock
    assert cloud != polaris_mock


def test_launcher_get_cloud_missing_value(mock_registry):
    with pytest.raises(ValueError) as exception_info:
        sky_mock = Mock(spec=BaseCloud)
        polaris_mock = Mock(spec=BaseCloud)

        def _sky_builder():
            return sky_mock

        def _polaris_builder():
            return polaris_mock

        mock_registry.get.return_value = None
        mock_registry.get_all.return_value = {
            "sky": _sky_builder,
            "polaris": _polaris_builder,
        }
        launcher = Launcher()
        launcher.get_cloud(_get_default_job("lambda"))
    assert "not found in the registry." in str(exception_info.value)


def test_launcher_get_cloud_empty(mock_registry):
    with pytest.raises(ValueError) as exception_info:
        mock_registry.get_all.return_value = {}
        mock_registry.get.return_value = None
        launcher = Launcher()
        launcher.get_cloud(_get_default_job("sky"))
    assert "not found in the registry." in str(exception_info.value)


def test_launcher_get_cloud_by_name_missing_value(mock_registry):
    with pytest.raises(ValueError) as exception_info:
        sky_mock = Mock(spec=BaseCloud)
        polaris_mock = Mock(spec=BaseCloud)

        def _sky_builder():
            return sky_mock

        def _polaris_builder():
            return polaris_mock

        mock_registry.get_all.return_value = {
            "sky": _sky_builder,
            "polaris": _polaris_builder,
        }
        mock_registry.get.return_value = None
        launcher = Launcher()
        _ = launcher.get_cloud("lambda")
    assert "not found in the registry." in str(exception_info.value)


def test_launcher_get_cloud_by_name_empty(mock_registry):
    with pytest.raises(ValueError) as exception_info:
        mock_registry.get_all.return_value = {}
        mock_registry.get.return_value = None
        launcher = Launcher()
        _ = launcher.get_cloud("lambda")
    assert "not found in the registry." in str(exception_info.value)


def test_launcher_up_succeeds(mock_registry):
    mock_cluster = Mock(spec=BaseCluster)
    mock_cloud = Mock(spec=BaseCloud)

    def _builder():
        return mock_cloud

    mock_registry.get_all.return_value = {
        "custom": _builder,
    }
    expected_job_status = JobStatus(
        id="job_id",
        cluster="custom",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cloud.up_cluster.return_value = expected_job_status
    mock_cloud.get_cluster.return_value = mock_cluster
    launcher = Launcher()
    job = _get_default_job("custom")
    result = launcher.up(job, "custom")
    mock_cloud.up_cluster.assert_called_once_with(job, "custom")
    mock_cloud.get_cluster.assert_called_once_with("custom")
    assert result == (mock_cluster, expected_job_status)


def test_launcher_up_succeeds_kwargs(mock_registry):
    mock_cluster = Mock(spec=BaseCluster)
    mock_cloud = Mock(spec=BaseCloud)

    def _builder():
        return mock_cloud

    mock_registry.get_all.return_value = {
        "custom": _builder,
    }
    expected_job_status = JobStatus(
        id="job_id",
        cluster="custom",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cloud.up_cluster.return_value = expected_job_status
    mock_cloud.get_cluster.return_value = mock_cluster
    launcher = Launcher()
    job = _get_default_job("custom")
    result = launcher.up(job, "custom", foo="bar")
    mock_cloud.up_cluster.assert_called_once_with(job, "custom", foo="bar")
    mock_cloud.get_cluster.assert_called_once_with("custom")
    assert result == (mock_cluster, expected_job_status)


def test_launcher_up_succeeds_no_name(mock_registry):
    mock_cluster = Mock(spec=BaseCluster)
    mock_cloud = Mock(spec=BaseCloud)

    def _builder():
        return mock_cloud

    mock_registry.get_all.return_value = {
        "custom": _builder,
    }
    expected_job_status = JobStatus(
        id="job_id",
        cluster="custom",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cloud.up_cluster.return_value = expected_job_status
    mock_cloud.get_cluster.return_value = mock_cluster
    launcher = Launcher()
    job = _get_default_job("custom")
    result = launcher.up(job, None)
    mock_cloud.up_cluster.assert_called_once_with(job, None)
    mock_cloud.get_cluster.assert_called_once_with("custom")
    assert result == (mock_cluster, expected_job_status)


def test_launcher_up_inavlid_cluster(mock_registry):
    with pytest.raises(RuntimeError) as exception_info:
        mock_cloud = Mock(spec=BaseCloud)

        def _builder():
            return mock_cloud

        mock_registry.get_all.return_value = {
            "custom": _builder,
        }
        expected_job_status = JobStatus(
            id="job_id",
            cluster="custom",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        )
        mock_cloud.up_cluster.return_value = expected_job_status
        mock_cloud.get_cluster.return_value = None
        launcher = Launcher()
        job = _get_default_job("custom")
        launcher.up(job, None)
    assert "not found" in str(exception_info.value)


def test_launcher_run_succeeds(mock_registry):
    mock_cluster = Mock(spec=BaseCluster)
    mock_cloud = Mock(spec=BaseCloud)

    def _builder():
        return mock_cloud

    mock_registry.get_all.return_value = {
        "custom": _builder,
    }
    expected_job_status = JobStatus(
        id="job_id",
        cluster="custom",
        name="foo",
        status="running",
        metadata="bar",
        done=False,
    )
    mock_cloud.get_cluster.return_value = mock_cluster
    mock_cluster.run_job.return_value = expected_job_status
    launcher = Launcher()
    job = _get_default_job("custom")
    result = launcher.run(job, "custom")
    mock_cloud.get_cluster.assert_called_once_with("custom")
    mock_cluster.run_job.assert_called_once_with(job)
    assert result == expected_job_status


def test_launcher_run_fails(mock_registry):
    with pytest.raises(ValueError) as exception_info:
        mock_cloud = Mock(spec=BaseCloud)

        def _builder():
            return mock_cloud

        mock_registry.get_all.return_value = {
            "custom": _builder,
        }
        mock_cloud.get_cluster.return_value = None
        launcher = Launcher()
        job = _get_default_job("custom")
        launcher.run(job, "custom")
    assert "not found" in str(exception_info.value)


def test_launcher_cancel_succeeds(mock_registry):
    mock_cluster = Mock(spec=BaseCluster)
    mock_cloud = Mock(spec=BaseCloud)

    def _builder():
        return mock_cloud

    mock_registry.get_all.return_value = {
        "cloud": _builder,
    }
    expected_job_status = JobStatus(
        id="job_id",
        cluster="cluster",
        name="foo",
        status="canceled",
        metadata="bar",
        done=False,
    )
    mock_cloud.get_cluster.return_value = mock_cluster
    mock_cluster.cancel_job.return_value = expected_job_status
    launcher = Launcher()
    result = launcher.cancel("1", "cloud", "cluster")
    mock_cloud.get_cluster.assert_called_once_with("cluster")
    mock_cluster.cancel_job.assert_called_once_with("1")
    assert result == expected_job_status


def test_launcher_cancel_fails(mock_registry):
    with pytest.raises(ValueError) as exception_info:
        mock_cloud = Mock(spec=BaseCloud)

        def _builder():
            return mock_cloud

        mock_registry.get_all.return_value = {
            "cloud": _builder,
        }
        mock_cloud.get_cluster.return_value = None
        launcher = Launcher()
        launcher.cancel("1", "cloud", "cluster")
    assert "not found" in str(exception_info.value)


def test_launcher_down_succeeds(mock_registry):
    mock_cluster = Mock(spec=BaseCluster)
    mock_cloud = Mock(spec=BaseCloud)

    def _builder():
        return mock_cloud

    mock_registry.get_all.return_value = {
        "cloud": _builder,
    }
    mock_cloud.get_cluster.return_value = mock_cluster
    launcher = Launcher()
    launcher.down("cloud", "cluster")
    mock_cloud.get_cluster.assert_called_once_with("cluster")
    mock_cluster.down.assert_called_once()


def test_launcher_down_fails(mock_registry):
    with pytest.raises(ValueError) as exception_info:
        mock_cloud = Mock(spec=BaseCloud)

        def _builder():
            return mock_cloud

        mock_registry.get_all.return_value = {
            "cloud": _builder,
        }
        mock_cloud.get_cluster.return_value = None
        launcher = Launcher()
        launcher.down("cloud", "cluster")
    assert "not found" in str(exception_info.value)


def test_launcher_status_multiple_clouds(mock_registry):
    sky_mock = Mock(spec=BaseCloud)
    polaris_mock = Mock(spec=BaseCloud)
    custom_mock = Mock(spec=BaseCloud)

    def _sky_builder():
        return sky_mock

    def _polaris_builder():
        return polaris_mock

    def _custom_builder():
        return custom_mock

    mock_registry.get_all.return_value = {
        "sky": _sky_builder,
        "polaris": _polaris_builder,
        "custom": _custom_builder,
    }
    mock_sky_cluster1 = Mock(spec=BaseCluster)
    mock_sky_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
        JobStatus(
            id="2",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_sky_cluster2 = Mock(spec=BaseCluster)
    mock_sky_cluster2.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky2",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster1 = Mock(spec=BaseCluster)
    mock_polaris_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster2 = Mock(spec=BaseCluster)
    mock_polaris_cluster2.get_jobs.return_value = []
    mock_polaris_cluster3 = Mock(spec=BaseCluster)
    mock_polaris_cluster3.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris3",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    sky_mock.list_clusters.return_value = [mock_sky_cluster1, mock_sky_cluster2]
    polaris_mock.list_clusters.return_value = [
        mock_polaris_cluster1,
        mock_polaris_cluster2,
        mock_polaris_cluster3,
    ]
    custom_mock.list_clusters.return_value = []
    launcher = Launcher()
    statuses = launcher.status()
    assert statuses == {
        "custom": [],
        "sky": [
            JobStatus(
                id="1",
                cluster="sky1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="2",
                cluster="sky1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="1",
                cluster="sky2",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
        ],
        "polaris": [
            JobStatus(
                id="1",
                cluster="polaris1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="1",
                cluster="polaris3",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
        ],
    }


def test_launcher_status_filters_clusters(mock_registry):
    sky_mock = Mock(spec=BaseCloud)
    polaris_mock = Mock(spec=BaseCloud)
    custom_mock = Mock(spec=BaseCloud)

    def _sky_builder():
        return sky_mock

    def _polaris_builder():
        return polaris_mock

    def _custom_builder():
        return custom_mock

    mock_registry.get_all.return_value = {
        "sky": _sky_builder,
        "polaris": _polaris_builder,
        "custom": _custom_builder,
    }
    mock_sky_cluster1 = Mock(spec=BaseCluster)
    mock_sky_cluster1.name.return_value = "sky1"
    mock_sky_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
        JobStatus(
            id="2",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_sky_cluster2 = Mock(spec=BaseCluster)
    mock_sky_cluster2.name.return_value = "sky2"
    mock_sky_cluster2.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky2",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster1 = Mock(spec=BaseCluster)
    mock_polaris_cluster1.name.return_value = "polaris1"
    mock_polaris_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster2 = Mock(spec=BaseCluster)
    mock_polaris_cluster2.name.return_value = "polaris2"
    mock_polaris_cluster2.get_jobs.return_value = []
    mock_polaris_cluster3 = Mock(spec=BaseCluster)
    mock_polaris_cluster3.name.return_value = "polaris3"
    mock_polaris_cluster3.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris3",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    sky_mock.list_clusters.return_value = [mock_sky_cluster1, mock_sky_cluster2]
    polaris_mock.list_clusters.return_value = [
        mock_polaris_cluster1,
        mock_polaris_cluster2,
        mock_polaris_cluster3,
    ]
    custom_mock.list_clusters.return_value = []
    launcher = Launcher()
    statuses = launcher.status(cluster="sky1")
    assert statuses == {
        "custom": [],
        "sky": [
            JobStatus(
                id="1",
                cluster="sky1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="2",
                cluster="sky1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
        ],
        "polaris": [],
    }


def test_launcher_status_filters_jobs(mock_registry):
    sky_mock = Mock(spec=BaseCloud)
    polaris_mock = Mock(spec=BaseCloud)
    custom_mock = Mock(spec=BaseCloud)

    def _sky_builder():
        return sky_mock

    def _polaris_builder():
        return polaris_mock

    def _custom_builder():
        return custom_mock

    mock_registry.get_all.return_value = {
        "sky": _sky_builder,
        "polaris": _polaris_builder,
        "custom": _custom_builder,
    }
    mock_sky_cluster1 = Mock(spec=BaseCluster)
    mock_sky_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
        JobStatus(
            id="2",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_sky_cluster2 = Mock(spec=BaseCluster)
    mock_sky_cluster2.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky2",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster1 = Mock(spec=BaseCluster)
    mock_polaris_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster2 = Mock(spec=BaseCluster)
    mock_polaris_cluster2.get_jobs.return_value = []
    mock_polaris_cluster3 = Mock(spec=BaseCluster)
    mock_polaris_cluster3.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris3",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    sky_mock.list_clusters.return_value = [mock_sky_cluster1, mock_sky_cluster2]
    polaris_mock.list_clusters.return_value = [
        mock_polaris_cluster1,
        mock_polaris_cluster2,
        mock_polaris_cluster3,
    ]
    custom_mock.list_clusters.return_value = []
    launcher = Launcher()
    statuses = launcher.status(id="1")
    assert statuses == {
        "custom": [],
        "sky": [
            JobStatus(
                id="1",
                cluster="sky1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="1",
                cluster="sky2",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
        ],
        "polaris": [
            JobStatus(
                id="1",
                cluster="polaris1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="1",
                cluster="polaris3",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
        ],
    }


def test_launcher_status_filters_clouds(mock_registry):
    sky_mock = Mock(spec=BaseCloud)
    polaris_mock = Mock(spec=BaseCloud)
    custom_mock = Mock(spec=BaseCloud)

    def _sky_builder():
        return sky_mock

    def _polaris_builder():
        return polaris_mock

    def _custom_builder():
        return custom_mock

    mock_registry.get_all.return_value = {
        "sky": _sky_builder,
        "polaris": _polaris_builder,
        "custom": _custom_builder,
    }
    mock_sky_cluster1 = Mock(spec=BaseCluster)
    mock_sky_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
        JobStatus(
            id="2",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_sky_cluster2 = Mock(spec=BaseCluster)
    mock_sky_cluster2.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky2",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster1 = Mock(spec=BaseCluster)
    mock_polaris_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster2 = Mock(spec=BaseCluster)
    mock_polaris_cluster2.get_jobs.return_value = []
    mock_polaris_cluster3 = Mock(spec=BaseCluster)
    mock_polaris_cluster3.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris3",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    sky_mock.list_clusters.return_value = [mock_sky_cluster1, mock_sky_cluster2]
    polaris_mock.list_clusters.return_value = [
        mock_polaris_cluster1,
        mock_polaris_cluster2,
        mock_polaris_cluster3,
    ]
    custom_mock.list_clusters.return_value = []
    launcher = Launcher()
    statuses = launcher.status(cloud="sky")
    assert statuses == {
        "sky": [
            JobStatus(
                id="1",
                cluster="sky1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="2",
                cluster="sky1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="1",
                cluster="sky2",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
        ],
    }


def test_launcher_status_all_filters(mock_registry):
    sky_mock = Mock(spec=BaseCloud)
    polaris_mock = Mock(spec=BaseCloud)
    custom_mock = Mock(spec=BaseCloud)

    def _sky_builder():
        return sky_mock

    def _polaris_builder():
        return polaris_mock

    def _custom_builder():
        return custom_mock

    mock_registry.get_all.return_value = {
        "sky": _sky_builder,
        "polaris": _polaris_builder,
        "custom": _custom_builder,
    }
    mock_sky_cluster1 = Mock(spec=BaseCluster)
    mock_sky_cluster1.name.return_value = "foo"
    mock_sky_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="foo",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
        JobStatus(
            id="2",
            cluster="foo",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_sky_cluster2 = Mock(spec=BaseCluster)
    mock_sky_cluster2.name.return_value = "sky2"
    mock_sky_cluster2.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky2",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster1 = Mock(spec=BaseCluster)
    mock_polaris_cluster1.name.return_value = "foo"
    mock_polaris_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="foo",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster2 = Mock(spec=BaseCluster)
    mock_polaris_cluster2.name.return_value = "polaris2"
    mock_polaris_cluster2.get_jobs.return_value = []
    mock_polaris_cluster3 = Mock(spec=BaseCluster)
    mock_polaris_cluster3.name.return_value = "polaris3"
    mock_polaris_cluster3.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris3",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    sky_mock.list_clusters.return_value = [mock_sky_cluster1, mock_sky_cluster2]
    polaris_mock.list_clusters.return_value = [
        mock_polaris_cluster1,
        mock_polaris_cluster2,
        mock_polaris_cluster3,
    ]
    custom_mock.list_clusters.return_value = []
    launcher = Launcher()
    statuses = launcher.status(cluster="foo", cloud="sky", id="1")
    assert statuses == {
        "sky": [
            JobStatus(
                id="1",
                cluster="foo",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
        ],
    }


def test_launcher_status_inits_new_clouds(mock_registry):
    sky_mock = Mock(spec=BaseCloud)
    polaris_mock = Mock(spec=BaseCloud)
    custom_mock = Mock(spec=BaseCloud)

    def _sky_builder():
        return sky_mock

    def _polaris_builder():
        return polaris_mock

    def _custom_builder():
        return custom_mock

    mock_registry.get_all.side_effect = [
        {},
        {},
        {
            "sky": _sky_builder,
            "polaris": _polaris_builder,
            "custom": _custom_builder,
        },
    ]
    mock_sky_cluster1 = Mock(spec=BaseCluster)
    mock_sky_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
        JobStatus(
            id="2",
            cluster="sky1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_sky_cluster2 = Mock(spec=BaseCluster)
    mock_sky_cluster2.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="sky2",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster1 = Mock(spec=BaseCluster)
    mock_polaris_cluster1.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris1",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    mock_polaris_cluster2 = Mock(spec=BaseCluster)
    mock_polaris_cluster2.get_jobs.return_value = []
    mock_polaris_cluster3 = Mock(spec=BaseCluster)
    mock_polaris_cluster3.get_jobs.return_value = [
        JobStatus(
            id="1",
            cluster="polaris3",
            name="foo",
            status="running",
            metadata="bar",
            done=False,
        ),
    ]
    sky_mock.list_clusters.return_value = [mock_sky_cluster1, mock_sky_cluster2]
    polaris_mock.list_clusters.return_value = [
        mock_polaris_cluster1,
        mock_polaris_cluster2,
        mock_polaris_cluster3,
    ]
    custom_mock.list_clusters.return_value = []
    launcher = Launcher()
    statuses = launcher.status()
    # On the first call, statuses should be empty.
    assert statuses == {}
    # On the second call, we've registered new clouds that yield jobs.
    new_statuses = launcher.status()
    assert new_statuses == {
        "custom": [],
        "sky": [
            JobStatus(
                id="1",
                cluster="sky1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="2",
                cluster="sky1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="1",
                cluster="sky2",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
        ],
        "polaris": [
            JobStatus(
                id="1",
                cluster="polaris1",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
            JobStatus(
                id="1",
                cluster="polaris3",
                name="foo",
                status="running",
                metadata="bar",
                done=False,
            ),
        ],
    }


def test_launcher_stop_succeeds(mock_registry):
    mock_cluster = Mock(spec=BaseCluster)
    mock_cloud = Mock(spec=BaseCloud)

    def _builder():
        return mock_cloud

    mock_registry.get_all.return_value = {
        "cloud": _builder,
    }
    mock_cloud.get_cluster.return_value = mock_cluster
    launcher = Launcher()
    launcher.stop("cloud", "cluster")
    mock_cloud.get_cluster.assert_called_once_with("cluster")
    mock_cluster.stop.assert_called_once()


def test_launcher_stop_fails(mock_registry):
    with pytest.raises(ValueError) as exception_info:
        mock_cloud = Mock(spec=BaseCloud)

        def _builder():
            return mock_cloud

        mock_registry.get_all.return_value = {
            "cloud": _builder,
        }
        mock_cloud.get_cluster.return_value = None
        launcher = Launcher()
        launcher.stop("cloud", "cluster")
    assert "not found" in str(exception_info.value)


def test_launcher_which_clouds_updates_over_time(mock_registry):
    sky_mock = Mock(spec=BaseCloud)
    polaris_mock = Mock(spec=BaseCloud)
    custom_mock = Mock(spec=BaseCloud)

    def _sky_builder():
        return sky_mock

    def _polaris_builder():
        return polaris_mock

    def _custom_builder():
        return custom_mock

    mock_registry.get_all.side_effect = [
        {},
        {
            "sky": _sky_builder,
        },
        {
            "polaris": _polaris_builder,
        },
        {
            "sky": _sky_builder,
            "polaris": _polaris_builder,
            "custom": _custom_builder,
        },
    ]
    launcher = Launcher()
    assert launcher.which_clouds() == ["sky"]
    assert launcher.which_clouds() == ["polaris"]
    assert launcher.which_clouds() == ["sky", "polaris", "custom"]


def test_launcher_export_methods(mock_registry):
    assert LAUNCHER.up == up
    assert LAUNCHER.run == run
    assert LAUNCHER.cancel == cancel
    assert LAUNCHER.down == down
    assert LAUNCHER.status == status
    assert LAUNCHER.stop == stop
    assert LAUNCHER.get_cloud == get_cloud
    assert LAUNCHER.which_clouds == which_clouds


def _verify_no_extra_import(extra_module: str):
    """Verifies that extra modules are not imported."""
    import sys

    import oumi.launcher  # noqa

    assert extra_module not in sys.modules, f"{extra_module} was imported."


def test_launcher_no_sky_dependency():
    # Ensure that sky is lazy loaded so it doesn't cause DB contention in multinode
    # jobs: https://github.com/oumi-ai/oumi/issues/1605

    set_start_method("spawn", force=True)
    process = Process(target=_verify_no_extra_import, args=["sky"])
    process.start()
    process.join()
    assert process.exitcode == 0, (
        "Sky was imported as part of the launcher module. This is a regression."
    )
