from unittest.mock import Mock

import pytest

from lema.core.types.configs import JobConfig
from lema.core.types.params.node_params import DiskTier, NodeParams, StorageMount
from lema.launcher.clients.sky_client import SkyClient
from lema.launcher.clouds.sky_cloud import SkyCloud


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
    _ = cloud.up_cluster(_get_default_job("gcp"), "new_cluster_name")
    mock_sky_client.launch.assert_called_once_with(
        _get_default_job("gcp"), "new_cluster_name"
    )
    # assert cluster.name() == "new_cluster_name"
