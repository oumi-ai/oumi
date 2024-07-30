from typing import List, Optional, Type, TypeVar

import sky

from lema.core.registry import register_cloud_builder
from lema.core.types.base_cloud import BaseCloud
from lema.core.types.base_cluster import BaseCluster
from lema.core.types.configs import JobConfig
from lema.launcher.clients.sky_client import SkyClient
from lema.launcher.clusters.sky_cluster import SkyCluster

T = TypeVar("T")


class SkyCloud(BaseCloud):
    """A resource pool capable of creating clusters using Sky Pilot."""

    def __init__(self, cloud_name: str, client: SkyClient):
        """Initializes a new instance of the SkyCloud class."""
        self._cloud_name = cloud_name
        self._client = client

    def _get_clusters_by_class(self, cloud_class: Type[T]) -> List[BaseCluster]:
        """Gets the appropriate clusters of type T."""
        return [
            SkyCluster(cluster["name"], self._client)
            for cluster in self._client.status()
            if isinstance(cluster["handle"].launched_resources.cloud, cloud_class)
        ]

    def up_cluster(self, job: JobConfig, name: Optional[str]) -> BaseCluster:
        """Creates a cluster and starts the provided Job."""
        cluster_name = self._client.launch(job, name)
        return SkyCluster(cluster_name, self._client)

    def get_cluster(self, name) -> Optional[BaseCluster]:
        """Gets the cluster with the specified name, or None if not found."""
        clusters = self.list_clusters()
        for cluster in clusters:
            if cluster.name() == name:
                return cluster
        return None

    def list_clusters(self) -> List[BaseCluster]:
        """Lists the active clusters on this cloud."""
        if self._cloud_name == SkyClient.SupportedClouds.GCP.value:
            return self._get_clusters_by_class(sky.clouds.GCP)
        elif self._cloud_name == SkyClient.SupportedClouds.RUNPOD.value:
            return self._get_clusters_by_class(sky.clouds.RunPod)
        elif self._cloud_name == SkyClient.SupportedClouds.LAMBDA.value:
            return self._get_clusters_by_class(sky.clouds.Lambda)
        raise ValueError(f"Unsupported cloud: {self._cloud_name}")


@register_cloud_builder("runpod")
def runpod_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for runpod."""
    return SkyCloud(SkyClient.SupportedClouds.RUNPOD.value, SkyClient())


@register_cloud_builder("gcp")
def gcp_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for Google Cloud Platform."""
    return SkyCloud(SkyClient.SupportedClouds.GCP.value, SkyClient())


@register_cloud_builder("lambda")
def lambda_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for Lambda."""
    return SkyCloud(SkyClient.SupportedClouds.LAMBDA.value, SkyClient())
