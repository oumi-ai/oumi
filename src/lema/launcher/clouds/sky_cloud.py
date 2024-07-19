from typing import List, Optional, Type, TypeVar

import sky

from lema.core.types.base_cloud import BaseCloud
from lema.core.types.base_cluster import BaseCluster
from lema.core.types.configs import JobConfig
from lema.launcher.clients.sky_client import SkyClient
from lema.launcher.clusters.sky_cluster import SkyCluster

T = TypeVar("T")


class SkyCloud(BaseCloud):
    """A resource pool capable of creating clusters using Sky Pilot."""

    def __init__(self, cloud: str, client: SkyClient):
        """Initializes a new instance of the SkyCloud class."""
        self._cloud = cloud
        self._client = client

    def _get_clusters_by_class(self, cloud_class: Type[T]) -> List[BaseCluster]:
        """Gets the GCP clusters."""
        return [
            SkyCluster(cluster["name"], self._client)
            for cluster in self._client.status()
            if isinstance(cluster["handle"].launched_resources.cloud, cloud_class)
        ]

    def up_cluster(self, job: JobConfig, name: Optional[str]) -> BaseCluster:
        """Creates a cluster and starts the provided Job."""
        cluster_name = self._client.launch(job, name)
        return SkyCluster(cluster_name, self._client)

    def get_cluster(self, name) -> BaseCluster:
        """Gets the cluster with the specified name."""
        clusters = self.list_clusters()
        for cluster in clusters:
            print(cluster)
            if cluster.name == name:
                return cluster
        raise ValueError(f"Cluster {name} not found.")

    def list_clusters(self) -> List[BaseCluster]:
        """List the active clusters on this cloud."""
        if self._cloud == self._client.get_gcp_cloud_name():
            return self._get_clusters_by_class(sky.clouds.GCP)
        elif self._cloud == self._client.get_runpod_cloud_name():
            return self._get_clusters_by_class(sky.clouds.RunPod)
        raise ValueError(f"Unsupported cloud: {self._cloud}")
