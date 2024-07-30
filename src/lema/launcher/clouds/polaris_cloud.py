from typing import List, Optional, TypeVar

from lema.core.types.base_cloud import BaseCloud
from lema.core.types.base_cluster import BaseCluster
from lema.core.types.configs import JobConfig
from lema.launcher.clients.polaris_client import PolarisClient
from lema.launcher.clusters.polaris_cluster import PolarisCluster

T = TypeVar("T")


class PolarisCloud(BaseCloud):
    """A resource pool for managing the Polaris ALCF job queues."""

    def __init__(self, cloud_name: str):
        """Initializes a new instance of the PolarisCloud class."""
        self._cloud_name = cloud_name
        self._clusters = {}

    def up_cluster(self, job: JobConfig, name: Optional[str]) -> BaseCluster:
        """Creates a cluster and starts the provided Job."""
        return PolarisCluster(self._cloud_name, PolarisClient(self._cloud_name))

    def get_cluster(self, name) -> Optional[BaseCluster]:
        """Gets the cluster with the specified name, or None if not found."""
        clusters = self.list_clusters()
        for cluster in clusters:
            if cluster.name() == name:
                return cluster
        return None

    def list_clusters(self) -> List[BaseCluster]:
        """Lists the active clusters on this cloud."""
        raise ValueError(f"Unsupported cloud: {self._cloud_name}")
