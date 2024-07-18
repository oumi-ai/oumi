from abc import abstractmethod
from typing import List, Optional

import sky
import sky.data

import lema.launcher.utils.sky_utils as sky_utils
from lema.core.types.base_cloud import BaseCloud
from lema.core.types.base_cluster import BaseCluster
from lema.core.types.configs import JobConfig
from lema.launcher.clusters.sky_cluster import SkyCluster


class SkyCloud(BaseCloud):
    """Base class for a resource pool capable of creating clusters."""

    def up_cluster(self, job: JobConfig, name: Optional[str]) -> BaseCluster:
        """Creates a cluster and starts the provided Job."""
        _, resource_handle = sky.launch(
            sky_utils.convert_job_to_task(job), cluster_name=name
        )
        return SkyCluster(resource_handle.name)

    @abstractmethod
    def get_cluster(self, name) -> BaseCluster:
        """Gets the cluster with the specified name."""
        clusters = self.list_clusters()
        for cluster in clusters:
            print(cluster)
            if cluster.name == name:
                return cluster
        raise ValueError(f"Cluster {name} not found.")

    @abstractmethod
    def list_clusters(self) -> List[BaseCluster]:
        """List the active clusters on this cloud."""
        clusters = sky.status()
        print(clusters)
        return [SkyCluster(cluster.name) for cluster in sky.status()]
