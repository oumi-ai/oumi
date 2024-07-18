from typing import List, Optional

import sky
import sky.data

import lema.launcher.utils.sky_utils as sky_utils
from lema.core.types.base_cloud import BaseCloud
from lema.core.types.base_cluster import BaseCluster
from lema.core.types.configs import JobConfig
from lema.core.types.params.node_params import SupportedCloud
from lema.launcher.clusters.sky_cluster import SkyCluster


class SkyCloud(BaseCloud):
    """Base class for a resource pool capable of creating clusters."""

    def __init__(self, backend: SupportedCloud):
        """Initializes a new instance of the SkyCloud class."""
        self._backend = backend

    def _get_gcp_clusters(self) -> List[BaseCluster]:
        """Gets the GCP clusters."""
        return [
            SkyCluster(cluster.name)
            for cluster in sky.status()
            if isinstance(cluster["handle"].launched_resources.cloud, sky.clouds.GCP)
        ]

    def _get_runpod_clusters(self) -> List[BaseCluster]:
        """Gets the RunPod clusters."""
        return [
            SkyCluster(cluster.name)
            for cluster in sky.status()
            if isinstance(cluster["handle"].launched_resources.cloud, sky.clouds.RunPod)
        ]

    def up_cluster(self, job: JobConfig, name: Optional[str]) -> BaseCluster:
        """Creates a cluster and starts the provided Job."""
        _, resource_handle = sky.launch(
            sky_utils.convert_job_to_task(job), cluster_name=name
        )
        resource_handle.launched_resources
        return SkyCluster(resource_handle.name)

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
        if self._backend == SupportedCloud.GCP:
            return self._get_gcp_clusters()
        elif self._backend == SupportedCloud.RUNPOD:
            return self._get_runpod_clusters()
        raise ValueError(f"Unsupported cloud: {self._backend}")
