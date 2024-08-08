from typing import List, Optional

from lema.core.registry import register_cloud_builder
from lema.core.types.base_cloud import BaseCloud
from lema.core.types.base_cluster import BaseCluster, JobStatus
from lema.core.types.configs import JobConfig
from lema.launcher.clients.local_client import LocalClient
from lema.launcher.clusters.local_cluster import LocalCluster


class LocalCloud(BaseCloud):
    """A resource pool for managing Local jobs.

    It is important to note that a single LocalCluster can only run one job at a time.
    Running multiple GPU jobs simultaneously on separate LocalClusters is not
    encouraged.
    """

    # The default cluster name. Used when no cluster name is provided.
    _DEFAULT_CLUSTER = "local"

    def __init__(self):
        """Initializes a new instance of the LocalCloud class."""
        # A mapping from cluster names to Local Cluster instances.
        self._clusters = {}

    def _get_or_create_cluster(self, name: str) -> LocalCluster:
        """Gets the cluster with the specified name, or creates one if it doesn't exist.

        Args:
            name: The name of the cluster.

        Returns:
            LocalCluster: The cluster instance.
        """
        if name not in self._clusters:
            self._clusters[name] = LocalCluster(name, LocalClient())
        return self._clusters[name]

    def up_cluster(self, job: JobConfig, name: Optional[str]) -> JobStatus:
        """Creates a cluster and starts the provided Job."""
        # The default cluster.
        cluster_name = name or self._DEFAULT_CLUSTER
        cluster = self._get_or_create_cluster(cluster_name)
        job_status = cluster.run_job(job)
        if not job_status:
            raise RuntimeError("Failed to start job.")
        return job_status

    def get_cluster(self, name) -> Optional[BaseCluster]:
        """Gets the cluster with the specified name, or None if not found."""
        clusters = self.list_clusters()
        for cluster in clusters:
            if cluster.name() == name:
                return cluster
        return None

    def list_clusters(self) -> List[BaseCluster]:
        """Lists the active clusters on this cloud."""
        return list(self._clusters.values())


@register_cloud_builder("local")
def Local_cloud_builder() -> LocalCloud:
    """Builds a LocalCloud instance."""
    return LocalCloud()
