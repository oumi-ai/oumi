from dataclasses import dataclass
from typing import List, Optional

from lema.core.configs import JobConfig
from lema.core.launcher import BaseCloud, BaseCluster, JobStatus
from lema.core.registry import register_cloud_builder
from lema.launcher.clients.polaris_client import PolarisClient
from lema.launcher.clusters.polaris_cluster import PolarisCluster
from lema.utils.logging import logger


@dataclass
class _ClusterInfo:
    """Dataclass to hold information about a cluster."""

    queue: str
    user: str

    def name(self):
        return f"{self.queue}.{self.user}"


class PolarisCloud(BaseCloud):
    """A resource pool for managing the Polaris ALCF job queues."""

    def __init__(self):
        """Initializes a new instance of the PolarisCloud class."""
        # A mapping from user names to Polaris Clients.
        self._clients = {}
        # A mapping from cluster names to Polaris Cluster instances.
        self._clusters = {}

    def _parse_cluster_name(self, name: str) -> _ClusterInfo:
        """Parses the cluster name into queue and user components.

        Args:
            name: The name of the cluster.

        Returns:
            _ClusterInfo: The parsed cluster information.
        """
        name_splits = name.split(".")
        if len(name_splits) != 2:
            raise ValueError(
                f"Invalid cluster name: {name}. " "Must be in the format 'queue.user'."
            )
        queue, user = name_splits
        return _ClusterInfo(queue, user)

    def _get_or_create_client(self, user: str) -> PolarisClient:
        """Gets the client for the specified user, or creates one if it doesn't exist.

        Args:
            user: The user to get the client for.

        Returns:
            PolarisClient: The client instance.
        """
        if user not in self._clients:
            self._clients[user] = PolarisClient(user)
        return self._clients[user]

    def _get_or_create_cluster(self, name: str) -> PolarisCluster:
        """Gets the cluster with the specified name, or creates one if it doesn't exist.

        Args:
            name: The name of the cluster.

        Returns:
            PolarisCluster: The cluster instance.
        """
        if name not in self._clusters:
            cluster_info = self._parse_cluster_name(name)
            self._clusters[name] = PolarisCluster(
                name, self._get_or_create_client(cluster_info.user)
            )
        return self._clusters[name]

    def initialize_clusters(self, user) -> List[BaseCluster]:
        """Initializes clusters for the specified user for all Polaris queues.

        Args:
            user: The user to initialize clusters for.

        Returns:
            List[PolarisCluster]: The list of initialized clusters.
        """
        clusters = []
        queue_set = {q.value for q in PolarisClient.SupportedQueues}
        for q in queue_set:
            name = f"{q}.{user}"
            cluster = self._get_or_create_cluster(name)
            clusters.append(cluster)
        return clusters

    def up_cluster(self, job: JobConfig, name: Optional[str]) -> JobStatus:
        """Creates a cluster and starts the provided Job."""
        if not job.user:
            raise ValueError("User must be provided in the job config.")
        # The default queue is PROD.
        cluster_info = _ClusterInfo(PolarisClient.SupportedQueues.PROD.value, job.user)
        if name:
            cluster_info = self._parse_cluster_name(name)
            if cluster_info.user != job.user:
                raise ValueError(
                    f"Invalid cluster name: {name}. "
                    "User must match the provided job user."
                )
        else:
            logger.warning(
                "No cluster name provided. Using default queue: "
                f"{PolarisClient.SupportedQueues.PROD.value}."
            )
        cluster = self._get_or_create_cluster(cluster_info.name())
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


@register_cloud_builder("polaris")
def polaris_cloud_builder() -> PolarisCloud:
    """Builds a PolarisCloud instance."""
    return PolarisCloud()
