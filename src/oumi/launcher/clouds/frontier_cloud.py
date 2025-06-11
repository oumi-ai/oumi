# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from dataclasses import dataclass
from typing import Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCloud, BaseCluster, JobStatus
from oumi.core.registry import register_cloud_builder
from oumi.launcher.clients.slurm_client import SlurmClient
from oumi.launcher.clusters.frontier_cluster import FrontierCluster
from oumi.utils.logging import logger


@dataclass
class _ClusterInfo:
    """Dataclass to hold information about a cluster."""

    queue: str
    user: str

    @property
    def name(self):
        return f"{self.queue}.{self.user}"


_FRONTIER_HOSTNAME = "frontier.olcf.ornl.gov"


class FrontierCloud(BaseCloud):
    """A resource pool for managing the OLCF Frontier job queues."""

    def __init__(self):
        """Initializes a new instance of the FrontierCloud class."""
        # A mapping from user names to Frontier Clients.
        self._clients = collections.OrderedDict()
        # A mapping from cluster names to Frontier Cluster instances.
        self._clusters = collections.OrderedDict()

        # Check if any users have open SSH tunnels to Frontier.
        for user in SlurmClient.get_active_users(_FRONTIER_HOSTNAME):
            self.initialize_clusters(user)

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
                f"Invalid cluster name: {name}. Must be in the format 'queue.user'."
            )
        queue, user = name_splits
        return _ClusterInfo(queue, user)

    def _get_or_create_client(self, cluster_info: _ClusterInfo) -> SlurmClient:
        """Gets the client for the specified user, or creates one if it doesn't exist.

        Args:
            cluster_info: The cluster information.

        Returns:
            SlurmClient: The client instance.
        """
        if cluster_info.user not in self._clients:
            self._clients[cluster_info.user] = SlurmClient(
                cluster_info.user, _FRONTIER_HOSTNAME, cluster_info.name
            )
        return self._clients[cluster_info.user]

    def _get_or_create_cluster(self, name: str) -> FrontierCluster:
        """Gets the cluster with the specified name, or creates one if it doesn't exist.

        Args:
            name: The name of the cluster.

        Returns:
            FrontierCluster: The cluster instance.
        """
        if name not in self._clusters:
            cluster_info = self._parse_cluster_name(name)
            self._clusters[name] = FrontierCluster(
                name, self._get_or_create_client(cluster_info)
            )
        return self._clusters[name]

    def initialize_clusters(self, user) -> list[BaseCluster]:
        """Initializes clusters for the specified user for all Frontier queues.

        Args:
            user: The user to initialize clusters for.

        Returns:
            List[FrontierCluster]: The list of initialized clusters.
        """
        clusters = []
        for q in sorted({q.value for q in FrontierCluster.SupportedQueues}):
            name = f"{q}.{user}"
            cluster = self._get_or_create_cluster(name)
            clusters.append(cluster)
        return clusters

    def up_cluster(self, job: JobConfig, name: Optional[str], **kwargs) -> JobStatus:
        """Creates a cluster and starts the provided Job."""
        if not job.user:
            raise ValueError("User must be provided in the job config.")
        # The default queue is BATCH.
        cluster_info = _ClusterInfo(
            FrontierCluster.SupportedQueues.BATCH.value, job.user
        )
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
                f"{FrontierCluster.SupportedQueues.BATCH.value}."
            )
        cluster = self._get_or_create_cluster(cluster_info.name)
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

    def list_clusters(self) -> list[BaseCluster]:
        """Lists the active clusters on this cloud."""
        return list(self._clusters.values())


@register_cloud_builder("frontier")
def frontier_cloud_builder() -> FrontierCloud:
    """Builds a FrontierCloud instance."""
    return FrontierCloud()
