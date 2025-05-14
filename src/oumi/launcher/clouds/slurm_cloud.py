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

from typing import Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCloud, BaseCluster, JobStatus
from oumi.core.registry import register_cloud_builder
from oumi.launcher.clients.slurm_client import SlurmClient
from oumi.launcher.clusters.slurm_cluster import SlurmCluster


class SlurmCloud(BaseCloud):
    """A resource pool for managing jobs in Slurm clusters."""

    def __init__(self):
        """Initializes a new instance of the SlurmCloud class."""
        # A mapping from cluster names to Slurm Cluster instances.
        self._clusters = {}

        # Initialize default connections.
        self.initialize_clusters()

    def _get_or_create_cluster(self, name: str) -> SlurmCluster:
        """Gets the cluster with the specified name, or creates one if it doesn't exist.

        Args:
            name: The name of the cluster.

        Returns:
            SlurmCluster: The cluster instance.
        """
        if name not in self._clusters:
            cluster_info = SlurmCluster.parse_cluster_name(name)
            self._clusters[name] = SlurmCluster(
                name,
                SlurmClient(
                    user=cluster_info.user,
                    slurm_host=cluster_info.hostname,
                    cluster_name=cluster_info.name,
                ),
            )
        return self._clusters[name]

    def initialize_clusters(self) -> list[BaseCluster]:
        """Initializes clusters for the specified user for all Slurm queues.

        Returns:
            List[SlurmCluster]: The list of initialized clusters.
        """
        connections = SlurmCluster.get_slurm_connections()
        clusters = []
        for c in connections:
            cluster = self._get_or_create_cluster(c.name)
            clusters.append(cluster)
        return clusters

    def up_cluster(self, job: JobConfig, name: Optional[str], **kwargs) -> JobStatus:
        """Creates a cluster and starts the provided Job."""
        if not job.user:
            raise ValueError("User must be provided in the job config.")
        if name:
            cluster_info = SlurmCluster.parse_cluster_name(name)
            if cluster_info.user != job.user:
                raise ValueError(
                    f"Invalid cluster name: `{name}`. "
                    f"User must match the provided job user: `{job.user}`."
                )
        else:
            raise ValueError(
                "A cluster name must be provided for Slurm. "
                "Cluster names are of the form 'user@hostname'."
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


@register_cloud_builder("slurm")
def slurm_cloud_builder() -> SlurmCloud:
    """Builds a SlurmCloud instance."""
    return SlurmCloud()
