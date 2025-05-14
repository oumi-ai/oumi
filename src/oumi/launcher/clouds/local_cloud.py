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
from oumi.launcher.clients.local_client import LocalClient
from oumi.launcher.clusters.local_cluster import LocalCluster


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

    def up_cluster(self, job: JobConfig, name: Optional[str], **kwargs) -> JobStatus:
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

    def list_clusters(self) -> list[BaseCluster]:
        """Lists the active clusters on this cloud."""
        return list(self._clusters.values())


@register_cloud_builder("local")
def Local_cloud_builder() -> LocalCloud:
    """Builds a LocalCloud instance."""
    return LocalCloud()
