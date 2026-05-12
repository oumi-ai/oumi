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

"""Modal-backed :class:`BaseCloud`.

Each :class:`ModalCluster` instance maps to one or more Modal sandboxes
sharing a logical cluster name (the cluster name is a caller-provided
label; the sandboxes are addressed by ``Sandbox.object_id``). The cloud
keeps an in-process registry of clusters it has launched so that
``get_cluster`` / ``list_clusters`` can return them without
round-tripping to Modal's control plane.
"""

from __future__ import annotations

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCloud, BaseCluster, JobStatus
from oumi.core.registry import register_cloud_builder
from oumi.launcher.clients.modal_client import ModalClient
from oumi.launcher.clusters.modal_cluster import ModalCluster


class ModalCloud(BaseCloud):
    """A resource pool capable of dispatching jobs to Modal."""

    def __init__(self) -> None:
        """Initializes a new instance of the ModalCloud class."""
        self._modal_client: ModalClient | None = None
        self._clusters: dict[str, ModalCluster] = {}

    @property
    def _client(self) -> ModalClient:
        """Returns a lazily-instantiated :class:`ModalClient`.

        Delaying instantiation avoids importing ``modal`` (and therefore
        loading credentials) at module import time.
        """
        if self._modal_client is None:
            self._modal_client = ModalClient()
        return self._modal_client

    def up_cluster(self, job: JobConfig, name: str | None, **kwargs) -> JobStatus:
        """Spawns a Modal Sandbox and registers a cluster wrapping it."""
        status = self._client.launch(job, cluster_name=name, **kwargs)
        cluster = ModalCluster(status.cluster, self._client)
        self._clusters[status.cluster] = cluster
        return status

    def get_cluster(self, name: str) -> BaseCluster | None:
        """Gets the cluster with the specified name, or None if not found.

        Falls back to constructing a :class:`ModalCluster` for any cluster
        name we have not seen in this process. The cluster's sandbox
        lookups happen lazily via ``ModalClient.find_sandboxes_for_cluster``
        (Modal tag-based query) so a freshly-constructed instance still
        sees prior sandboxes across worker restarts.
        """
        if name in self._clusters:
            return self._clusters[name]
        # Construct on demand; resolution happens via ModalClient.get_call.
        cluster = ModalCluster(name, self._client)
        self._clusters[name] = cluster
        return cluster

    def list_clusters(self) -> list[BaseCluster]:
        """Lists the clusters tracked by this process."""
        return list(self._clusters.values())


@register_cloud_builder("modal")
def modal_cloud_builder() -> ModalCloud:
    """Builds a ModalCloud instance."""
    return ModalCloud()
