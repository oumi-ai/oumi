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

"""Modal-backed cluster implementation.

Modal has no native cluster concept — every job is a single ``Sandbox``.
``ModalCluster`` is a thin façade that maps a logical cluster name (the
SkyPilot-style identifier callers like the Oumi worker pass to
``oumi.launcher.up``) onto sandbox lookups by ``object_id``. Job lookups
use the ``job_id`` argument directly so callers don't need to know the
mapping.

``stop()`` and ``down()`` cancel every sandbox the in-process
``ModalClient`` has launched under this cluster name. Across worker
restarts the mapping is lost; cleanup at that point should fall back
to per-sandbox ``cancel_job`` using the ``job_id`` persisted by the
caller alongside the cluster name.
"""

from __future__ import annotations

from typing import Any

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCluster, ClusterNotFoundError, JobStatus
from oumi.launcher.clients.modal_client import ModalClient, ModalLogStream


class ModalCluster(BaseCluster):
    """A cluster implementation backed by Modal sandboxes."""

    def __init__(self, name: str, client: ModalClient) -> None:
        """Initializes a new instance of the ModalCluster class.

        Args:
            name: Logical cluster name (typically the
                ``cluster-job-{project}-{op}-...`` style identifier the
                caller used when invoking ``oumi.launcher.up``).
            client: A configured ``ModalClient``.
        """
        self._name = name
        self._client = client

    def __eq__(self, other: Any) -> bool:
        """Checks if two ModalClusters are equal."""
        if not isinstance(other, ModalCluster):
            return False
        return self.name() == other.name()

    def __hash__(self) -> int:
        """Hashes by cluster name so instances can live in sets/dicts."""
        return hash(self._name)

    def name(self) -> str:
        """Gets the cluster name."""
        return self._name

    def get_job(self, job_id: str) -> JobStatus | None:
        """Gets the status of the sandbox identified by ``job_id``.

        ``job_id`` is the opaque ``Sandbox.object_id`` returned at launch
        time (and persisted by the caller). The cluster name is purely
        logical, so this method ignores ``self._name`` and goes straight
        to the sandbox lookup.
        """
        try:
            return self._client.get_status(job_id)
        except ClusterNotFoundError:
            return None

    def get_jobs(self) -> list[JobStatus]:
        """Lists the jobs spawned under this cluster name in this process."""
        statuses: list[JobStatus] = []
        for sandbox_id in self._client.find_sandboxes_for_cluster(self._name):
            try:
                statuses.append(self._client.get_status(sandbox_id))
            except ClusterNotFoundError:
                continue
        return statuses

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancels the sandbox identified by ``job_id`` and returns its status."""
        self._client.cancel(job_id)
        return self._client.get_status(job_id)

    def run_job(self, job: JobConfig) -> JobStatus:
        """Re-running on a Modal cluster is unsupported.

        Modal jobs are 1:1 with sandboxes. To run a new job, allocate a
        new sandbox via ``ModalCloud.up_cluster``.
        """
        raise NotImplementedError(
            "Modal does not support re-running jobs on an existing cluster. "
            "Call ModalCloud.up_cluster(...) to spawn a new sandbox."
        )

    def stop(self) -> None:
        """Best-effort cancel of every sandbox tracked under this cluster name."""
        for sandbox_id in self._client.find_sandboxes_for_cluster(self._name):
            self._client.cancel(sandbox_id)

    def down(self) -> None:
        """Alias for ``stop`` — Modal is serverless, nothing else to tear down."""
        self.stop()

    def get_logs_stream(
        self, cluster_name: str, job_id: str | None = None
    ) -> ModalLogStream:
        """Returns a stream of logs for ``job_id`` (sandbox object_id).

        ``cluster_name`` is accepted for interface compatibility and
        ignored. ``job_id`` is the canonical handle. If ``job_id`` is
        omitted, falls back to the most recently launched sandbox under
        this cluster name (in this process).
        """
        target_sandbox = job_id
        if target_sandbox is None:
            tracked = self._client.find_sandboxes_for_cluster(self._name)
            if not tracked:
                raise ClusterNotFoundError(
                    f"No sandboxes tracked for cluster '{self._name}' "
                    "and no job_id provided."
                )
            target_sandbox = tracked[-1]
        return self._client.get_logs_stream(target_sandbox)
