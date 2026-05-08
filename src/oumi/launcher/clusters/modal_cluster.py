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

A "cluster" on Modal corresponds to a single ``FunctionCall``. Modal is
serverless — there is no shared compute pool to bring up or down. ``stop()``
and ``down()`` therefore best-effort cancel the call if it is still in
flight; once the call is terminal, both methods are no-ops.
"""

from __future__ import annotations

from typing import Any

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCluster, JobStatus
from oumi.launcher.clients.modal_client import ModalClient, ModalLogStream


class ModalCluster(BaseCluster):
    """A cluster implementation backed by a single Modal FunctionCall."""

    def __init__(self, name: str, client: ModalClient) -> None:
        """Initializes a new instance of the ModalCluster class.

        Args:
            name: The Modal ``FunctionCall.object_id`` used as the cluster
                identifier.
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
        """Gets the name (FunctionCall ID) of the cluster."""
        return self._name

    def get_job(self, job_id: str) -> JobStatus | None:
        """Gets the single job tracked by this cluster, if it matches.

        Modal clusters host exactly one FunctionCall (the one named after the
        cluster). Anything else returns ``None``.
        """
        if job_id != self._name:
            return None
        return self._client.get_status(self._name)

    def get_jobs(self) -> list[JobStatus]:
        """Lists the jobs on this cluster (always exactly one for Modal)."""
        return [self._client.get_status(self._name)]

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancels the FunctionCall tied to this cluster."""
        if job_id != self._name:
            raise RuntimeError(f"Job {job_id} not found on cluster {self._name}.")
        self._client.cancel(self._name)
        return self._client.get_status(self._name)

    def run_job(self, job: JobConfig) -> JobStatus:
        """Re-running on a Modal "cluster" is unsupported.

        Modal jobs are 1:1 with FunctionCalls. To run a new job, allocate a
        new cluster via ``ModalCloud.up_cluster``.
        """
        raise NotImplementedError(
            "Modal does not support re-running jobs on an existing cluster. "
            "Call ModalCloud.up_cluster(...) to spawn a new FunctionCall."
        )

    def stop(self) -> None:
        """Best-effort cancel — Modal has no separate stop semantics."""
        self._client.cancel(self._name)

    def down(self) -> None:
        """Best-effort cancel — Modal is serverless so there is nothing to tear down."""
        self._client.cancel(self._name)

    def get_logs_stream(
        self, cluster_name: str, job_id: str | None = None
    ) -> ModalLogStream:
        """Returns a stream of logs for the underlying FunctionCall.

        ``cluster_name`` and ``job_id`` are accepted for interface
        compatibility but ignored — a Modal cluster has exactly one job.
        """
        return self._client.get_logs_stream(self._name)
