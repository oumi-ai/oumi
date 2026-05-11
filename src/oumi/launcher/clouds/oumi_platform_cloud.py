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

"""Launcher cloud target that submits jobs to the Oumi Enterprise platform.

Selecting ``cluster=oumi-platform`` in a :class:`JobConfig` submits the job
to the platform's job-submission endpoint and surfaces it back through the
existing :class:`BaseCloud`/:class:`BaseCluster` abstractions: status,
cancel, and log streaming all forward to platform endpoints. No SkyPilot or
local subprocess is involved.

The endpoint shape used here is::

    POST /v1/projects/{project_id}/jobs:submit
    Body: { "kind": "train" | "evaluate" | "synth" | "judge" | "infer",
             "config": <JobConfig serialized to JSON> }

The platform server dispatches to its existing Temporal workflows. Until
that single submission endpoint ships on the platform side, this cloud
implementation surfaces a clear error pointing users at the wider plan.
"""

import io
import json
from dataclasses import asdict
from typing import Any

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCloud, BaseCluster, JobStatus
from oumi.core.launcher.base_cluster import JobState
from oumi.core.registry import register_cloud_builder
from oumi.platform.client import Client
from oumi.platform.exceptions import PlatformAPIError, PlatformError

_CLOUD_NAME = "oumi-platform"
_JOBS_SUBMIT_PATH = "/jobs:submit"


# Map platform operation statuses to launcher JobState values.
_STATUS_TO_STATE: dict[str, JobState] = {
    "unknown": JobState.PENDING,
    "pending": JobState.PENDING,
    "running": JobState.RUNNING,
    "completed": JobState.SUCCEEDED,
    "failed": JobState.FAILED,
    "cancelled": JobState.CANCELLED,
    "cancelling": JobState.RUNNING,
}


def _infer_job_kind(job: JobConfig) -> str:
    """Best-effort guess at what kind of job ``job`` describes.

    The platform's :submit endpoint needs to know which workflow to start.
    We inspect the run command (a shell snippet that may invoke ``oumi
    train``/``oumi evaluate``/etc.) and fall back to ``"train"`` when no
    clear signal is present. Callers that want to force a kind can set
    ``job.envs["OUMI_JOB_KIND"]`` to override.
    """
    explicit = (job.envs or {}).get("OUMI_JOB_KIND")
    if explicit:
        return explicit
    cmd = (job.run or "").lower()
    for kind in ("evaluate", "judge", "synth", "infer", "train"):
        if f"oumi {kind}" in cmd:
            return kind
    return "train"


def _job_config_to_payload(job: JobConfig) -> dict[str, Any]:
    """Serialize ``job`` to a JSON-safe payload for :submit."""
    raw = asdict(job)
    # asdict turns Enum values into their .value already; nothing further to do
    # for the launcher dataclasses, but probe-validate it serializes.
    json.dumps(raw)  # surface non-JSON-safe fields up front
    return raw


class OumiPlatformCluster(BaseCluster):
    """A "cluster" representing a single platform-side operation.

    The operation id is treated as the cluster name and the job id, since the
    platform's job model is one-operation-per-submission rather than a
    long-lived cluster owning many jobs.
    """

    def __init__(self, operation_id: str, client: Client):
        """Initialize a cluster wrapping platform operation ``operation_id``."""
        self._operation_id = operation_id
        self._client = client

    def name(self) -> str:
        """Return the operation id; used as both cluster and job name."""
        return self._operation_id

    def get_job(self, job_id: str) -> JobStatus:
        """Fetch and translate the platform operation into a :class:`JobStatus`."""
        op = self._client.operations.get(job_id)
        return _operation_to_job_status(op, cluster_name=self._operation_id)

    def get_jobs(self) -> list[JobStatus]:
        """Return the single platform operation backing this cluster."""
        return [self.get_job(self._operation_id)]

    def cancel_job(self, job_id: str) -> JobStatus:
        """Request cancellation of the platform operation."""
        op = self._client.operations.stop(job_id)
        return _operation_to_job_status(op, cluster_name=self._operation_id)

    def run_job(self, job: JobConfig) -> JobStatus:  # noqa: ARG002
        """Submitting more jobs onto an existing platform operation is not supported.

        The platform's submission model creates a new operation per job. To
        run another job, call :meth:`OumiPlatformCloud.up_cluster` again with
        a fresh ``JobConfig``.
        """
        raise NotImplementedError(
            "OumiPlatformCluster wraps a single platform operation; submit a "
            "new JobConfig via up_cluster() to start another job."
        )

    def stop(self) -> None:
        """Cancel the underlying platform operation."""
        self.cancel_job(self._operation_id)

    def down(self) -> None:
        """No persistent cluster lives on the platform side; cancel instead."""
        self.cancel_job(self._operation_id)

    def get_logs_stream(
        self,
        cluster_name: str,  # noqa: ARG002
        job_id: str | None = None,  # noqa: ARG002
    ) -> io.TextIOBase:
        """Streaming training logs is not yet supported through this cloud."""
        raise NotImplementedError(
            "Streaming logs from the Oumi Enterprise platform via the "
            "launcher is not yet implemented; use the web UI or "
            "`oumi platform operations status <id>` instead."
        )


class OumiPlatformCloud(BaseCloud):
    """Cloud target that submits jobs to the Oumi Enterprise platform.

    Each call to :meth:`up_cluster` creates a new platform operation and
    returns an :class:`OumiPlatformCluster` wrapping it.
    """

    def __init__(self, client: Client | None = None):
        """Initialize the cloud.

        Args:
            client: Pre-built :class:`Client`. If omitted, one is created
                lazily on first use (via :func:`get_default_client`).
        """
        self._client = client
        self._clusters: dict[str, OumiPlatformCluster] = {}

    def _ensure_client(self) -> Client:
        if self._client is None:
            from oumi.platform.client import get_default_client

            self._client = get_default_client()
        return self._client

    def up_cluster(
        self,
        job: JobConfig,
        name: str | None,
        **kwargs,
    ) -> JobStatus:
        """Submit ``job`` to the platform and wrap the resulting operation."""
        client = self._ensure_client()
        project_id = client._resolve_project_id(None)
        kind = kwargs.pop("kind", None) or _infer_job_kind(job)
        payload = {
            "kind": kind,
            "config": _job_config_to_payload(job),
        }
        if name:
            payload["displayName"] = name
        try:
            op = client.request(
                "POST",
                f"/v1/projects/{project_id}{_JOBS_SUBMIT_PATH}",
                json_body=payload,
            )
        except PlatformAPIError as exc:
            if exc.status_code == 404:
                raise PlatformError(
                    "The Oumi Enterprise platform at "
                    f"{client.credentials.api_url} does not yet expose "
                    "POST /v1/projects/{}/jobs:submit. Update the platform "
                    "to a version that includes the unified job-submission "
                    "endpoint, or use the platform web UI to start the job."
                ) from exc
            raise

        operation_id = _extract_operation_id(op)
        cluster = OumiPlatformCluster(operation_id, client)
        self._clusters[operation_id] = cluster
        return _operation_to_job_status(op, cluster_name=operation_id)

    def get_cluster(self, name: str) -> BaseCluster | None:
        """Return the cluster wrapping platform operation ``name`` (if known)."""
        return self._clusters.get(name)

    def list_clusters(self) -> list[BaseCluster]:
        """Return every platform operation submitted through this process."""
        return list(self._clusters.values())


def _extract_operation_id(op: Any) -> str:
    if not isinstance(op, dict):
        raise PlatformError(
            f"Platform :submit returned a non-object response: {op!r}"
        )
    op_id = op.get("id") or op.get("operationId") or op.get("name")
    if op_id is None:
        raise PlatformError(
            f"Platform :submit response had no operation id: {op!r}"
        )
    return str(op_id)


def _operation_to_job_status(op: Any, *, cluster_name: str) -> JobStatus:
    """Translate a platform operation payload into a launcher :class:`JobStatus`."""
    if not isinstance(op, dict):
        raise PlatformError(
            f"Expected operation payload to be an object, got {op!r}"
        )
    raw_status = str(op.get("status") or "unknown").lower()
    state = _STATUS_TO_STATE.get(raw_status, JobState.PENDING)
    done = bool(op.get("done")) or state in (
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    )
    op_id = _extract_operation_id(op)
    return JobStatus(
        name=op.get("displayName") or op_id,
        id=op_id,
        status=raw_status,
        cluster=cluster_name,
        metadata=json.dumps(op.get("metadata") or {}),
        done=done,
        state=state,
    )


@register_cloud_builder(_CLOUD_NAME)
def OumiPlatform_cloud_builder() -> OumiPlatformCloud:
    """Build a :class:`OumiPlatformCloud` instance.

    Registered under the ``oumi-platform`` name so a :class:`JobConfig` with
    ``resources.cloud="oumi-platform"`` dispatches here.
    """
    return OumiPlatformCloud()
