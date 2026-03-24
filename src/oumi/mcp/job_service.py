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

"""Job management service for Oumi MCP execution tools.

Provides job submission, status polling, cancellation, and log streaming
for both local and cloud execution.
"""

import asyncio
import logging
from typing import Any

import oumi.launcher as launcher
from oumi.mcp.job_launcher import (
    cancel,
    launch_job,
    poll_status,
)
from oumi.mcp.job_logs import (
    get_cloud_logs,
    get_log_paths,
    read_log_tail,
    stream_cloud_logs,
    tail_log_file,
)
from oumi.mcp.job_registry import (
    JobRecord,
    JobRegistry,
    extract_job_metadata,
    get_registry,
    make_job_id,
    reset_registry,
)
from oumi.mcp.job_runtime import (
    JobRuntime,
    evict_runtime,
    get_runtime,
)
from oumi.mcp.models import (
    ClusterLifecycleResponse,
    JobCancelResponse,
    JobLogsResponse,
    JobStatusResponse,
    JobSummary,
)

logger = logging.getLogger(__name__)

__all__ = [
    "JobRecord",
    "JobRegistry",
    "JobRuntime",
    "cancel",
    "cancel_job_impl",
    "down_cluster_impl",
    "evict_runtime",
    "extract_job_metadata",
    "fetch_logs",
    "fetch_status",
    "get_cloud_logs",
    "get_log_paths",
    "get_registry",
    "get_runtime",
    "launch_job",
    "make_job_id",
    "poll_status",
    "read_log_tail",
    "reset_registry",
    "stop_cluster_impl",
    "stream_cloud_logs",
    "tail_log_file",
]


def _job_status_str(record: JobRecord, rt: JobRuntime) -> str:
    """Derive a human-readable status string for any job (local or cloud)."""
    if rt.cancel_requested:
        return "cancelled"
    is_local = record.cloud == "local"
    if is_local:
        proc = rt.process
        if proc is None:
            if rt.error_message:
                return "failed"
            return "completed"
        rc = proc.poll()
        if rc is None:
            return "running"
        return "completed" if rc == 0 else "failed"
    if rt.oumi_status:
        return rt.oumi_status.status
    if rt.error_message:
        return "failed"
    return "unknown"


def _is_job_done(record: JobRecord, rt: JobRuntime) -> bool:
    """Return True if the job is in a terminal state."""
    is_local = record.cloud == "local"
    if is_local:
        if rt.process is not None:
            return rt.process.poll() is not None
        return True
    if rt.oumi_status and rt.oumi_status.done:
        return True
    if rt.error_message:
        return True
    if rt.cancel_requested:
        return True
    return False


def _build_status_response(
    record: JobRecord,
    rt: JobRuntime,
    *,
    log_file: str = "",
) -> JobStatusResponse:
    """Build a ``JobStatusResponse`` from a ``JobRecord`` and ``JobRuntime``."""
    status = rt.oumi_status
    is_local = record.cloud == "local"

    status_str = _job_status_str(record, rt)
    if is_local:
        state_str = status_str.upper()
        cluster_str = "local"
    else:
        state_str = status.state.name if status and status.state else ""
        cluster_str = status.cluster if status else record.cluster_name

    base: JobStatusResponse = {
        "success": True,
        "job_id": record.job_id,
        "status": status_str,
        "state": state_str,
        "command": record.command,
        "config_path": record.config_path,
        "cloud": record.cloud,
        "cluster": cluster_str,
        "model_name": record.model_name,
        "is_done": _is_job_done(record, rt),
        "error": rt.error_message,
    }

    if status and status.metadata:
        base["metadata"] = (
            status.metadata
            if isinstance(status.metadata, dict)
            else {"raw": str(status.metadata)}
        )
    if log_file:
        base["log_file"] = log_file

    return base


def _not_found_response(job_id: str) -> JobStatusResponse:
    """Return a ``JobStatusResponse`` for a missing job ID."""
    return {
        "success": False,
        "job_id": job_id,
        "status": "not_found",
        "state": "",
        "command": "",
        "config_path": "",
        "cloud": "",
        "cluster": "",
        "model_name": "",
        "is_done": False,
        "error": (
            f"Job '{job_id}' not found. "
            "Use list_jobs() for MCP-managed jobs, or provide "
            "job_id + cloud (+ cluster_name) for direct cloud lookup."
        ),
    }


def _resolve_job_record(
    *,
    job_id: str = "",
    cloud: str = "",
) -> JobRecord | None:
    """Resolve a job record by unified job_id."""
    reg = get_registry()
    if job_id:
        record = reg.get(job_id)
        if record:
            return record
        if cloud:
            return reg.find_by_cloud(cloud, job_id)
    return None


async def _fetch_cloud_status_direct(
    *,
    job_id: str,
    cloud: str,
    cluster_name: str = "",
) -> Any | None:
    try:
        statuses_by_cloud = await asyncio.to_thread(
            launcher.status,
            cloud=cloud,
            cluster=cluster_name or None,
            id=job_id,
        )
    except Exception:
        return None
    for _cloud_name, statuses in statuses_by_cloud.items():
        for status in statuses:
            if status.id == job_id:
                return status
    return None


async def _list_job_summaries(status_filter: str = "all") -> list[JobSummary]:
    """Build job summaries from launcher (cloud) and registry (local)."""
    reg = get_registry()
    summaries: list[JobSummary] = []

    try:
        all_statuses = await asyncio.to_thread(launcher.status)
        for cloud_name, jobs in all_statuses.items():
            for job_status in jobs:
                mapping = reg.find_by_cloud(cloud_name, job_status.id)
                mcp_id = mapping.job_id if mapping else ""
                model = mapping.model_name if mapping else ""
                cmd = mapping.command if mapping else ""

                is_done = bool(job_status.done)
                if status_filter == "running" and is_done:
                    continue
                if status_filter == "completed" and not is_done:
                    continue

                summaries.append(
                    {
                        "job_id": mcp_id or job_status.id,
                        "command": cmd,
                        "status": job_status.status,
                        "cloud": cloud_name,
                        "cluster": job_status.cluster,
                        "model_name": model,
                        "is_done": is_done,
                    }
                )
    except Exception:
        logger.warning(
            "launcher.status failed; falling back to registry only", exc_info=True
        )

    for record in reg.all():
        if record.cloud != "local":
            continue
        rt = await get_runtime(record.job_id)
        status_str = _job_status_str(record, rt)
        is_done = _is_job_done(record, rt)
        if status_filter == "running" and is_done:
            continue
        if status_filter == "completed" and not is_done:
            continue
        summaries.append(
            {
                "job_id": record.job_id,
                "command": record.command,
                "status": status_str,
                "cloud": "local",
                "cluster": "local",
                "model_name": record.model_name,
                "is_done": is_done,
            }
        )

    return summaries


async def fetch_status(
    *,
    job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobStatusResponse:
    """Fetch status for a single job (by MCP ID or cloud identity)."""
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    job_id = job_id.strip()

    if not job_id:
        return _not_found_response("")

    record = _resolve_job_record(
        job_id=job_id,
        cloud=cloud,
    )
    if not record:
        if not cloud:
            return _not_found_response(job_id)
        direct_status = await _fetch_cloud_status_direct(
            job_id=job_id,
            cloud=cloud,
            cluster_name=cluster_name,
        )
        if not direct_status:
            return _not_found_response(job_id)
        return {
            "success": True,
            "job_id": job_id,
            "status": direct_status.status,
            "state": direct_status.state.name if direct_status.state else "",
            "command": "",
            "config_path": "",
            "cloud": cloud,
            "cluster": direct_status.cluster or cluster_name,
            "model_name": "",
            "is_done": bool(direct_status.done),
            "metadata": direct_status.metadata if direct_status.metadata else {},
            "error": None,
        }

    rt = await get_runtime(record.job_id)
    await poll_status(record, rt)
    log_paths = get_log_paths(record, rt)
    return _build_status_response(
        record,
        rt,
        log_file=str(log_paths["stdout"]) if log_paths["stdout"] else "",
    )


async def fetch_logs(
    *,
    job_id: str = "",
    lines: int = 200,
    cloud: str = "",
    cluster_name: str = "",
) -> JobLogsResponse:
    """Fetch a bounded log snapshot for a job."""
    if lines < 0:
        lines = 0
    lines = min(lines, 10000)

    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    job_id = job_id.strip()

    record = _resolve_job_record(
        job_id=job_id,
        cloud=cloud,
    )
    if not record:
        if job_id and cloud and cluster_name:
            ephemeral = JobRecord(
                job_id=job_id,
                command="",
                config_path="",
                cloud=cloud,
                cluster_name=cluster_name,
                model_name="",
                submit_time="",
            )
            cloud_result = await get_cloud_logs(ephemeral, JobRuntime(), lines)
            if cloud_result is not None:
                cloud_logs, cloud_lines = cloud_result
                return {
                    "success": True,
                    "job_id": job_id,
                    "lines_requested": lines,
                    "lines_returned": cloud_lines,
                    "log_file": f"cloud:{cloud}/{cluster_name}",
                    "logs": cloud_logs,
                    "error": None,
                }
            return {
                "success": False,
                "job_id": job_id,
                "lines_requested": lines,
                "lines_returned": 0,
                "log_file": "",
                "logs": "",
                "error": (
                    f"Cloud log retrieval failed for {cloud}/{cluster_name}. "
                    f"The cluster may no longer exist or SSH timed out."
                ),
            }
        if job_id and cloud:
            return {
                "success": False,
                "job_id": job_id,
                "lines_requested": lines,
                "lines_returned": 0,
                "log_file": "",
                "logs": "",
                "error": (
                    "cluster_name is required for direct cloud log retrieval. "
                    "Provide job_id + cloud + cluster_name."
                ),
            }
        return {
            "success": False,
            "job_id": job_id,
            "lines_requested": lines,
            "lines_returned": 0,
            "log_file": "",
            "logs": "",
            "error": f"Job '{job_id}' not found.",
        }

    rt = await get_runtime(record.job_id)
    await poll_status(record, rt)
    log_paths = get_log_paths(record, rt)
    stdout_path = log_paths.get("stdout")
    resolved_job_id = record.job_id

    if not stdout_path or not stdout_path.exists():
        if record.cloud and record.cloud != "local":
            cloud_result = await get_cloud_logs(record, rt, lines)
            if cloud_result is not None:
                cloud_logs, cloud_lines = cloud_result
                return {
                    "success": True,
                    "job_id": resolved_job_id,
                    "lines_requested": lines,
                    "lines_returned": cloud_lines,
                    "log_file": f"cloud:{record.cloud}/{record.cluster_name}",
                    "logs": cloud_logs,
                    "error": None,
                }
            return {
                "success": False,
                "job_id": resolved_job_id,
                "lines_requested": lines,
                "lines_returned": 0,
                "log_file": "",
                "logs": "",
                "error": (
                    "No local log file and cloud log retrieval failed. "
                    f"The cluster '{record.cluster_name}' may no longer exist. "
                    f"Try `sky logs {record.cluster_name}` directly."
                ),
            }
        return {
            "success": False,
            "job_id": resolved_job_id,
            "lines_requested": lines,
            "lines_returned": 0,
            "log_file": "",
            "logs": "",
            "error": "No stdout log file available yet for this job.",
        }

    try:
        logs, lines_returned = await asyncio.to_thread(
            read_log_tail, stdout_path, lines
        )
    except OSError as exc:
        return {
            "success": False,
            "job_id": resolved_job_id,
            "lines_requested": lines,
            "lines_returned": 0,
            "log_file": str(stdout_path),
            "logs": "",
            "error": f"Failed to read log file: {exc}",
        }

    return {
        "success": True,
        "job_id": resolved_job_id,
        "lines_requested": lines,
        "lines_returned": lines_returned,
        "log_file": str(stdout_path),
        "logs": logs,
        "error": None,
    }


async def cancel_job_impl(
    *,
    job_id: str = "",
    force: bool = False,
    cloud: str = "",
    cluster_name: str = "",
) -> JobCancelResponse:
    """Cancel a running or pending job."""
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    job_id = job_id.strip()

    record = _resolve_job_record(
        job_id=job_id,
        cloud=cloud,
    )

    if not record:
        if job_id and cloud:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        launcher.cancel,
                        job_id,
                        cloud,
                        cluster_name,
                    ),
                    timeout=30.0,
                )
            except TimeoutError:
                return {
                    "success": False,
                    "error": (
                        f"Cancel timed out after 30s "
                        f"(cloud={cloud}, cluster={cluster_name}, id={job_id}). "
                        "The cancellation may still be in progress. "
                        "Check cloud console or retry."
                    ),
                }
            except Exception as exc:
                return {
                    "success": False,
                    "error": (
                        "Failed to cancel cloud job by direct identity "
                        f"(cloud={cloud}, cluster={cluster_name}, id={job_id}): {exc}"
                    ),
                }
            return {
                "success": True,
                "message": (
                    "Cancel requested by direct cloud identity "
                    f"(cloud={cloud}, cluster={cluster_name}, id={job_id})."
                ),
            }
        return {
            "success": False,
            "error": f"Job '{job_id}' not found.",
        }

    rt = await get_runtime(record.job_id)

    if record.cloud != "local":
        live = await poll_status(record, rt)
        if live and live.done:
            return {
                "success": False,
                "error": (
                    f"Job {record.job_id} is already finished (status: {live.status})"
                ),
            }

    return await cancel(record, rt, force=force)


async def stop_cluster_impl(cloud: str, cluster_name: str) -> ClusterLifecycleResponse:
    """Stop a running cluster, preserving infra."""
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    if not cloud or not cluster_name:
        return {
            "success": False,
            "error": "cloud and cluster_name are required.",
        }
    try:
        await asyncio.to_thread(launcher.stop, cloud, cluster_name)
        return {
            "success": True,
            "message": (
                f"Cluster '{cluster_name}' on {cloud} stopped. "
                "Infra is preserved; restart by submitting a new job with "
                f"cluster_name='{cluster_name}'. Storage costs may still apply. "
                f"Use down_cluster to fully delete."
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to stop cluster '{cluster_name}' on {cloud}: {exc}",
        }


async def down_cluster_impl(
    cloud: str,
    cluster_name: str,
    confirm: bool = False,
    user_confirmation: str = "",
) -> ClusterLifecycleResponse:
    """Delete a cluster and all its resources (irreversible)."""
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    if not cloud or not cluster_name:
        return {
            "success": False,
            "error": "cloud and cluster_name are required.",
        }
    if not confirm:
        return {
            "success": True,
            "message": (
                f"Dry run: would permanently delete cluster "
                f"'{cluster_name}' on {cloud}. "
                "IRREVERSIBLE — all cluster resources and data will be deleted and "
                "billing will stop. To confirm, re-call with "
                "confirm=True, user_confirmation='DOWN'."
            ),
        }
    if user_confirmation != "DOWN":
        return {
            "success": False,
            "error": "Confirmation phrase must be exactly 'DOWN'. Deletion blocked.",
        }
    try:
        await asyncio.to_thread(launcher.down, cloud, cluster_name)
        return {
            "success": True,
            "message": (
                f"Cluster '{cluster_name}' on {cloud} deleted. "
                "All resources have been removed and billing has stopped."
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to delete cluster '{cluster_name}' on {cloud}: {exc}",
        }
