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

"""Job launch logic for local subprocess and cloud execution."""

import asyncio
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import oumi.launcher as launcher
from oumi.core.launcher.base_cluster import JobStatus as OumiJobStatus
from oumi.mcp.config_service import parse_yaml
from oumi.mcp.job_registry import JobRecord, get_registry
from oumi.mcp.job_runtime import JobRuntime, evict_runtime, migrate_runtime
from oumi.mcp.models import JobCancelResponse

logger = logging.getLogger(__name__)

_COMMAND_MAP: dict[str, str] = {
    "train": "oumi train",
    "evaluate": "oumi evaluate",
    "eval": "oumi evaluate",
    "infer": "oumi infer",
    "synth": "oumi synthesize",
    "analyze": "oumi analyze",
    "tune": "oumi tune",
    "quantize": "oumi quantize",
}


def _is_job_config(config_path: Path) -> bool:
    """Return True if *config_path* is a launcher job config (not a training config)."""
    try:
        data = parse_yaml(str(config_path))
        if not isinstance(data, dict):
            return False
        job_config_keys = {"resources", "setup", "run"}
        return bool(job_config_keys.intersection(data.keys()))
    except Exception:
        return False


def _build_local_command(config_path: str, command: str) -> list[str]:
    """Build an argv list for a local Oumi CLI invocation (no shell)."""
    oumi_cmd = _COMMAND_MAP.get(command, f"oumi {command}")
    parts = oumi_cmd.split()  # e.g. ["oumi", "train"]
    return [*parts, "-c", config_path]


def _stage_cloud_config(
    record: JobRecord, rt: JobRuntime, *, working_dir: str | None = None
) -> str:
    """Copy config into a per-job run directory."""
    assert rt.run_dir is not None
    rt.run_dir.mkdir(parents=True, exist_ok=True)

    if working_dir:
        src = Path(working_dir).expanduser()
        if src.is_dir() and src != rt.run_dir:
            shutil.copytree(src, rt.run_dir, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, rt.run_dir / src.name)

    staged_config = rt.run_dir / "config.yaml"
    shutil.copy2(record.config_path, staged_config)
    rt.staged_config_path = str(staged_config)
    return staged_config.name


def start_local_job(record: JobRecord, rt: JobRuntime, client_cwd: str = "") -> None:
    """Start a local job by spawning the Oumi CLI as a subprocess."""
    cmd_argv = _build_local_command(record.config_path, record.command)

    assert rt.log_dir is not None
    rt.log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    stdout_path = rt.log_dir / f"{ts}_{record.job_id}.stdout"
    stderr_path = rt.log_dir / f"{ts}_{record.job_id}.stderr"

    env = os.environ.copy()
    env["OUMI_LOGGING_DIR"] = str(rt.log_dir)

    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")

    popen_kwargs: dict[str, Any] = {
        "env": env,
        "stdout": stdout_f,
        "stderr": stderr_f,
    }
    if client_cwd:
        popen_kwargs["cwd"] = client_cwd

    try:
        proc = subprocess.Popen(cmd_argv, **popen_kwargs)
    except Exception:
        stdout_f.close()
        stderr_f.close()
        raise

    rt.process = proc
    rt.stdout_f = stdout_f
    rt.stderr_f = stderr_f
    logger.info(
        "Local job %s started (pid=%s): %s",
        record.job_id,
        proc.pid,
        " ".join(cmd_argv),
    )


async def wait_local_completion(record: JobRecord, rt: JobRuntime) -> None:
    """Await completion of a local job subprocess."""
    proc = rt.process
    if proc is None:
        return

    stderr_path = None
    if rt.stderr_f is not None:
        try:
            stderr_path = rt.stderr_f.name
        except Exception:
            pass

    try:
        returncode = await asyncio.to_thread(proc.wait)

        if returncode != 0:
            rt.error_message = f"Process exited with code {returncode}." + (
                f" See stderr: {stderr_path}" if stderr_path else ""
            )
            logger.warning(
                "Local job %s exited with code %d", record.job_id, returncode
            )
        else:
            logger.info("Local job %s completed successfully", record.job_id)
    except Exception as exc:
        rt.error_message = str(exc)
        logger.exception("Failed to run local job %s", record.job_id)
    finally:
        rt.close_log_files()


async def _launch_cloud(
    record: JobRecord,
    rt: JobRuntime,
    *,
    client_cwd: str = "",
) -> str:
    """Launch a cloud job via ``oumi.launcher.up()``.

    Blocks until the launcher returns the sky job ID. The record is
    registered with the sky ID as the canonical identifier — no temporary
    MCP IDs are used.

    Returns the sky job ID on success. On failure, sets ``rt.error_message``
    and returns an empty string.
    """
    reg = get_registry()

    try:
        config_path = Path(record.config_path)

        if not _is_job_config(config_path):
            rt.error_message = (
                "Cloud runs require a job config (with resources/setup/run keys). "
                "Read the guidance://cloud-launch resource to build one from "
                "your training config."
            )
            await evict_runtime(record.job_id)
            return ""

        config_parent = str(Path(record.config_path).expanduser().resolve().parent)
        _stage_cloud_config(record, rt, working_dir=config_parent)
        job_config = launcher.JobConfig.from_yaml(rt.staged_config_path)
        if not job_config.name:
            job_config.name = record.job_id
        if client_cwd and job_config.working_dir:
            wd = Path(job_config.working_dir).expanduser()
            if not wd.is_absolute():
                job_config.working_dir = str((Path(client_cwd) / wd).resolve())
        elif client_cwd and not job_config.working_dir:
            job_config.working_dir = client_cwd

        cluster, status = await asyncio.to_thread(
            launcher.up,
            job_config,
            record.cluster_name or None,
        )
        rt.cluster_obj = cluster
        rt.oumi_status = status

        sky_job_id = str(status.id) if status and status.id else record.job_id
        original_id = record.job_id
        record.job_id = sky_job_id
        record.cluster_name = (
            status.cluster if status and status.cluster else record.cluster_name
        )
        if sky_job_id != original_id:
            reg.remove(original_id)
        reg.add(record)

        logger.info(
            "Cloud job %s launched on %s",
            sky_job_id,
            record.cloud,
        )

        if sky_job_id != original_id:
            await migrate_runtime(original_id, sky_job_id)

        if rt.cancel_requested and status and status.id:
            try:
                result_status = await asyncio.to_thread(
                    launcher.cancel,
                    sky_job_id,
                    record.cloud,
                    record.cluster_name,
                )
                rt.oumi_status = result_status
            except Exception as cancel_exc:
                rt.error_message = (
                    "Cancellation was requested during launch, but automatic "
                    f"cloud cancellation failed: {cancel_exc}"
                )
            await evict_runtime(sky_job_id)
            return sky_job_id

        await evict_runtime(sky_job_id)
        return sky_job_id
    except Exception as exc:
        rt.error_message = str(exc)
        logger.exception("Failed to launch cloud job %s", record.job_id)
        await evict_runtime(record.job_id)
        return ""


async def launch_job(
    record: JobRecord,
    rt: JobRuntime,
    *,
    client_cwd: str = "",
) -> str:
    """Launch a job (local or cloud).

    Returns the canonical job ID. For local jobs this is the MCP-generated ID.
    For cloud jobs this is the sky job ID returned by the launcher.
    """
    if record.cloud == "local":
        start_local_job(record, rt, client_cwd=client_cwd)
        await wait_local_completion(record, rt)
        if rt.log_dir:
            reg = get_registry()
            reg.update(record.job_id, log_dir=str(rt.log_dir))
        await evict_runtime(record.job_id)
        return record.job_id
    else:
        return await _launch_cloud(
            record,
            rt,
            client_cwd=client_cwd,
        )


async def cancel(
    record: JobRecord, rt: JobRuntime, *, force: bool = False
) -> JobCancelResponse:
    """Cancel a job (SIGTERM/SIGKILL for local, launcher.cancel for cloud)."""
    launch_pending = (
        rt.runner_task is not None
        and not rt.runner_task.done()
        and rt.cluster_obj is None
    )
    if record.cloud != "local" and launch_pending:
        rt.cancel_requested = True
        rt.error_message = "Cancellation requested while launch is pending."
        return {
            "success": True,
            "message": (
                f"Cancellation requested for {record.job_id}. "
                "If the cloud launch completes, the MCP will attempt "
                "best-effort cancellation."
            ),
        }

    if record.cloud == "local" and rt.process is not None:
        try:
            if force:
                try:
                    rt.process.kill()
                except ProcessLookupError:
                    pass
                action = "killed (SIGKILL)"
            else:
                try:
                    rt.process.terminate()
                except ProcessLookupError:
                    pass
                action = "terminated (SIGTERM)"
            await asyncio.to_thread(rt.process.wait)
            rt.cancel_requested = True
            rt.error_message = f"Cancelled by user ({action})"
            logger.info("Local job %s %s", record.job_id, action)
            return {
                "success": True,
                "message": f"Job {record.job_id} {action}.",
            }
        except OSError as exc:
            return {
                "success": False,
                "error": f"Failed to cancel local job {record.job_id}: {exc}",
            }
    try:
        result_status = await asyncio.to_thread(
            launcher.cancel,
            record.job_id,
            record.cloud,
            record.cluster_name,
        )
        rt.cancel_requested = True
        rt.oumi_status = result_status
        return {
            "success": True,
            "message": (
                f"Job {record.job_id} cancel requested on "
                f"{record.cloud}/{record.cluster_name}."
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to cancel job {record.job_id}: {exc}",
        }


async def poll_status(record: JobRecord, rt: JobRuntime) -> OumiJobStatus | None:
    """Fetch the latest status for a job.

    Returns None for local jobs (status derived from ``rt.process``).
    """
    if record.cloud == "local":
        return None

    if rt.error_message and rt.cluster_obj is None:
        return rt.oumi_status

    if rt.cluster_obj and record.job_id:
        try:
            status = await asyncio.to_thread(rt.cluster_obj.get_job, record.job_id)
            if status:
                rt.oumi_status = status
                reg = get_registry()
                reg.update(
                    record.job_id,
                    cluster_name=status.cluster or record.cluster_name,
                )
                return status
        except Exception:
            logger.warning(
                "cluster.get_job failed for %s; falling back to launcher.status",
                record.job_id,
                exc_info=True,
            )
    try:
        all_statuses = await asyncio.to_thread(
            launcher.status,
            cloud=record.cloud,
            cluster=record.cluster_name or None,
            id=record.job_id,
        )
        for _, jobs in all_statuses.items():
            for s in jobs:
                if s.id == record.job_id:
                    rt.oumi_status = s
                    reg = get_registry()
                    reg.update(
                        record.job_id,
                        cluster_name=s.cluster or record.cluster_name,
                    )
                    return s
    except Exception:
        logger.warning(
            "launcher.status failed for %s; returning stale status",
            record.job_id,
            exc_info=True,
        )

    return rt.oumi_status
