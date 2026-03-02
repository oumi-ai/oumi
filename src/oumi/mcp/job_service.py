"""Job management service for Oumi MCP execution tools.

Provides job submission, status polling, cancellation, and log streaming
for both local and cloud execution.

Design:
    - **Local jobs** (``cloud == "local"``): spawned directly via
      ``subprocess.Popen`` running the Oumi CLI (e.g. ``oumi train -c …``).
      This avoids the ``oumi.launcher.LocalCluster`` requirement for a
      ``working_dir``, which the MCP server cannot reliably provide.
    - **Cloud jobs**: delegated to ``oumi.launcher.up()`` which handles
      SkyPilot cluster lifecycle, multi-cloud routing, etc.
    - A thin ``JobRegistry`` maps MCP job IDs to ``JobRecord`` objects.
    - ``tail_log_file()`` provides async log tailing for streaming to
      the MCP client via ``ctx.info()``.
"""

import asyncio
import dataclasses
import io
import json
import logging
import os
import shlex
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

import oumi.launcher as launcher
from oumi.core.launcher.base_cluster import BaseCluster
from oumi.core.launcher.base_cluster import JobStatus as OumiJobStatus
from oumi.mcp.config_service import parse_yaml
from oumi.mcp.constants import (
    DEFAULT_JOBS_FILE,
    LOG_TAIL_INTERVAL_SECONDS,
)
from oumi.mcp.models import (
    ClusterLifecycleResponse,
    JobCancelResponse,
    JobLogsResponse,
    JobStatusResponse,
    JobSummary,
)

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    """Persisted job metadata — identity mapping only.

    All fields are strings for simple JSON serde.
    The registry does NOT store job status; status is always
    queried live from ``oumi.launcher.status()`` (cloud) or
    ``rt.process.poll()`` (local).
    """

    job_id: str
    command: str
    config_path: str
    cloud: str
    cluster_name: str
    oumi_job_id: str
    model_name: str
    submit_time: str  # ISO 8601
    output_dir: str = ""


@dataclass
class JobRuntime:
    """Ephemeral per-job state -- lives only in memory, never persisted."""

    process: subprocess.Popen | None = None  # type: ignore[type-arg]
    cluster_obj: BaseCluster | None = None
    runner_task: asyncio.Task[None] | None = None
    oumi_status: OumiJobStatus | None = None
    stdout_f: Any = None
    stderr_f: Any = None
    log_dir: Path | None = None
    run_dir: Path | None = None
    staged_config_path: str = ""
    cancel_requested: bool = False
    error_message: str | None = None

    def close_log_files(self) -> None:
        for f in (self.stdout_f, self.stderr_f):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self.stdout_f = None
        self.stderr_f = None


_MAX_REGISTRY_AGE_DAYS = 7
_MAX_REGISTRY_SIZE = 200
_CLOUD_LOG_TIMEOUT = 30.0


class JobRegistry:
    """Single-file JSON registry mapping MCP job IDs to cloud identities.

    Evicts entries older than ``_MAX_REGISTRY_AGE_DAYS`` on load.
    Caps total records at ``_MAX_REGISTRY_SIZE``, dropping oldest first.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._jobs: dict[str, JobRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for entry in data:
                # Handle legacy records that still have 'status' field
                entry.pop("status", None)
                r = JobRecord(**entry)
                self._jobs[r.job_id] = r
        except Exception:
            logger.warning("Could not load %s, starting fresh", self._path)
        pruned = self._prune()
        if pruned:
            logger.info("Pruned %d stale job records from registry", pruned)
            self._save()

    def _prune(self) -> int:
        """Remove entries older than the age cutoff, then cap total size."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=_MAX_REGISTRY_AGE_DAYS)
        to_remove: list[str] = []
        for jid, rec in self._jobs.items():
            try:
                ts = datetime.fromisoformat(rec.submit_time)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    to_remove.append(jid)
            except (ValueError, TypeError):
                to_remove.append(jid)
        for jid in to_remove:
            del self._jobs[jid]
        removed = len(to_remove)

        # Cap total size — drop oldest first
        if len(self._jobs) > _MAX_REGISTRY_SIZE:
            by_time = sorted(self._jobs.items(), key=lambda x: x[1].submit_time)
            while len(self._jobs) > _MAX_REGISTRY_SIZE and by_time:
                jid, _ = by_time.pop(0)
                del self._jobs[jid]
                removed += 1

        return removed

    def _save(self) -> None:
        records = [dataclasses.asdict(r) for r in self._jobs.values()]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
        tmp.rename(self._path)

    def add(self, record: JobRecord) -> None:
        self._jobs[record.job_id] = record
        self._save()

    def update(self, job_id: str, **fields: Any) -> None:
        record = self._jobs.get(job_id)
        if record is None:
            logger.warning("Registry.update: job_id %s not found, skipping", job_id)
            return
        for k, v in fields.items():
            setattr(record, k, v)
        self._save()

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def find_by_cloud_identity(self, cloud: str, oumi_job_id: str) -> JobRecord | None:
        for r in self._jobs.values():
            if r.cloud == cloud and r.oumi_job_id == oumi_job_id:
                return r
        return None

    def all(self) -> list[JobRecord]:
        return list(self._jobs.values())

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._save()


_runtimes: dict[str, JobRuntime] = {}


def get_runtime(job_id: str) -> JobRuntime:
    if job_id not in _runtimes:
        _runtimes[job_id] = JobRuntime()
    return _runtimes[job_id]


def evict_runtime(job_id: str) -> None:
    """Remove a runtime entry, closing any open handles."""
    rt = _runtimes.pop(job_id, None)
    if rt is None:
        return
    rt.close_log_files()
    if rt.runner_task and not rt.runner_task.done():
        rt.runner_task.cancel()


def cleanup_stale_runtimes() -> None:
    """Remove runtime entries whose job_id is no longer in the registry."""
    reg = get_registry()
    stale = [jid for jid in _runtimes if reg.get(jid) is None]
    for jid in stale:
        evict_runtime(jid)
    if stale:
        logger.info("Evicted %d stale runtime entries", len(stale))


_registry: JobRegistry | None = None


def get_registry(path: Path | None = None) -> JobRegistry:
    """Return the global ``JobRegistry``, creating it on first access."""
    global _registry
    if _registry is None:
        _registry = JobRegistry(path or DEFAULT_JOBS_FILE)
    return _registry


def make_job_id(command: str, job_name: str | None = None) -> str:
    """Generate a human-friendly job ID.

    Format: ``{command}_{YYYYMMDD_HHMMSS}_{6-hex}`` or the caller-supplied
    *job_name* if provided (sanitized to prevent path traversal).
    """
    if job_name:
        sanitized = job_name.replace("..", "_").replace("/", "_").replace("\\", "_")
        sanitized = sanitized.strip("._- ")
        if not sanitized:
            raise ValueError(f"Invalid job_name after sanitization: {job_name!r}")
        return sanitized
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{command}_{ts}_{short}"


def extract_job_metadata(config_path: str) -> dict[str, Any]:
    """Extract model_name and output_dir from an Oumi YAML config.

    Returns a dict with ``model_name`` and ``output_dir`` keys.
    Missing values default to ``"unknown"`` / ``"./output"``.
    """
    config = parse_yaml(config_path)
    model_name = config.get("model", {}).get("model_name", "unknown") or "unknown"

    output_dir = (
        config.get("training", {}).get("output_dir")
        or config.get("output_dir")
        or "./output"
    )
    return {"model_name": model_name, "output_dir": output_dir}


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

_DEFAULT_CLOUD_SETUP_SCRIPT = """set -e
pip install uv
uv pip install --system oumi[gpu]
command -v oumi || { echo "ERROR: oumi not found after install"; exit 1; }
"""

_DEFAULT_CREDENTIAL_FILES: list[str] = [
    "~/.cache/huggingface/token",
    "~/.netrc",
]


def _is_job_config(config_path: Path) -> bool:
    """Return True if *config_path* is a launcher job config (not a training config).

    A job config has top-level ``resources``, ``setup``, or ``run`` keys rather
    than training-specific keys like ``model`` or ``training``.
    """
    try:
        data = parse_yaml(str(config_path))
        if not isinstance(data, dict):
            return False
        job_config_keys = {"resources", "setup", "run"}
        return bool(job_config_keys.intersection(data.keys()))
    except Exception:
        return False


def _parse_gpu_count(accelerators: str | None) -> int:
    """Parse the number of GPUs from an accelerator spec string.

    Handles formats like ``"A100:8"`` (→ 8), ``"A100"`` (→ 1),
    and ``None`` (→ 0).
    """
    if not accelerators:
        return 0
    parts = accelerators.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return 1


def _default_file_mounts() -> dict[str, str]:
    """Return file mounts for common credential files that exist locally.

    Includes ``~/.cache/huggingface/token`` (HuggingFace auth) and
    ``~/.netrc`` (WandB / general HTTP credentials) when they exist.
    """
    mounts: dict[str, str] = {}
    for cred_path in _DEFAULT_CREDENTIAL_FILES:
        local_path = Path(cred_path).expanduser()
        if local_path.exists():
            mounts[cred_path] = cred_path
    return mounts


def _build_local_command(config_path: str, command: str) -> list[str]:
    """Build an argv list for a local Oumi CLI invocation (no shell)."""
    oumi_cmd = _COMMAND_MAP.get(command, f"oumi {command}")
    parts = oumi_cmd.split()  # e.g. ["oumi", "train"]
    return [*parts, "-c", config_path]


def _build_shell_command(
    config_path: str,
    command: str,
    *,
    num_nodes: int = 1,
    accelerators: str | None = None,
) -> str:
    """Build the shell command string for an Oumi CLI invocation (cloud runs).

    SkyPilot copies ``working_dir`` to the remote and executes this script
    from within it, so *config_path* is a relative filename (e.g. ``config.yaml``).

    Extends PATH to cover common uv/pip install locations before executing so
    the oumi binary is reachable even when the run step's shell differs from
    the setup step's shell.  Verifies oumi is found before running.
    """
    num_gpus = _parse_gpu_count(accelerators)
    if num_gpus > 1 or num_nodes > 1:
        oumi_cmd = f"oumi distributed torchrun -m oumi {command}"
    else:
        oumi_cmd = _COMMAND_MAP.get(command, f"oumi {command}")

    path_preamble = (
        'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"\n'
        'command -v oumi || { echo "ERROR: oumi not found on PATH after setup"; exit 1; }\n'
    )
    return f"set -e\n{path_preamble}{oumi_cmd} -c {shlex.quote(config_path)}"


def _stage_cloud_config(
    record: JobRecord, rt: JobRuntime, *, working_dir: str | None = None
) -> str:
    """Copy config (and optionally a working directory) into a per-job run directory.

    For training-config wrapping mode, only the config file is copied.
    For job-config passthrough mode, pass *working_dir* to copy the entire
    source directory tree so relative references inside the config are preserved.

    Returns the staged config filename (relative to the run directory).
    """
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


def _build_cloud_job_config(
    config_path: str,
    command: str,
    *,
    cloud: str,
    working_dir: str,
    accelerators: str | None = None,
    job_name: str | None = None,
    envs: dict[str, str] | None = None,
    file_mounts: dict[str, str] | None = None,
    disk_size: int | None = None,
    use_spot: bool = False,
    num_nodes: int = 1,
    setup_script: str = "",
    run_script: str = "",
) -> launcher.JobConfig:
    """Build an ``oumi.launcher.JobConfig`` for **cloud** execution.

    For cloud jobs the launcher handles cluster lifecycle via SkyPilot.
    ``working_dir`` is a per-job staging directory copied by the launcher.
    The run script references the staged config path relative to that directory.

    Automatically selects ``oumi distributed torchrun`` when multiple GPUs or
    nodes are requested.  Default file mounts include common credential files
    (``~/.cache/huggingface/token``, ``~/.netrc``) when they exist locally;
    *file_mounts* entries take precedence and can override them.

    When *setup_script* or *run_script* are provided, they override the
    defaults (``_DEFAULT_CLOUD_SETUP_SCRIPT`` and the auto-generated run
    command respectively).
    """
    effective_run = run_script or _build_shell_command(
        config_path, command, num_nodes=num_nodes, accelerators=accelerators
    )
    effective_setup = setup_script or _DEFAULT_CLOUD_SETUP_SCRIPT

    resources = launcher.JobResources(cloud=cloud)
    if accelerators:
        resources.accelerators = accelerators
    if disk_size is not None:
        resources.disk_size = disk_size
    resources.use_spot = use_spot
    effective_mounts = _default_file_mounts()
    if file_mounts:
        effective_mounts.update(file_mounts)

    return launcher.JobConfig(
        name=job_name,
        num_nodes=num_nodes,
        resources=resources,
        working_dir=working_dir,
        setup=effective_setup,
        run=effective_run,
        envs=envs or {},
        file_mounts=effective_mounts,
    )


def _generate_job_config_template(
    config_path: str,
    command: str,
    cloud: str,
    model_name: str,
    *,
    client_cwd: str = "",
    job_name: str = "",
    accelerators: str = "",
    num_nodes: int = 1,
    envs: dict[str, str] | None = None,
    setup_script: str = "",
    run_script: str = "",
) -> str:
    """Generate a complete, annotated job config YAML template.

    Produces a ready-to-customize YAML string with TODO markers on sections
    that need user attention. Pre-fills the training command, model info,
    resources, and auto-detected credential mounts.
    """
    from datetime import datetime, timezone

    date_tag = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    safe_model = model_name.split("/")[-1].lower().replace("-", "").replace("_", "")
    name = job_name or f"{command}-{safe_model}-{date_tag}"

    if run_script:
        run_block = run_script
    else:
        num_gpus = _parse_gpu_count(accelerators or None)
        if num_gpus > 1 or num_nodes > 1:
            run_block = (
                f"set -e\noumi distributed torchrun -m oumi {command} -c ./config.yaml"
            )
        else:
            oumi_cmd = _COMMAND_MAP.get(command, f"oumi {command}")
            run_block = f"set -e\n{oumi_cmd} -c ./config.yaml"

    if setup_script:
        setup_block = setup_script
    else:
        setup_block = (
            "set -e\n"
            "pip install uv && uv pip install --system 'oumi[gpu]'\n"
            "# TODO: Add dataset downloads, extra packages, preprocessing\n"
            "# Example: huggingface-cli download <dataset-id> --repo-type dataset --local-dir ./data\n"
            "# Example: uv pip install --system 'oumi[gpu,evaluation]'"
        )

    file_mount_lines = []
    for cred_path in _DEFAULT_CREDENTIAL_FILES:
        local_path = Path(cred_path).expanduser()
        if local_path.exists():
            file_mount_lines.append(f"  {cred_path}: {cred_path}")

    env_lines = []
    if envs:
        for k, v in envs.items():
            env_lines.append(f"  {k}: {json.dumps(v)}")

    lines = [
        "# Oumi Cloud Job Config — customize TODO sections before launching",
        f"# Model: {model_name}",
        "# Docs: https://oumi.ai/docs/en/latest/user_guides/launch/launch.html",
        f"name: {name}",
        "",
        "resources:",
        f"  cloud: {cloud}",
    ]
    if accelerators:
        lines.append(f'  accelerators: "{accelerators}"')
    else:
        lines.append('  accelerators: "A100:8"  # TODO: Set GPU type and count')
    lines += [
        "  use_spot: false",
        "  disk_size: 500",
        "",
        f"num_nodes: {num_nodes}",
        "working_dir: .  # Resolved to client_cwd at launch time. Synced to ~/sky_workdir on the VM.",
        "",
    ]

    if file_mount_lines:
        lines.append("# Auto-detected credential files (add local dataset mounts below if needed)")
        lines.append("file_mounts:")
        lines.extend(file_mount_lines)
        lines.append("  # TODO: Mount local dataset files if not git-tracked in working_dir")
        lines.append("  # ~/sky_workdir/data/train.jsonl: /absolute/path/to/local/train.jsonl")
    else:
        lines.append("# TODO: Mount credential and data files needed on the remote VM")
        lines.append("# file_mounts:")
        lines.append("#   ~/.cache/huggingface/token: ~/.cache/huggingface/token")
        lines.append("#   ~/sky_workdir/data/train.jsonl: /absolute/path/to/local/train.jsonl")
    lines.append("")

    lines += [
        "# TODO: Mount cloud storage for persistent output (recommended for spot instances)",
        "# storage_mounts:",
        "#   /output:",
        "#     source: gs://your-bucket/output",
        "#     store: gcs",
        "",
    ]

    lines.append("envs:")
    if env_lines:
        lines.extend(env_lines)
    else:
        lines.append(f'  OUMI_RUN_NAME: "{name}"')
        lines.append("  # TODO: Add env vars (WANDB_API_KEY, HF_TOKEN, etc.)")
    lines.append("  # Recommended for CUDA memory stability:")
    lines.append('  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"')
    lines.append("")

    lines.append("# Setup script — runs once when the VM is provisioned")
    lines.append("setup: |")
    for setup_line in setup_block.split("\n"):
        lines.append(f"  {setup_line}")
    lines.append("")

    lines.append("# Run command (auto-generated from your training config)")
    lines.append("run: |")
    for run_line in run_block.split("\n"):
        lines.append(f"  {run_line}")
    lines.append("")

    return "\n".join(lines)


def start_local_job(
    record: JobRecord, rt: JobRuntime, client_cwd: str = ""
) -> None:
    """Start a local job by spawning the Oumi CLI directly.

    Creates the log directory, starts the subprocess via ``Popen``, and
    sets ``rt.process`` and ``record.oumi_job_id``. Raises on failure
    (e.g. command not found, permission denied).

    When *client_cwd* is provided, the subprocess runs from that directory
    so relative paths in the config resolve against the user's project root.

    Stdout and stderr are written to files in ``rt.log_dir`` so
    that ``tail_log_file()`` can stream them to the MCP client.
    """
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
    record.oumi_job_id = str(proc.pid)
    get_registry().update(
        record.job_id, oumi_job_id=record.oumi_job_id
    )
    rt.stdout_f = stdout_f
    rt.stderr_f = stderr_f
    logger.info(
        "Local job %s started (pid=%s): %s",
        record.job_id,
        proc.pid,
        " ".join(cmd_argv),
    )


async def wait_local_completion(record: JobRecord, rt: JobRuntime) -> None:
    """Await completion of a local job subprocess.

    Waits for the process to exit (in a thread) and sets ``rt.error_message``
    on failure.  Status is derived at runtime from ``rt.process.poll()``.
    """
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
    accelerators: str | None = None,
    envs: dict[str, str] | None = None,
    file_mounts: dict[str, str] | None = None,
    disk_size: int | None = None,
    use_spot: bool = False,
    num_nodes: int = 1,
    setup_script: str | None = None,
    run_script: str | None = None,
) -> None:
    """Launch a cloud job via ``oumi.launcher.up()``.

    Supports two modes:

    * **Job-config passthrough** (when the config file itself is a launcher job
      config with ``resources``/``setup``/``run`` keys): loads the config directly
      via ``launcher.JobConfig.from_yaml()`` so all cloud-specific fields
      (``envs``, ``file_mounts``, ``storage_mounts``, ``disk_size``, etc.) are
      preserved as written.

    * **Training-config wrapping**: wraps a training YAML in a minimal
      ``launcher.JobConfig``, applying any caller-supplied *envs*, *file_mounts*,
      *disk_size*, *use_spot*, *num_nodes*, and *setup_script* overrides on top of
      sensible defaults (oumi with GPU extras, auto-mounted credential files).

    Updates *record* and *rt* in-place with the cluster object, oumi job ID, and
    initial status.  On failure, sets ``rt.error_message``.
    """
    reg = get_registry()

    try:
        config_path = Path(record.config_path)
        job_config_mode = _is_job_config(config_path)

        if job_config_mode:
            config_parent = str(Path(record.config_path).expanduser().resolve().parent)
            _stage_cloud_config(record, rt, working_dir=config_parent)
            job_config = launcher.JobConfig.from_yaml(rt.staged_config_path)
            if not job_config.name:
                job_config.name = record.job_id
            # Resolve relative working_dir against client_cwd so SkyPilot
            # syncs the user's project, not the MCP staging directory.
            if client_cwd and job_config.working_dir:
                wd = Path(job_config.working_dir).expanduser()
                if not wd.is_absolute():
                    job_config.working_dir = str(
                        (Path(client_cwd) / wd).resolve()
                    )
            elif client_cwd and not job_config.working_dir:
                job_config.working_dir = client_cwd
            if envs:
                merged = dict(job_config.envs or {})
                merged.update(envs)
                job_config.envs = merged
            if file_mounts:
                merged_mounts = dict(job_config.file_mounts or {})
                merged_mounts.update(file_mounts)
                job_config.file_mounts = merged_mounts
        else:
            staged_config_name = _stage_cloud_config(record, rt)
            job_config = _build_cloud_job_config(
                staged_config_name,
                record.command,
                cloud=record.cloud,
                working_dir=client_cwd or str(rt.run_dir),
                accelerators=accelerators,
                job_name=record.job_id,
                envs=envs,
                file_mounts=file_mounts,
                disk_size=disk_size,
                use_spot=use_spot,
                num_nodes=num_nodes,
                setup_script=setup_script or "",
                run_script=run_script or "",
            )
        cluster, status = await asyncio.to_thread(
            launcher.up,
            job_config,
            record.cluster_name or None,
        )
        rt.cluster_obj = cluster
        oumi_job_id = status.id if status else ""
        rt.oumi_status = status
        cluster_name = status.cluster if status else record.cluster_name

        # Update registry with cloud identity (no status field)
        reg.update(
            record.job_id,
            oumi_job_id=oumi_job_id,
            cluster_name=cluster_name,
        )
        record = reg.get(record.job_id) or record
        logger.info(
            "Cloud job %s launched on %s (oumi_id=%s)",
            record.job_id,
            record.cloud,
            record.oumi_job_id,
        )

        # Race guard: if cancel was requested while launcher.up was in-flight,
        # immediately cancel the cloud job now that we have an oumi_job_id.
        if rt.cancel_requested and record.oumi_job_id:
            try:
                result_status = await asyncio.to_thread(
                    launcher.cancel,
                    record.oumi_job_id,
                    record.cloud,
                    record.cluster_name,
                )
                rt.oumi_status = result_status
            except Exception as cancel_exc:
                rt.error_message = (
                    "Cancellation was requested during launch, but automatic "
                    f"cloud cancellation failed: {cancel_exc}"
                )
            # Evict cloud runtime after reconciliation — launcher is source of truth now
            evict_runtime(record.job_id)
            return

        # Cloud launch succeeded — evict runtime (launcher.status is source of truth)
        evict_runtime(record.job_id)
    except Exception as exc:
        rt.error_message = str(exc)
        logger.exception("Failed to launch cloud job %s", record.job_id)


async def launch_job(
    record: JobRecord,
    rt: JobRuntime,
    *,
    client_cwd: str = "",
    accelerators: str | None = None,
    envs: dict[str, str] | None = None,
    file_mounts: dict[str, str] | None = None,
    disk_size: int | None = None,
    use_spot: bool = False,
    num_nodes: int = 1,
    setup_script: str | None = None,
    run_script: str | None = None,
) -> None:
    """Launch a job -- local or cloud -- in a background thread.

    For local jobs, spawns the Oumi CLI directly via subprocess.
    For cloud jobs, delegates to ``oumi.launcher.up()``.
    """
    if record.cloud == "local":
        start_local_job(record, rt, client_cwd=client_cwd)
        await wait_local_completion(record, rt)
    else:
        await _launch_cloud(
            record,
            rt,
            client_cwd=client_cwd,
            accelerators=accelerators,
            envs=envs,
            file_mounts=file_mounts,
            disk_size=disk_size,
            use_spot=use_spot,
            num_nodes=num_nodes,
            setup_script=setup_script,
            run_script=run_script,
        )


async def poll_status(record: JobRecord, rt: JobRuntime) -> OumiJobStatus | None:
    """Fetch the latest status for a job from the launcher.

    For **local** jobs, returns None (status derived from ``rt.process``).
    For **cloud** jobs, queries the cluster or launcher and updates ``rt.oumi_status``.
    """
    if record.cloud == "local":
        return None

    if rt.error_message and rt.cluster_obj is None:
        return rt.oumi_status

    # Try cluster.get_job first (fastest path)
    if rt.cluster_obj and record.oumi_job_id:
        try:
            status = await asyncio.to_thread(
                rt.cluster_obj.get_job, record.oumi_job_id
            )
            if status:
                rt.oumi_status = status
                # Update registry with cloud identity if it changed
                reg = get_registry()
                reg.update(
                    record.job_id,
                    oumi_job_id=status.id or record.oumi_job_id,
                    cluster_name=status.cluster or record.cluster_name,
                )
                return status
        except Exception:
            logger.warning(
                "cluster.get_job failed for %s; falling back to launcher.status",
                record.job_id,
                exc_info=True,
            )

    # Fallback: launcher.status (works even without a cluster object)
    try:
        if not record.oumi_job_id:
            return rt.oumi_status
        all_statuses = await asyncio.to_thread(
            launcher.status,
            cloud=record.cloud,
            cluster=record.cluster_name or None,
            id=record.oumi_job_id,
        )
        for _, jobs in all_statuses.items():
            for s in jobs:
                if s.id == record.oumi_job_id:
                    rt.oumi_status = s
                    reg = get_registry()
                    reg.update(
                        record.job_id,
                        oumi_job_id=s.id or record.oumi_job_id,
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


async def cancel(
    record: JobRecord, rt: JobRuntime, *, force: bool = False
) -> JobCancelResponse:
    """Cancel a job.

    For **local** jobs, sends SIGTERM (or SIGKILL if *force* is True).
    For **cloud** jobs, delegates to ``oumi.launcher.cancel()``.
    """
    # Pre-launch cancel: job hasn't reached the cloud yet
    if not record.oumi_job_id and rt.process is None:
        rt.cancel_requested = True
        rt.error_message = "Cancellation requested while launch is pending."
        if rt.runner_task and not rt.runner_task.done():
            rt.runner_task.cancel()
        return {
            "success": True,
            "message": (
                f"Cancellation requested for {record.job_id}. "
                "If the cloud launch completes, the MCP will attempt "
                "best-effort cancellation."
            ),
        }

    # Local job cancel
    if record.cloud == "local" and rt.process is not None:
        try:
            if force:
                rt.process.kill()
                action = "killed (SIGKILL)"
            else:
                rt.process.terminate()
                action = "terminated (SIGTERM)"
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

    # Cloud job cancel — delegate to launcher
    try:
        result_status = await asyncio.to_thread(
            launcher.cancel,
            record.oumi_job_id,
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


def get_log_paths(record: JobRecord, rt: JobRuntime) -> dict[str, Path | None]:
    """Return paths to the stdout and stderr log files for a job.

    Returns a dict with ``"stdout"`` and ``"stderr"`` keys, each
    mapping to a ``Path`` or ``None`` if the file doesn't exist yet.
    """
    result: dict[str, Path | None] = {"stdout": None, "stderr": None}
    log_dir = rt.log_dir
    if log_dir is None or not log_dir.is_dir():
        return result

    id_candidates = [record.job_id]
    if record.oumi_job_id and record.oumi_job_id != record.job_id:
        id_candidates.append(record.oumi_job_id)

    for suffix in ("stdout", "stderr"):
        for candidate_id in id_candidates:
            matches = sorted(log_dir.glob(f"*_{candidate_id}.{suffix}"))
            if matches:
                result[suffix] = matches[-1]
                break
        else:
            matches = sorted(log_dir.glob(f"*.{suffix}"))
            if matches:
                result[suffix] = matches[-1]

    return result


async def tail_log_file(
    path: Path,
    done_event: asyncio.Event,
    poll_interval: float = LOG_TAIL_INTERVAL_SECONDS,
) -> AsyncIterator[str]:
    """Async generator that yields new lines from *path* as they appear.

    Behaves like ``tail -f``: opens the file, seeks to the current end,
    then yields new complete lines as they are written.  Stops when
    *done_event* is set **and** no more data is available.

    If the file does not exist yet, waits up to ``poll_interval`` between
    checks until it appears or *done_event* fires.
    """
    while not path.exists():
        if done_event.is_set():
            return
        await asyncio.sleep(poll_interval)

    position = 0
    partial = ""

    while True:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0

        if size > position:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    f.seek(position)
                    chunk = f.read()
                    position = f.tell()
            except OSError:
                chunk = ""

            if chunk:
                partial += chunk
                while "\n" in partial:
                    line, partial = partial.split("\n", 1)
                    yield line

        if done_event.is_set():
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    f.seek(position)
                    remaining = f.read()
            except OSError:
                remaining = ""
            if remaining:
                partial += remaining
            if partial:
                yield partial
            return

        await asyncio.sleep(poll_interval)


async def _get_cloud_logs(
    record: JobRecord,
    rt: JobRuntime,
    lines: int = 200,
) -> tuple[str, int] | None:
    """Fetch the last *lines* of logs from a cloud job.

    Uses ``cluster.get_logs_stream()`` which calls ``sky.tail_logs(follow=True)``.
    Since ``follow=True`` means the stream never ends for running jobs, we read
    with a timeout and return whatever was accumulated (partial logs are better
    than nothing).
    """
    cluster = rt.cluster_obj

    if cluster is None and record.cloud and record.cluster_name:
        try:
            cloud_obj = await asyncio.to_thread(launcher.get_cloud, record.cloud)
            cluster = await asyncio.to_thread(
                cloud_obj.get_cluster, record.cluster_name
            )
        except Exception:
            logger.debug(
                "Failed to reconstruct cluster for %s/%s",
                record.cloud,
                record.cluster_name,
                exc_info=True,
            )
            return None

    if cluster is None:
        logger.debug(
            "Cluster %s/%s not found (may have been deleted)",
            record.cloud,
            record.cluster_name,
        )
        return None

    try:
        stream: io.TextIOBase = await asyncio.to_thread(
            cluster.get_logs_stream,
            record.cluster_name,
            record.oumi_job_id or None,
        )
    except NotImplementedError:
        logger.debug(
            "Cloud %s does not support get_logs_stream for job %s",
            record.cloud,
            record.job_id,
        )
        return None
    except Exception:
        logger.debug(
            "get_logs_stream failed for job %s",
            record.job_id,
            exc_info=True,
        )
        return None

    chunks: list[str] = []

    def _read_stream() -> str:
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                chunks.append(line)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass
        return "".join(chunks)

    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_read_stream),
            timeout=_CLOUD_LOG_TIMEOUT,
        )
    except asyncio.TimeoutError:
        # Don't cross-thread close the stream — just return partial output.
        # The worker thread will finish when the stream yields EOF or errors.
        raw = "".join(chunks)
        if raw:
            logger.debug(
                "Cloud log read timed out for job %s after %.0fs, "
                "returning %d partial lines",
                record.job_id,
                _CLOUD_LOG_TIMEOUT,
                raw.count("\n"),
            )

    if not raw:
        return None

    all_lines = raw.splitlines()
    tail = all_lines[-lines:] if lines > 0 else all_lines
    return ("\n".join(tail), len(tail))


async def stream_cloud_logs(
    record: JobRecord,
    rt: JobRuntime,
    done_event: asyncio.Event,
) -> AsyncIterator[str]:
    """Yield log lines from ``cluster.get_logs_stream()`` for cloud jobs.

    Falls back silently (returns without yielding) if the cluster does not
    support log streaming (raises ``NotImplementedError``).
    """
    cluster = rt.cluster_obj
    if cluster is None:
        return

    try:
        stream: io.TextIOBase = await asyncio.to_thread(
            cluster.get_logs_stream,
            record.cluster_name,
            record.oumi_job_id or None,
        )
    except NotImplementedError:
        logger.debug(
            "Cloud %s does not support get_logs_stream for job %s",
            record.cloud,
            record.job_id,
        )
        return
    except Exception:
        logger.debug(
            "get_logs_stream failed for job %s",
            record.job_id,
            exc_info=True,
        )
        return

    def _read_lines() -> list[str]:
        """Read available lines from the stream (blocking)."""
        lines: list[str] = []
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        except Exception:
            pass
        return lines

    try:
        while not done_event.is_set():
            lines = await asyncio.to_thread(_read_lines)
            for line in lines:
                yield line
            if not lines:
                await asyncio.sleep(LOG_TAIL_INTERVAL_SECONDS)
    finally:
        try:
            stream.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Job status / logs / cancel helpers (moved from server.py)
# ---------------------------------------------------------------------------


def _jobconfig_to_yaml(jc: launcher.JobConfig) -> str:
    """Render a JobConfig as compact YAML for dry-run display.

    Omits None values and empty dicts/lists so the preview stays readable.
    """
    d = {k: v for k, v in dataclasses.asdict(jc).items() if v not in (None, {}, [], "")}
    if "resources" in d and isinstance(d["resources"], dict):
        d["resources"] = {
            k: v for k, v in d["resources"].items() if v not in (None, False, "")
        }
    return yaml.dump(d, default_flow_style=False, sort_keys=False)


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
            # Still launching (runner_task exists but process not spawned yet)
            if rt.runner_task and not rt.runner_task.done():
                return "launching"
            return "unknown"
        rc = proc.poll()
        if rc is None:
            return "running"
        return "completed" if rc == 0 else "failed"
    # Cloud job — use launcher status
    if rt.oumi_status:
        return rt.oumi_status.status
    if rt.error_message:
        return "failed"
    # No oumi_job_id yet — still launching
    if not record.oumi_job_id:
        return "launching"
    return "unknown"


def _is_job_done(record: JobRecord, rt: JobRuntime) -> bool:
    """Return True if the job is in a terminal state."""
    is_local = record.cloud == "local"
    if is_local and rt.process is not None:
        return rt.process.poll() is not None
    if rt.oumi_status and rt.oumi_status.done:
        return True
    if rt.error_message and not rt.runner_task:
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
        oumi_job_id = record.oumi_job_id
        state_str = status_str.upper()
        cluster_str = "local"
    else:
        oumi_job_id = status.id if status else record.oumi_job_id
        state_str = status.state.name if status and status.state else ""
        cluster_str = status.cluster if status else record.cluster_name

    base: JobStatusResponse = {
        "success": True,
        "job_id": record.job_id,
        "oumi_job_id": oumi_job_id,
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
        "oumi_job_id": "",
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
            "oumi_job_id + cloud (+ cluster_name) for direct cloud lookup."
        ),
    }


def _resolve_job_record(
    *,
    job_id: str = "",
    oumi_job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobRecord | None:
    reg = get_registry()
    if job_id:
        record = reg.get(job_id)
        if record:
            return record
    if oumi_job_id and cloud:
        return reg.find_by_cloud_identity(cloud, oumi_job_id)
    return None


async def _fetch_cloud_status_direct(
    *,
    oumi_job_id: str,
    cloud: str,
    cluster_name: str = "",
) -> Any | None:
    try:
        statuses_by_cloud = await asyncio.to_thread(
            launcher.status,
            cloud=cloud,
            cluster=cluster_name or None,
            id=oumi_job_id,
        )
    except Exception:
        return None
    for _cloud_name, statuses in statuses_by_cloud.items():
        for status in statuses:
            if status.id == oumi_job_id:
                return status
    return None


def _read_log_tail(stdout_path: Path, lines: int) -> tuple[str, int]:
    """Read the trailing *lines* from *stdout_path* efficiently."""
    if lines <= 0:
        return ("", 0)
    block_size = 8192
    data = b""
    newline_count = 0

    with stdout_path.open("rb") as f:
        pos = f.seek(0, 2)
        while pos > 0 and newline_count <= lines:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            data = chunk + data
            newline_count = data.count(b"\n")

    text = data.decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    if not all_lines:
        return ("", 0)
    tail_lines = all_lines[-lines:]
    return ("\n".join(tail_lines), len(tail_lines))


async def _list_job_summaries(status_filter: str = "all") -> list[JobSummary]:
    """Build job summaries from launcher (cloud) and registry (local)."""
    reg = get_registry()
    summaries: list[JobSummary] = []

    # Cloud jobs: query launcher for live state
    try:
        all_statuses = await asyncio.to_thread(launcher.status)
        for cloud_name, jobs in all_statuses.items():
            for job_status in jobs:
                # Try to find MCP job ID from registry
                mapping = reg.find_by_cloud_identity(cloud_name, job_status.id)
                mcp_id = mapping.job_id if mapping else ""
                model = mapping.model_name if mapping else ""
                cmd = mapping.command if mapping else ""

                is_done = bool(job_status.done)
                if status_filter == "running" and is_done:
                    continue
                if status_filter == "completed" and not is_done:
                    continue

                summaries.append({
                    "job_id": mcp_id or job_status.id,
                    "command": cmd,
                    "status": job_status.status,
                    "cloud": cloud_name,
                    "cluster": job_status.cluster,
                    "model_name": model,
                    "is_done": is_done,
                })
    except Exception:
        logger.warning("launcher.status failed; falling back to registry only", exc_info=True)

    # Local jobs: check from registry + runtime
    for record in reg.all():
        if record.cloud != "local":
            continue
        rt = get_runtime(record.job_id)
        status_str = _job_status_str(record, rt)
        is_done = _is_job_done(record, rt)
        if status_filter == "running" and is_done:
            continue
        if status_filter == "completed" and not is_done:
            continue
        summaries.append({
            "job_id": record.job_id,
            "command": record.command,
            "status": status_str,
            "cloud": "local",
            "cluster": "local",
            "model_name": record.model_name,
            "is_done": is_done,
        })

    return summaries


# ---------------------------------------------------------------------------
# Service-level functions for MCP tool bodies
# ---------------------------------------------------------------------------


async def fetch_status(
    *,
    job_id: str = "",
    oumi_job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobStatusResponse:
    """Fetch status for a single job (by MCP ID or cloud identity)."""
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    job_id = job_id.strip()
    oumi_job_id = oumi_job_id.strip()

    if not job_id and not oumi_job_id:
        return _not_found_response("")

    record = _resolve_job_record(
        job_id=job_id,
        oumi_job_id=oumi_job_id,
        cloud=cloud,
        cluster_name=cluster_name,
    )
    if not record:
        if not oumi_job_id or not cloud:
            return _not_found_response(job_id or oumi_job_id)
        direct_status = await _fetch_cloud_status_direct(
            oumi_job_id=oumi_job_id,
            cloud=cloud,
            cluster_name=cluster_name,
        )
        if not direct_status:
            return _not_found_response(job_id or oumi_job_id)
        return {
            "success": True,
            "job_id": job_id or oumi_job_id,
            "oumi_job_id": direct_status.id,
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

    rt = get_runtime(record.job_id)
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
    oumi_job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobLogsResponse:
    """Fetch a bounded log snapshot for a job."""
    if lines < 0:
        lines = 0

    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    job_id = job_id.strip()
    oumi_job_id = oumi_job_id.strip()

    record = _resolve_job_record(
        job_id=job_id,
        oumi_job_id=oumi_job_id,
        cloud=cloud,
        cluster_name=cluster_name,
    )
    if not record:
        # Direct cloud log retrieval for untracked jobs (bypasses registry)
        if oumi_job_id and cloud and cluster_name:
            ephemeral = JobRecord(
                job_id=oumi_job_id,
                command="",
                config_path="",
                cloud=cloud,
                cluster_name=cluster_name,
                oumi_job_id=oumi_job_id,
                model_name="",
                submit_time="",
            )
            cloud_result = await _get_cloud_logs(ephemeral, JobRuntime(), lines)
            if cloud_result is not None:
                cloud_logs, cloud_lines = cloud_result
                return {
                    "success": True,
                    "job_id": oumi_job_id,
                    "lines_requested": lines,
                    "lines_returned": cloud_lines,
                    "log_file": f"cloud:{cloud}/{cluster_name}",
                    "logs": cloud_logs,
                    "error": None,
                }
            return {
                "success": False,
                "job_id": oumi_job_id,
                "lines_requested": lines,
                "lines_returned": 0,
                "log_file": "",
                "logs": "",
                "error": (
                    f"Cloud log retrieval failed for {cloud}/{cluster_name}. "
                    f"The cluster may no longer exist or SSH timed out."
                ),
            }
        if oumi_job_id and cloud:
            return {
                "success": False,
                "job_id": job_id or oumi_job_id,
                "lines_requested": lines,
                "lines_returned": 0,
                "log_file": "",
                "logs": "",
                "error": (
                    "cluster_name is required for direct cloud log retrieval. "
                    "Provide oumi_job_id + cloud + cluster_name."
                ),
            }
        return {
            "success": False,
            "job_id": job_id or oumi_job_id,
            "lines_requested": lines,
            "lines_returned": 0,
            "log_file": "",
            "logs": "",
            "error": f"Job '{job_id or oumi_job_id}' not found.",
        }

    rt = get_runtime(record.job_id)
    await poll_status(record, rt)
    log_paths = get_log_paths(record, rt)
    stdout_path = log_paths.get("stdout")
    resolved_job_id = record.job_id

    if not stdout_path or not stdout_path.exists():
        # Cloud fallback: fetch logs from cluster via get_logs_stream
        if record.cloud and record.cloud != "local":
            cloud_result = await _get_cloud_logs(record, rt, lines)
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
            _read_log_tail, stdout_path, lines
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
    oumi_job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobCancelResponse:
    """Cancel a running or pending job."""
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    job_id = job_id.strip()
    oumi_job_id = oumi_job_id.strip()

    record = _resolve_job_record(
        job_id=job_id,
        oumi_job_id=oumi_job_id,
        cloud=cloud,
        cluster_name=cluster_name,
    )

    if not record:
        if oumi_job_id and cloud:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        launcher.cancel,
                        oumi_job_id,
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
                        f"(cloud={cloud}, cluster={cluster_name}, id={oumi_job_id}). "
                        "The cancellation may still be in progress. "
                        "Check cloud console or retry."
                    ),
                }
            except Exception as exc:
                return {
                    "success": False,
                    "error": (
                        "Failed to cancel cloud job by direct identity "
                        f"(cloud={cloud}, cluster={cluster_name}, id={oumi_job_id}): {exc}"
                    ),
                }
            return {
                "success": True,
                "message": (
                    "Cancel requested by direct cloud identity "
                    f"(cloud={cloud}, cluster={cluster_name}, id={oumi_job_id})."
                ),
            }
        return {
            "success": False,
            "error": f"Job '{job_id or oumi_job_id}' not found.",
        }

    rt = get_runtime(record.job_id)

    # For cloud jobs, check live status first — the job may already be done
    if record.cloud != "local" and record.oumi_job_id:
        live = await poll_status(record, rt)
        if live and live.done:
            return {
                "success": False,
                "error": (
                    f"Job {record.job_id} is already finished "
                    f"(status: {live.status})"
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
                f"Dry run: would permanently delete cluster '{cluster_name}' on {cloud}. "
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


async def list_jobs_impl(status: str = "all") -> list[JobSummary]:
    """List running and completed jobs."""
    return await _list_job_summaries(status_filter=status)
