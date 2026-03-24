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

"""Oumi MCP Server - ML Training Config Discovery and Execution.

~500 ready-to-use YAML configs for fine-tuning LLMs (Llama, Qwen, Phi, etc.).
Local execution via subprocess, cloud execution via oumi.launcher.

IMPORTANT — ALWAYS call get_started() FIRST before using any other tool.
get_started() returns the full tool catalog, resource list, and recommended
workflow. Without it you will miss critical path-resolution rules and the
correct order of operations.

Path rules:
- All path-sensitive tools require client_cwd (the user's project root).
- Config file path: absolute OR relative to client_cwd.
- Local jobs: subprocess runs from client_cwd; paths inside YAML resolve there.
- Cloud jobs: client_cwd becomes working_dir on remote VM;
  use repo-relative paths inside YAML. NEVER use local machine paths.
"""

import asyncio
import json
import logging
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from fastmcp import FastMCP  # type: ignore[import-not-found]

import oumi.launcher as launcher
from oumi.mcp.config_service import (
    TASK_MAPPING,
    find_config_match,
    get_all_configs,
    get_categories,
    get_configs_dir,
    load_yaml_strict,
    resolve_config_path,
)
from oumi.mcp.config_service import (
    search_configs as search_configs_service,
)
from oumi.mcp.constants import (
    DEFAULT_STREAM_LINES,
    JOB_LOGS_DIR,
    JOB_RUNS_DIR,
    VALID_OUMI_COMMANDS,
    ValidatorTaskType,
)
from oumi.mcp.docs_service import (
    get_module_list,
    search_docs,
    start_background_indexing,
)
from oumi.mcp.environment_service import (
    _strip_oumi_env_overrides,
)
from oumi.mcp.job_launcher import (
    _is_job_config,
    launch_job,
    poll_status,
    start_local_job,
    wait_local_completion,
)
from oumi.mcp.job_logs import (
    get_log_paths,
)
from oumi.mcp.job_registry import (
    JobRecord,
    get_registry,
    make_job_id,
)
from oumi.mcp.job_runtime import (
    get_runtime,
)
from oumi.mcp.job_service import (
    _list_job_summaries,
    cancel_job_impl,
    down_cluster_impl,
    fetch_logs,
    fetch_status,
    stop_cluster_impl,
)
from oumi.mcp.models import (
    CategoriesResponse,
    ClusterLifecycleResponse,
    ConfigDetail,
    ConfigMetadata,
    DocsSearchResponse,
    JobCancelResponse,
    JobLogsResponse,
    JobStatusResponse,
    JobSubmissionResponse,
    JobSummary,
    ListModulesResponse,
    PreFlightCheckResponse,
    PreFlightSummary,
    ValidateConfigResponse,
)
from oumi.mcp.preflight_service import (
    _pre_flight_check,
)
from oumi.mcp.prompts import (
    ANALYZE_COMMAND_RESOURCE,
    CLOUD_LAUNCH_RESOURCE,
    EVAL_COMMAND_RESOURCE,
    GET_STARTED_CONTENT,
    INFER_COMMAND_RESOURCE,
    MLE_WORKFLOW_RESOURCE,
    POST_TRAINING_RESOURCE,
    SYNTH_COMMAND_RESOURCE,
    TRAIN_COMMAND_RESOURCE,
)
from oumi.mcp.sync_service import (
    config_sync,
    get_configs_source,
    get_oumi_version,
    is_oumi_dev_build,
)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Oumi Config Server",
    instructions=(
        "IMPORTANT: Always call get_started() FIRST before using any other tool. "
        "It returns the full tool catalog, path rules, and recommended workflow."
    ),
)


def _configure_logging() -> None:
    """Redirect all oumi logs to stderr so they don't corrupt MCP stdio protocol."""
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    for name in ("oumi", __name__):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.addHandler(stderr_handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False

    for noisy_logger in (
        "mcp.server.lowlevel.server",
        "mcp.server.lowlevel",
        "mcp.shared.session",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _extract_job_metadata_from_cfg(cfg: dict[str, Any]) -> tuple[str, str]:
    """Extract model name and output dir from parsed config."""
    model_name = (
        (cfg.get("model") or {}).get("model_name", "unknown")
        if isinstance(cfg.get("model"), dict)
        else "unknown"
    )
    if not model_name:
        model_name = "unknown"
    raw_training = cfg.get("training")
    training = raw_training if isinstance(raw_training, dict) else {}
    output_dir = training.get("output_dir") or cfg.get("output_dir") or "./output"
    return str(model_name), str(output_dir)


@mcp.resource("guidance://mle-workflow")
async def get_mle_workflow_guidance() -> str:
    """ML engineering workflow guidance for Oumi.

    This resource provides a full ML workflow and tool usage guidance for
    training LLMs with Oumi. Cursor may choose to fetch this resource and
    include it as context when working with Oumi MCP tools.
    """
    return MLE_WORKFLOW_RESOURCE


@mcp.resource("guidance://mle-train")
async def get_train_command_guidance() -> str:
    """MLE guidance for oumi train."""
    return TRAIN_COMMAND_RESOURCE


@mcp.resource("guidance://mle-synth")
async def get_synth_command_guidance() -> str:
    """MLE guidance for oumi synth."""
    return SYNTH_COMMAND_RESOURCE


@mcp.resource("guidance://mle-analyze")
async def get_analyze_command_guidance() -> str:
    """MLE guidance for oumi analyze."""
    return ANALYZE_COMMAND_RESOURCE


@mcp.resource("guidance://mle-eval")
async def get_eval_command_guidance() -> str:
    """MLE guidance for oumi evaluate/eval."""
    return EVAL_COMMAND_RESOURCE


@mcp.resource("guidance://mle-infer")
async def get_infer_command_guidance() -> str:
    """MLE guidance for oumi infer."""
    return INFER_COMMAND_RESOURCE


@mcp.resource("guidance://cloud-launch")
async def get_cloud_launch_guidance() -> str:
    """Cloud job launch guidance — job config anatomy, setup patterns, examples.

    Read this resource when planning a cloud training run. Explains what a job
    config is, the key fields to customize, common setup patterns (dataset
    downloads, extra packages), and how ``run_oumi_job`` works with both
    training configs and job configs.
    """
    return CLOUD_LAUNCH_RESOURCE


@mcp.resource("guidance://post-training")
async def get_post_training_guidance() -> str:
    """Post-training guidance — downloading weights, evaluation, teardown, merging.

    Read this resource after a cloud training job succeeds. Covers the full
    post-training lifecycle: downloading model weights via SkyPilot CLI,
    running evaluation on the live cluster, tearing down to stop billing,
    merging LoRA adapters locally, and pushing to HuggingFace Hub.
    """
    return POST_TRAINING_RESOURCE


@mcp.tool()
def search_configs(
    query: list[str] | None = None,
    content_match: list[str] | None = None,
    limit: int = 20,
) -> list[ConfigMetadata]:
    """Search the Oumi config library (~500 configs for LLM fine-tuning).

    Args:
        query: Terms matched against config paths (AND logic,
            case-insensitive). Paths encode model family, size, task, and
            technique — e.g. ["llama3_1", "8b", "sft", "lora"],
            ["qwen3", "grpo"], ["deepseek_r1", "eval"].
        content_match: Substrings matched against YAML file content
            (AND logic, case-insensitive). Use for values not in the path,
            e.g. a dataset name or HuggingFace model ID.
        limit: Max results (default 20).
    """
    configs = get_all_configs()
    return search_configs_service(configs, query, content_match, limit)


@mcp.tool()
def get_config(path: str) -> ConfigDetail:
    """Get details about a specific Oumi config file.

    Use the returned config as a REFERENCE to understand structure, field names,
    and reasonable defaults — do NOT copy it verbatim. Build the user's config
    from scratch, adapting only the relevant settings (model, dataset, training
    params, PEFT) to match their specific requirements, hardware, and data.

    Args:
        path: Config path from search_configs(), or a partial path
            (e.g. "llama3_1/sft/8b_lora" will match).
    """
    configs = get_all_configs()
    match = find_config_match(path, configs)

    if match is None:
        return {
            "path": "",
            "description": "",
            "model_name": "",
            "task_type": "",
            "datasets": [],
            "reward_functions": [],
            "peft_type": "",
            "content": "",
            "error": f"Config not found: {path}",
        }

    configs_dir = get_configs_dir()
    config_path = configs_dir / match["path"]

    try:
        content = config_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read config content: {e}")
        content = f"Error reading file: {e}"

    return {
        "path": match["path"],
        "description": match["description"],
        "model_name": match["model_name"],
        "task_type": match["task_type"],
        "datasets": match["datasets"] or [],
        "reward_functions": match["reward_functions"] or [],
        "peft_type": match["peft_type"] or "",
        "content": content,
        "error": "",
    }


def _build_version_warning() -> str:
    """Return a warning string if configs may not match the installed oumi."""
    source = get_configs_source()
    oumi_ver = get_oumi_version()

    if oumi_ver == "unknown":
        return ""

    if source.startswith("cache:main") and not is_oumi_dev_build(oumi_ver):
        return (
            f"Configs were synced from the main branch but oumi {oumi_ver} "
            "is a release build. Config fields may not match the installed "
            "library. Run config_sync(force=True) after upgrading oumi."
        )

    if source.startswith("bundled:"):
        bundled_ver = source.split(":", 1)[1]
        if bundled_ver != oumi_ver and not is_oumi_dev_build(oumi_ver):
            return (
                f"Using bundled configs from oumi {bundled_ver} but oumi "
                f"{oumi_ver} is installed. Some configs may reference fields "
                "not present in (or removed from) the installed library."
            )

    return ""


@mcp.tool()
def list_categories() -> CategoriesResponse:
    """List available config categories, model families, and API providers."""
    configs_dir = get_configs_dir()
    configs = get_all_configs()
    return get_categories(
        configs_dir,
        len(configs),
        oumi_version=get_oumi_version(),
        configs_source=get_configs_source(),
        version_warning=_build_version_warning(),
    )


@mcp.tool()
def pre_flight_check(
    config: str, client_cwd: str, cloud: str = ""
) -> PreFlightCheckResponse:
    """Run pre-flight checks to catch issues before launching.

    Validates: HF auth & gated repo access, hardware/packages, local paths,
    and cloud credentials (with actual API calls, not just file checks).

    When ``blocking=True`` in the response, there are hard blockers that
    WILL prevent the run from succeeding — surface these as showstoppers.

    Args:
        config: Absolute path, or relative to client_cwd, to the YAML config file.
        client_cwd: REQUIRED. Absolute path to the client's working directory
            (project root). Resolves relative config paths and sets the execution
            context for local and cloud jobs.
        cloud: Target cloud provider (e.g. "gcp", "aws"). Validates
            credentials and returns ``suggested_configs`` for that cloud.
            Leave empty for local runs.
    """
    return _pre_flight_check(config, client_cwd=client_cwd, cloud=cloud)


@mcp.tool()
def validate_config(
    config: str, task_type: ValidatorTaskType, client_cwd: str
) -> ValidateConfigResponse:
    """Validate an Oumi YAML config against its schema.

    Args:
        config: Absolute path, or relative to client_cwd, to the YAML config file.
        task_type: Config type: training, evaluation, inference, tuning,
            synthesis, quantization, job, judge, analyze, async_evaluation.
        client_cwd: REQUIRED. Absolute path to the client's working directory
            (project root). Resolves relative config paths.
    """
    config_path, path_error = resolve_config_path(config, client_cwd)
    if path_error:
        return {"valid": False, "error": path_error}
    try:
        cfg = TASK_MAPPING[task_type].from_yaml(config_path)
        cfg.finalize_and_validate()
        return {"valid": True, "error": None}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@mcp.tool()
def get_started() -> str:
    """Overview of all tools, resources, and recommended workflow.

    CALL THIS FIRST — before using any other Oumi MCP tool.
    Returns the full tool catalog, resource list, path-resolution rules,
    and the correct order of operations for both local and cloud workflows.
    """
    return GET_STARTED_CONTENT


@mcp.tool()
async def run_oumi_job(
    config_path: str,
    command: str,
    client_cwd: str,
    dry_run: bool = True,
    job_name: str | None = None,
    cloud: str = "local",
    cluster_name: str = "",
    skip_preflight: bool = False,
) -> JobSubmissionResponse:
    """Execute an Oumi CLI command with background job tracking.

    Two-step safety: first call with dry_run=True (default) to preview
    the execution plan, then call again with dry_run=False to launch.
    IMPORTANT: Always confirm with the user before calling with
    dry_run=False. Cloud runs execute a pre-flight check that may block
    launch.

    Cloud runs require a job config YAML (with ``resources``/``setup``/``run``
    keys). Training configs are only supported for local runs.
    Read ``guidance://cloud-launch`` for how to build a job config.

    Args:
        config_path: Absolute path, or relative to client_cwd, to an Oumi YAML config.
        command: Oumi subcommand: train, analyze, synth, evaluate, eval,
            infer, tune, quantize. Ignored for job configs.
        client_cwd: REQUIRED. Absolute path to the client's working directory
            (project root). Resolves relative config paths. For local jobs, the
            Oumi CLI subprocess runs from this directory. For cloud jobs, this
            directory is synced to the remote VM as the working directory.
        dry_run: Preview execution plan without running (default True).
            Set to False only after showing the preview to the user and
            receiving their explicit approval.
        job_name: Optional name; auto-generated if omitted.
        cloud: ``"local"`` (default) or a cloud provider name.
        cluster_name: Cluster name for cloud launches.
        skip_preflight: Skip inline pre-flight checks (default False).
            Set to True when pre-flight was already run separately.
    """
    command = command.strip().lower()
    cloud = cloud.strip().lower() or "local"
    cluster_name = cluster_name.strip()

    def _error_response(error: str, **overrides: Any) -> JobSubmissionResponse:
        base: JobSubmissionResponse = {
            "success": False,
            "job_id": "",
            "status": "error",
            "dry_run": dry_run,
            "command": command,
            "config_path": config_path,
            "cloud": cloud,
            "cluster_name": cluster_name,
            "model_name": "",
            "message": "",
            "error": error,
            "launch_confirmed": None,
            "preflight": None,
        }
        base.update(overrides)  # type: ignore[typeddict-item]
        return base

    if command not in VALID_OUMI_COMMANDS:
        return _error_response(
            f"Invalid command: '{command}'. "
            f"Must be one of: {sorted(VALID_OUMI_COMMANDS)}"
        )

    config_file, path_error = resolve_config_path(config_path, client_cwd)
    if path_error:
        return _error_response(path_error)

    abs_config = str(config_file)
    parsed_cfg, parse_error = load_yaml_strict(config_file)
    if parse_error or parsed_cfg is None:
        return _error_response(
            (
                f"{parse_error} "
                "Run validate_config(..., task_type=...) before launching."
            ),
            config_path=abs_config,
        )
    try:
        model_name, output_dir = _extract_job_metadata_from_cfg(parsed_cfg)
    except Exception as exc:
        return _error_response(
            f"Failed to parse config metadata: {exc}",
            config_path=abs_config,
        )

    job_id = make_job_id(command, job_name)

    is_job_config_file = _is_job_config(config_file) if cloud != "local" else False

    if cloud != "local" and not is_job_config_file:
        return _error_response(
            "Cloud runs require a job config (with resources/setup/run keys). "
            "Your config appears to be a training config. "
            "Read the guidance://cloud-launch resource to build a job config, "
            "then re-submit with the job config.",
            config_path=abs_config,
            model_name=model_name,
        )

    if dry_run:
        cmd_preview = (
            f"oumi launch up -c {abs_config}"
            if is_job_config_file
            else f"oumi {command} -c {abs_config}"
        )
        message = "\n".join(
            [
                f"Dry run: would execute `{cmd_preview}` on {cloud}",
                f"Model: {model_name}",
                f"Output: {output_dir}",
                "Validation: strict YAML parsing passed.",
                "To execute, confirm with the user, then re-call with dry_run=False.",
            ]
        )
        if cloud != "local" and is_job_config_file:
            try:
                job_cfg_yaml = str(launcher.JobConfig.from_yaml(abs_config))
            except Exception:
                job_cfg_yaml = "(could not parse job config for preview)"
            message += (
                f"\n\n--- Generated JobConfig (review before executing) ---\n"
                f"{job_cfg_yaml}"
                f"\n-----------------------------------------------------"
            )
        return {
            "success": True,
            "job_id": job_id,
            "status": "dry_run",
            "dry_run": True,
            "command": command,
            "config_path": abs_config,
            "cloud": cloud,
            "cluster_name": cluster_name,
            "model_name": model_name,
            "message": message,
            "error": None,
            "launch_confirmed": None,
            "preflight": None,
        }

    preflight_data: PreFlightSummary | None = None

    if cloud != "local" and not skip_preflight:
        preflight_result = _pre_flight_check(
            abs_config, client_cwd=client_cwd, cloud=cloud
        )
        preflight_data = {
            "summary": preflight_result.get("summary", ""),
            "blocking": bool(preflight_result.get("blocking")),
            "errors": preflight_result.get("errors", []) or [],
            "warnings": list(preflight_result.get("warnings", []) or []),
        }
        if preflight_data["blocking"]:
            return _error_response(
                f"Pre-flight checks failed: {preflight_data['summary']}",
                status="blocked",
                job_id=job_id,
                config_path=abs_config,
                model_name=model_name,
                preflight=preflight_data,
            )
        if preflight_result.get("skypilot_compat_issue"):
            preflight_data["blocking"] = True
            return _error_response(
                "Pre-flight detected a SkyPilot compatibility issue. "
                "Align Oumi/SkyPilot versions and run `sky check` before launching.",
                status="blocked",
                job_id=job_id,
                config_path=abs_config,
                model_name=model_name,
                preflight=preflight_data,
            )

    submit_time = datetime.now(tz=timezone.utc).isoformat()
    record = JobRecord(
        job_id=job_id,
        command=command,
        config_path=abs_config,
        cloud=cloud,
        cluster_name=cluster_name,
        model_name=model_name,
        submit_time=submit_time,
        output_dir=output_dir,
    )
    reg = get_registry()
    reg.add(record)

    rt = await get_runtime(job_id)
    rt.log_dir = JOB_LOGS_DIR / job_id
    rt.run_dir = JOB_RUNS_DIR / job_id

    is_local = cloud == "local"
    if is_local:
        try:
            start_local_job(record, rt, client_cwd=client_cwd)
        except Exception as exc:
            rt.error_message = str(exc)
            return _error_response(
                f"Failed to start job: {exc}",
                job_id=job_id,
                config_path=abs_config,
                model_name=model_name,
            )
        runner: asyncio.Task[Any] = asyncio.create_task(
            wait_local_completion(record, rt),
            name=f"oumi-job-{job_id}",
        )
    else:
        runner = asyncio.create_task(
            launch_job(
                record,
                rt,
                client_cwd=client_cwd,
            ),
            name=f"oumi-job-{job_id}",
        )
    rt.runner_task = runner

    launch_confirmed = False
    if not is_local:
        try:
            await asyncio.wait_for(asyncio.shield(runner), timeout=10.0)
            launch_confirmed = rt.error_message is None
        except asyncio.TimeoutError:
            launch_confirmed = False
        except Exception:
            launch_confirmed = False

    logger.info(
        "Job %s submitted on %s — launching `oumi %s` in background",
        job_id,
        cloud,
        command,
    )

    # After cloud launch, record.job_id may have been re-keyed to the
    # SkyPilot ID.  Use it for the response so the agent gets the real ID.
    final_job_id = record.job_id

    if rt.error_message and not is_local:
        overrides: dict[str, Any] = {
            "status": "failed",
            "job_id": final_job_id,
            "config_path": abs_config,
            "model_name": model_name,
            "launch_confirmed": launch_confirmed,
        }
        if preflight_data is not None:
            overrides["preflight"] = preflight_data
        return _error_response(
            f"Failed to launch cloud job: {rt.error_message}",
            **overrides,
        )

    message = (
        f"Job {final_job_id} submitted on {cloud}. "
        f"Use get_job_status('{final_job_id}') for status and "
        f"get_job_logs('{final_job_id}', lines=200) for logs."
    )
    if not is_local and not launch_confirmed:
        message = message + " Launch confirmation is pending; re-check status shortly."

    response: JobSubmissionResponse = {
        "success": True,
        "job_id": final_job_id,
        "status": "submitted",
        "dry_run": False,
        "command": command,
        "config_path": abs_config,
        "cloud": cloud,
        "cluster_name": cluster_name,
        "model_name": model_name,
        "launch_confirmed": launch_confirmed if not is_local else True,
        "message": message,
        "error": None,
        "preflight": preflight_data,
    }
    return response


@mcp.tool()
async def get_job_status(
    job_id: str,
    cloud: str = "",
    cluster_name: str = "",
) -> JobStatusResponse:
    """Return a single status snapshot for an Oumi job.

    For jobs launched by this MCP, just pass job_id — cloud and cluster
    are auto-resolved from the registry. For external jobs, pass all three.
    """
    return await fetch_status(job_id=job_id, cloud=cloud, cluster_name=cluster_name)


@mcp.tool()
async def get_job_logs(
    job_id: str,
    lines: int = DEFAULT_STREAM_LINES,
    cloud: str = "",
    cluster_name: str = "",
) -> JobLogsResponse:
    """Return a bounded log snapshot for an Oumi job."""
    return await fetch_logs(
        job_id=job_id, lines=lines, cloud=cloud, cluster_name=cluster_name
    )


@mcp.tool()
async def cancel_job(
    job_id: str,
    force: bool = False,
    cloud: str = "",
    cluster_name: str = "",
) -> JobCancelResponse:
    """Cancel a running or pending Oumi job.

    Local jobs: SIGTERM (or SIGKILL with force=True).
    Cloud jobs: delegates to ``oumi.launcher.cancel()``.
    """
    return await cancel_job_impl(
        job_id=job_id, force=force, cloud=cloud, cluster_name=cluster_name
    )


@mcp.tool()
async def stop_cluster(cloud: str, cluster_name: str) -> ClusterLifecycleResponse:
    """Stop a running cluster, preserving infra and reducing compute cost.

    Restart by submitting a new job with the same cluster_name.
    Use ``down_cluster`` to fully delete and stop all billing.

    Args:
        cloud: Cloud provider (e.g. ``"gcp"``, ``"aws"``).
        cluster_name: Name of the cluster to stop.
    """
    return await stop_cluster_impl(cloud, cluster_name)


@mcp.tool()
async def down_cluster(
    cloud: str,
    cluster_name: str,
    confirm: bool = False,
    user_confirmation: str = "",
) -> ClusterLifecycleResponse:
    """IRREVERSIBLE: delete a cluster and all its resources.

    Requires ``confirm=True`` and ``user_confirmation="DOWN"``.
    Without these, returns a dry-run description.
    Use ``stop_cluster`` to pause without deleting.

    Args:
        cloud: Cloud provider (e.g. ``"gcp"``, ``"aws"``).
        cluster_name: Name of the cluster to delete.
        confirm: Must be True for actual deletion.
        user_confirmation: Must be exactly ``"DOWN"``.
    """
    return await down_cluster_impl(cloud, cluster_name, confirm, user_confirmation)


@mcp.tool()
async def list_jobs(
    status: str = "all",
) -> list[JobSummary]:
    """List running and completed jobs.

    Args:
        status: ``"all"`` (default), ``"running"``, or ``"completed"``.
    """
    return await _list_job_summaries(status_filter=status)


@mcp.tool()
def get_docs(
    query: list[str],
    module: str = "",
    kind: str = "",
    limit: int = 10,
    summarize: bool = False,
) -> DocsSearchResponse:
    """Search Oumi's indexed Python API docs.

    Matches by: (1) exact qualified name, (2) exact short name, then
    (3) relevance-ranked keyword search over names, fields, and docstrings.

    Args:
        query: Exact names or keywords, e.g. ["TrainingConfig"] or
            ["learning_rate", "lora"].
        module: Module prefix filter (e.g. "oumi.core.configs").
        kind: Kind filter: "class", "dataclass", "function", or "method".
        limit: Max results (default 10).
        summarize: Compact output omitting fields and docstring sections.
    """
    return search_docs(
        query=query, module=module, kind=kind, limit=limit, summarize=summarize
    )


@mcp.tool()
def list_modules() -> ListModulesResponse:
    """List indexed API modules available for ``get_docs`` searches."""
    return get_module_list()


@mcp.resource("jobs://running", mime_type="application/json")
async def list_running_jobs() -> str:
    """List all currently running Oumi jobs.

    Returns a JSON array of job summaries with job_id, command, status,
    cloud, cluster, model_name, and is_done.
    """
    summaries = await _list_job_summaries(status_filter="running")
    return json.dumps(summaries, indent=2)


@mcp.resource("jobs://completed", mime_type="application/json")
async def list_completed_jobs() -> str:
    """List recently completed, failed, or cancelled Oumi jobs.

    Returns a JSON array of job summaries with job_id, command, status,
    cloud, cluster, model_name, and is_done.
    """
    summaries = await _list_job_summaries(status_filter="completed")
    return json.dumps(summaries, indent=2)


@mcp.resource("jobs://{job_id}/logs", mime_type="text/plain")
async def get_job_logs_resource(job_id: str) -> str:
    """Full log output for a specific job.

    For local jobs, returns the contents of the stdout log file on disk.
    For cloud jobs, returns metadata about how to access logs
    (e.g. via ``sky logs``).
    """
    record = get_registry().get(job_id)
    if not record:
        return json.dumps({"error": f"Job '{job_id}' not found"})

    rt = await get_runtime(record.job_id)
    status = await poll_status(record, rt)

    header_parts = [f"Job: {record.job_id}"]
    if status:
        header_parts.append(f"Oumi ID: {status.id}")
        header_parts.append(f"Status: {status.status}")
        header_parts.append(f"Cluster: {status.cluster}")
        header_parts.append(f"Done: {status.done}")
        if status.metadata:
            header_parts.append(f"Metadata: {status.metadata}")
    else:
        header_parts.append(f"Cloud: {record.cloud}")
        if rt.error_message:
            header_parts.append(f"Error: {rt.error_message}")
        else:
            header_parts.append("Status: launching (no status available yet)")

    header = "\n".join(header_parts)

    log_paths = get_log_paths(record, rt)
    stdout_path = log_paths.get("stdout")
    if stdout_path and stdout_path.exists():
        try:
            log_content = stdout_path.read_text(encoding="utf-8", errors="replace")
            return f"{header}\nLog file: {stdout_path}\n\n--- stdout ---\n{log_content}"
        except OSError as exc:
            header += f"\nFailed to read log file: {exc}"

    if record.cloud != "local" and status:
        header += (
            f"\n\nFor cloud jobs, use `sky logs {status.cluster}` "
            f"to stream full logs, or call "
            f"`get_job_logs('{record.job_id}', lines=200)` for a snapshot."
        )

    return header


def main() -> None:
    """Run the MCP server.

    On startup:
    1. Attempts to sync configs from GitHub (skipped if cache is fresh).
    2. Falls back to bundled configs if sync fails and no cache exists.
    3. Starts the MCP server.
    """
    _configure_logging()
    _strip_oumi_env_overrides()
    logger.info("Starting Oumi Config MCP Server")

    def _background_sync() -> None:
        """Sync configs and index docs in background to avoid blocking startup.

        MCP clients (Claude Code, Cursor) have connection timeouts (~10-30s).
        Config sync downloads ~500 files from GitHub and can take 30+ seconds,
        causing the client to give up before the server starts the protocol.
        Running sync in background lets the server accept connections immediately
        while using cached/bundled configs until sync completes.
        """
        sync_result = config_sync()
        if sync_result["ok"]:
            if sync_result.get("skipped"):
                logger.info("Using cached configs (still fresh)")
            else:
                logger.info(
                    f"Config sync completed: {sync_result['configs_synced']} files"
                )
        else:
            logger.warning(
                f"Config sync failed: {sync_result['error']}. "
                "Falling back to bundled/cached configs."
            )

        configs_dir = get_configs_dir()
        yaml_count = (
            len(list(configs_dir.rglob("*.yaml"))) if configs_dir.exists() else 0
        )
        logger.info(f"Serving {yaml_count} configs from {configs_dir}")

        if yaml_count == 0:
            logger.error("No configs available. Server may not function correctly.")

        start_background_indexing()

    threading.Thread(target=_background_sync, daemon=True).start()

    mcp.run()


if __name__ == "__main__":
    main()
