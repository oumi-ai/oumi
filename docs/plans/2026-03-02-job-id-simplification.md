# Job ID Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify the dual-ID job system (MCP ID + SkyPilot ID) into a single ID per job, simplify `run_oumi_job` from 19 to 9 params, and make `launcher.status()` the source of truth for cloud jobs.

**Architecture:** Remove `oumi_job_id` field from `JobRecord`, re-key registry by SkyPilot ID for cloud jobs and MCP ID for local jobs. Strip inline cloud config params from `run_oumi_job` — agents must submit complete job config YAMLs. Simplify status/logs/cancel tools to use one `job_id` param with auto-resolution from registry.

**Tech Stack:** Python, FastMCP, oumi.launcher, asyncio

---

### Task 1: Update `JobRecord` dataclass — remove `oumi_job_id`

**Files:**
- Modify: `src/oumi/mcp/job_service.py:55-74`

**Step 1: Edit JobRecord**

Remove the `oumi_job_id` field. The `job_id` field now holds the SkyPilot ID (cloud) or MCP-generated ID (local).

```python
@dataclass
class JobRecord:
    """Persisted job metadata — identity mapping only.

    All fields are strings for simple JSON serde.
    The registry does NOT store job status; status is always
    queried live from ``oumi.launcher.status()`` (cloud) or
    ``rt.process.poll()`` (local).

    For cloud jobs, ``job_id`` is the SkyPilot job ID.
    For local jobs, ``job_id`` is an MCP-generated ID (e.g. ``train_20250302_a7f2b1``).
    The ``cloud`` field disambiguates which type of ID it is.
    """

    job_id: str
    command: str
    config_path: str
    cloud: str
    cluster_name: str
    model_name: str
    submit_time: str  # ISO 8601
    output_dir: str = ""
```

**Step 2: Update `JobRegistry.find_by_cloud_identity`**

This method searched by `oumi_job_id`. Replace it with a lookup by `job_id` + `cloud`:

```python
def find_by_cloud(self, cloud: str, job_id: str) -> JobRecord | None:
    """Find a record by cloud provider and job ID."""
    for r in self._jobs.values():
        if r.cloud == cloud and r.job_id == job_id:
            return r
    return None
```

Note: The registry key is `record.job_id` which is now the SkyPilot ID for cloud jobs. So `self._jobs.get(job_id)` should already work for direct lookups. This method is a fallback scan.

**Step 3: Run tests to check for immediate breakage**

Run: `python -m pytest tests/unit/mcp/ -x -q 2>&1 | head -40`
Expected: Some failures due to old `oumi_job_id` references — that's expected, we'll fix them in later tasks.

**Step 4: Commit**

```bash
git add src/oumi/mcp/job_service.py
git commit -m "refactor(mcp): remove oumi_job_id from JobRecord, unify to single job_id"
```

---

### Task 2: Update response TypedDicts in `models.py`

**Files:**
- Modify: `src/oumi/mcp/models.py:172-310`

**Step 1: Update `JobSubmissionResponse`**

Remove `oumi_job_id` and `cluster` (redundant with `cluster_name`) NotRequired fields:

In the TypedDict at lines 172-216, remove these two lines:
```python
    oumi_job_id: NotRequired[str]
    cluster: NotRequired[str]
```

**Step 2: Update `JobStatusResponse`**

Remove `oumi_job_id` field at lines 218-252. Change:
```python
    success: bool
    job_id: str
    oumi_job_id: str  # REMOVE THIS LINE
    status: str
    ...
```

**Step 3: Commit**

```bash
git add src/oumi/mcp/models.py
git commit -m "refactor(mcp): remove oumi_job_id from response TypedDicts"
```

---

### Task 3: Simplify `run_oumi_job` signature in `server.py`

**Files:**
- Modify: `src/oumi/mcp/server.py:394-782`

**Step 1: Strip inline cloud params from function signature**

Replace the current 19-param signature (lines 394-413) with:

```python
@mcp.tool()
async def run_oumi_job(
    config_path: str,
    command: str,
    client_cwd: str,
    dry_run: bool = True,
    confirm: bool = False,
    user_confirmation: str = "",
    job_name: str | None = None,
    cloud: str = "local",
    cluster_name: str = "",
) -> JobSubmissionResponse:
    """Execute an Oumi CLI command with background job tracking.

    Two-step safety: call with dry_run=True (default) to preview, then
    dry_run=False, confirm=True, user_confirmation="EXECUTE" to launch.
    Cloud runs execute a pre-flight check that may block launch.

    Job configs (with ``resources``/``setup``/``run`` keys) pass through
    directly to ``oumi launch up``. Training configs generate a job config
    template on dry-run for the agent to customize and re-submit.

    Args:
        config_path: Absolute path, or relative to client_cwd, to an Oumi YAML config.
        command: Oumi subcommand: train, analyze, synth, evaluate, eval,
            infer, tune, quantize. Ignored for job configs.
        client_cwd: REQUIRED. Absolute path to the client's working directory.
        dry_run: Preview execution plan without running (default True).
        confirm: Must be True for actual execution.
        user_confirmation: Must be ``"EXECUTE"`` when dry_run=False.
        job_name: Optional name; auto-generated if omitted.
        cloud: ``"local"`` (default) or a cloud provider name.
        cluster_name: Cluster name for cloud launches.
    """
```

**Step 2: Update the function body**

Key changes inside the function:
1. Remove all references to removed params (`accelerators`, `envs`, `file_mounts`, `disk_size`, `use_spot`, `num_nodes`, `setup_script`, `run_script`)
2. Remove `num_gpus = _parse_gpu_count(accelerators or None)` — GPU count comes from the job config YAML
3. In the dry-run path for cloud: keep `_generate_job_config_template` call but remove the inline override params. The template is generated from the training config + command only.
4. In the execution path for cloud: require the config to be a job config (has `resources`/`setup`/`run` keys). If it's a training config, return an error telling the agent to use dry-run first to get the template.
5. Remove `_build_cloud_job_config` usage — no more inline wrapping.
6. In `launch_job` / `_launch_cloud` calls, remove the inline cloud params.
7. For cloud execution: after `launcher.up()` returns, set `record.job_id` to the SkyPilot job ID (overwriting the temporary MCP ID).
8. Remove `oumi_job_id` and `cluster` from all response dicts.

**Step 3: Simplify dry-run for cloud training configs**

The dry-run template generation (`_generate_job_config_template`) should still work but with fewer params. Update the call to only pass what's available:

```python
job_config_template = _generate_job_config_template(
    abs_config,
    command,
    cloud,
    model_name,
    client_cwd=client_cwd,
    job_name=job_id,
)
```

**Step 4: For cloud execution, require job config format**

After the dry-run block, add a check:
```python
if cloud != "local" and not is_job_config_file:
    return _error_response(
        "Cloud execution requires a job config file (with resources/setup/run keys). "
        "Run with dry_run=True first to generate a job config template from your "
        "training config, save it as a YAML file, customize it, then re-submit.",
        config_path=abs_config,
    )
```

**Step 5: Update cloud job ID assignment**

After `launcher.up()` returns in `_launch_cloud`, update the registry key:
```python
# In _launch_cloud, after getting status from launcher.up():
sky_job_id = status.id  # SkyPilot job ID
old_id = record.job_id
record.job_id = sky_job_id
reg = get_registry()
reg.remove(old_id)  # Remove the temp MCP ID entry
reg.add(record)     # Re-add with SkyPilot ID as key
```

**Step 6: Commit**

```bash
git add src/oumi/mcp/server.py src/oumi/mcp/job_service.py
git commit -m "refactor(mcp): simplify run_oumi_job to 9 params, use SkyPilot ID for cloud jobs"
```

---

### Task 4: Simplify `launch_job` and `_launch_cloud` in `job_service.py`

**Files:**
- Modify: `src/oumi/mcp/job_service.py:702-866`

**Step 1: Strip inline cloud params from `launch_job`**

```python
async def launch_job(
    record: JobRecord,
    rt: JobRuntime,
    *,
    client_cwd: str = "",
) -> None:
    """Launch a job -- local or cloud -- in a background thread."""
    if record.cloud == "local":
        start_local_job(record, rt, client_cwd=client_cwd)
        await wait_local_completion(record, rt)
    else:
        await _launch_cloud(record, rt, client_cwd=client_cwd)
```

**Step 2: Strip inline cloud params from `_launch_cloud`**

```python
async def _launch_cloud(
    record: JobRecord,
    rt: JobRuntime,
    *,
    client_cwd: str = "",
) -> None:
    """Launch a cloud job via ``oumi.launcher.up()``.

    The config file must be a job config (has resources/setup/run keys).
    Loads via ``launcher.JobConfig.from_yaml()`` so all cloud-specific
    fields are preserved as written.
    """
```

Remove the training-config wrapping branch entirely. Only keep the job-config passthrough path:
```python
    try:
        job_cfg = launcher.JobConfig.from_yaml(record.config_path)
    except Exception as exc:
        rt.error_message = f"Failed to parse job config: {exc}"
        return
```

**Step 3: After `launcher.up()`, re-key the registry**

```python
    # After launcher.up() returns (cluster, status):
    sky_job_id = str(status.id) if status else ""
    if sky_job_id:
        old_id = record.job_id
        record.job_id = sky_job_id
        record.cluster_name = status.cluster or record.cluster_name
        reg = get_registry()
        reg.remove(old_id)
        reg.add(record)
```

**Step 4: Remove `_build_cloud_job_config` function**

Delete the function at lines 416-472. It's no longer called since we don't do inline training-config wrapping.

**Step 5: Simplify `_generate_job_config_template`**

Remove the inline override params (`accelerators`, `num_nodes`, `envs`, `setup_script`, `run_script`) from its signature. Keep only the essential params needed to generate a template:

```python
def _generate_job_config_template(
    config_path: str,
    command: str,
    cloud: str,
    model_name: str,
    *,
    client_cwd: str = "",
    job_name: str = "",
) -> str:
```

Update the template generation to always use TODO placeholders for accelerators, setup, etc. (no pre-filling from inline params).

**Step 6: Commit**

```bash
git add src/oumi/mcp/job_service.py
git commit -m "refactor(mcp): simplify launch_job and _launch_cloud, remove _build_cloud_job_config"
```

---

### Task 5: Simplify status/logs/cancel tool signatures

**Files:**
- Modify: `src/oumi/mcp/server.py:786-907`
- Modify: `src/oumi/mcp/job_service.py:1383-1805`

**Step 1: Update `get_job_status` in server.py**

```python
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
    return await fetch_status(
        job_id=job_id, cloud=cloud, cluster_name=cluster_name,
    )
```

**Step 2: Update `get_job_logs` in server.py**

```python
@mcp.tool()
async def get_job_logs(
    job_id: str,
    lines: int = DEFAULT_STREAM_LINES,
    cloud: str = "",
    cluster_name: str = "",
) -> JobLogsResponse:
    """Return a bounded log snapshot for an Oumi job."""
    return await fetch_logs(
        job_id=job_id, lines=lines, cloud=cloud, cluster_name=cluster_name,
    )
```

**Step 3: Update `cancel_job` in server.py**

```python
@mcp.tool()
async def cancel_job(
    job_id: str,
    force: bool = False,
    cloud: str = "",
    cluster_name: str = "",
) -> JobCancelResponse:
    """Cancel a running or pending Oumi job."""
    return await cancel_job_impl(
        job_id=job_id, force=force, cloud=cloud, cluster_name=cluster_name,
    )
```

**Step 4: Simplify `_resolve_job_record` in job_service.py**

```python
def _resolve_job_record(
    *,
    job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobRecord | None:
    """Resolve a job record by unified job_id.

    Tries direct registry lookup first. If cloud is provided,
    also tries find_by_cloud as a fallback.
    """
    reg = get_registry()
    if job_id:
        record = reg.get(job_id)
        if record:
            return record
        # For cloud jobs, SkyPilot IDs might need cloud context
        if cloud:
            return reg.find_by_cloud(cloud, job_id)
    return None
```

**Step 5: Simplify `fetch_status` in job_service.py**

Remove the `oumi_job_id` param. The function now takes `job_id`, `cloud`, `cluster_name`:

```python
async def fetch_status(
    *,
    job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobStatusResponse:
```

When record is not found in registry but `cloud` is provided, query `launcher.status()` directly using `job_id` as the SkyPilot ID:

```python
    if not record:
        if not cloud:
            return _not_found_response(job_id)
        direct_status = await _fetch_cloud_status_direct(
            job_id=job_id, cloud=cloud, cluster_name=cluster_name,
        )
        ...
```

**Step 6: Update `_fetch_cloud_status_direct`**

Rename `oumi_job_id` param to `job_id`:

```python
async def _fetch_cloud_status_direct(
    *, job_id: str, cloud: str, cluster_name: str = "",
) -> Any | None:
    try:
        statuses_by_cloud = await asyncio.to_thread(
            launcher.status, cloud=cloud, cluster=cluster_name or None, id=job_id,
        )
    except Exception:
        return None
    for _cloud_name, statuses in statuses_by_cloud.items():
        for status in statuses:
            if status.id == job_id:
                return status
    return None
```

**Step 7: Update `fetch_logs` and `cancel_job_impl` similarly**

Remove `oumi_job_id` param, use `job_id` throughout.

**Step 8: Remove `oumi_job_id` from `_build_status_response`**

Find where responses are constructed and remove all `oumi_job_id` fields.

**Step 9: Commit**

```bash
git add src/oumi/mcp/server.py src/oumi/mcp/job_service.py
git commit -m "refactor(mcp): simplify status/logs/cancel to single job_id param"
```

---

### Task 6: Update `list_jobs` and job resource endpoints

**Files:**
- Modify: `src/oumi/mcp/server.py` (list_jobs, job resources)
- Modify: `src/oumi/mcp/job_service.py` (_list_job_summaries)

**Step 1: Update `_list_job_summaries` in job_service.py**

Find the function that builds `JobSummary` dicts. Remove any `oumi_job_id` references. The `job_id` in the summary is now the unified ID.

**Step 2: Update job resource endpoints**

Find `get_job_logs_resource` and any `jobs://running` / `jobs://completed` resource handlers. Remove `oumi_job_id` references, use `record.job_id` directly.

**Step 3: Commit**

```bash
git add src/oumi/mcp/server.py src/oumi/mcp/job_service.py
git commit -m "refactor(mcp): update list_jobs and resource endpoints to unified ID"
```

---

### Task 7: Update `start_local_job` to not set `oumi_job_id`

**Files:**
- Modify: `src/oumi/mcp/job_service.py:611-664`

**Step 1: Remove `record.oumi_job_id = str(proc.pid)` line**

In `start_local_job`, find where it sets `record.oumi_job_id` to the PID. Remove that line. The `job_id` for local jobs is already set (the MCP-generated ID). The PID can be stored on the runtime (`rt`) if needed for process management, but it's not an ID field.

**Step 2: Commit**

```bash
git add src/oumi/mcp/job_service.py
git commit -m "refactor(mcp): remove oumi_job_id assignment in start_local_job"
```

---

### Task 8: Update MLE prompt and workflow docs

**Files:**
- Modify: `src/oumi/mcp/prompts/mle_prompt.py`

**Step 1: Update tool signatures in GET_STARTED_CONTENT**

Update the Execution table to show simplified signatures:
- `run_oumi_job(config, cmd, client_cwd)` — no change needed (already shows minimal)
- `get_job_status(job_id)` — remove mention of `oumi_job_id`
- Update the cloud workflow steps to reflect that agents must build a job config YAML

**Step 2: Update the cloud workflow steps**

The current flow (lines 96-118) shows inline param usage. Update to reflect the new flow:
1. pre_flight_check
2. dry-run → generates job config template
3. Save template, customize
4. Submit the job config
5. Get status using the returned job_id

**Step 3: Update the execution pattern**

Remove `oumi_job_id` references from examples. Show that `get_job_status(job_id)` works directly.

**Step 4: Commit**

```bash
git add src/oumi/mcp/prompts/mle_prompt.py
git commit -m "docs(mcp): update MLE prompt for simplified job tool signatures"
```

---

### Task 9: Update tests

**Files:**
- Modify: `tests/unit/mcp/test_server_tools.py`
- Modify: `tests/unit/mcp/test_config_service.py` (if job-related)
- Modify: any other test files referencing `oumi_job_id`

**Step 1: Find all test references to old API**

Run: `grep -rn "oumi_job_id\|accelerators\|file_mounts\|setup_script\|run_script" tests/unit/mcp/`

**Step 2: Update test fixtures and assertions**

- Remove `oumi_job_id` from any `JobRecord` construction
- Update `run_oumi_job` calls to use simplified signature
- Update assertions on response dicts to not check for `oumi_job_id`

**Step 3: Run all MCP tests**

Run: `python -m pytest tests/unit/mcp/ -v 2>&1 | tail -30`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/unit/mcp/
git commit -m "test(mcp): update tests for unified job ID system"
```

---

### Task 10: Clean up unused imports and dead code

**Files:**
- Modify: `src/oumi/mcp/job_service.py`
- Modify: `src/oumi/mcp/server.py`

**Step 1: Remove unused functions**

- `_build_cloud_job_config` (already removed in Task 4)
- `find_by_cloud_identity` on `JobRegistry` (replaced by `find_by_cloud`)
- Any helper functions only called by removed code

**Step 2: Remove unused imports**

Run: `ruff check src/oumi/mcp/job_service.py src/oumi/mcp/server.py --select F401`

Fix any unused import warnings.

**Step 3: Run full test suite**

Run: `python -m pytest tests/unit/mcp/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add src/oumi/mcp/
git commit -m "refactor(mcp): remove dead code and unused imports from job simplification"
```
