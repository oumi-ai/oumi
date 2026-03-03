# Job ID Simplification & run_oumi_job Cleanup

**Date**: 2026-03-02
**Status**: Approved

## Problem

The MCP job system has two parallel ID systems:
1. **MCP job ID** (e.g. `train_20250302_a7f2b1`) — generated locally, primary key in the registry
2. **SkyPilot/oumi_job_id** — the real cloud identity, stored as a secondary field

If the local JSON registry (`~/.cache/oumi-mcp/oumi-jobs.json`) is lost, the MCP ID becomes useless — but `launcher.status()` still works because SkyPilot is the source of truth.

Additionally, `run_oumi_job` has 19 parameters, most of which are inline cloud config overrides (accelerators, envs, file_mounts, etc.). In practice, agents build a complete job config YAML from documentation and submit it — the inline params add complexity without value.

## Design

### 1. Unified ID System

One ID per job:
- **Cloud jobs**: The SkyPilot job ID returned by `launcher.up()` (integer, per-cluster)
- **Local jobs**: MCP-generated ID (e.g. `train_20250302_a7f2b1`)

The `cloud` field ("local", "gcp", "aws") disambiguates which type of ID it is.

On dry-run, no ID is returned (job doesn't exist yet). On execution:
- Local: returns the MCP ID immediately (process started)
- Cloud: returns the SkyPilot job ID after `launcher.up()` completes

### 2. Simplified `run_oumi_job` Signature

**Before** (19 params):
```
config_path, command, client_cwd, dry_run, confirm, user_confirmation,
job_name, cloud, cluster_name, accelerators, envs, file_mounts,
disk_size, use_spot, num_nodes, setup_script, run_script
```

**After** (9 params):
```python
async def run_oumi_job(
    config_path: str,              # Path to YAML config
    command: str,                  # train, evaluate, infer, etc.
    client_cwd: str,               # User's project root
    dry_run: bool = True,          # Preview mode (default)
    confirm: bool = False,         # Safety gate
    user_confirmation: str = "",   # Must be "EXECUTE"
    job_name: str | None = None,   # Optional name (local jobs only)
    cloud: str = "local",          # "local" or cloud provider
    cluster_name: str = "",        # Target cluster for cloud jobs
) -> JobSubmissionResponse:
```

**Removed**: `accelerators, envs, file_mounts, disk_size, use_spot, num_nodes, setup_script, run_script`

All cloud configuration lives in the job config YAML. Agents build the YAML (using dry-run template + guidance://cloud-launch), customize it, and submit it.

For cloud execution, the config must be a job config (has `resources`/`setup`/`run` keys). Training configs without these keys are rejected with a message pointing to the dry-run template flow.

### 3. Status/Logs/Cancel Tool Signatures

**Before**: Each tool takes `job_id=""`, `oumi_job_id=""`, `cloud=""`, `cluster_name=""` — two lookup paths.

**After**: Single lookup path using the unified ID:

```python
async def get_job_status(
    job_id: str,              # SkyPilot ID (cloud) or MCP ID (local)
    cloud: str = "",          # Auto-resolved from registry if omitted
    cluster_name: str = "",   # Auto-resolved from registry if omitted
) -> JobStatusResponse:

async def get_job_logs(
    job_id: str,
    lines: int = 200,
    cloud: str = "",
    cluster_name: str = "",
) -> JobLogsResponse:

async def cancel_job(
    job_id: str,
    force: bool = False,
    cloud: str = "",
    cluster_name: str = "",
) -> JobCancelResponse:
```

**Cloud status flow**: `launcher.status(cloud=cloud, cluster=cluster_name, id=job_id)` — direct query, registry provides metadata enrichment.

**Local status flow**: Registry lookup → poll subprocess from runtime.

Auto-resolution: If the job was launched via this MCP, the registry knows `cloud` + `cluster_name`. Agent just passes `job_id`. Falls back to requiring explicit params if not in registry.

### 4. Registry Changes

**Before**:
```python
@dataclass
class JobRecord:
    job_id: str          # MCP-generated ID (primary key)
    oumi_job_id: str     # SkyPilot ID (secondary)
    ...
```

**After**:
```python
@dataclass
class JobRecord:
    job_id: str          # Unified: SkyPilot ID (cloud) or MCP ID (local)
    cloud: str           # "local", "gcp", "aws", etc.
    cluster_name: str    # Cluster name (cloud only)
    command: str         # train, evaluate, etc.
    config_path: str     # Absolute path to config
    model_name: str      # HF model ID
    submit_time: str     # ISO 8601
    output_dir: str      # Output directory
```

Removed: `oumi_job_id` field (merged into `job_id`).

Source of truth: `launcher.status()` for cloud jobs; subprocess poll for local jobs. Registry = convenience cache for metadata enrichment.

### 5. File Changes

| File | Changes |
|------|---------|
| `models.py` | Remove `oumi_job_id` from `JobRecord`. Update response TypedDicts to drop dual-ID fields. |
| `job_service.py` | Remove `make_job_id` usage for cloud. `_launch_cloud` sets `record.job_id` = SkyPilot ID after `launcher.up()`. Simplify `fetch_status`/`fetch_logs` to unified ID lookup. Remove `_fetch_cloud_status_direct` (now the main path). Remove `_generate_job_config_template` and `_build_cloud_job_config` (inline config generation). |
| `server.py` | Strip `run_oumi_job` to 9 params. Remove `oumi_job_id` from status/logs/cancel tools. Remove inline cloud config generation from dry-run path. |
| `prompts/mle_prompt.py` | Update workflow docs to show simplified tool signatures. |
| Tests | Update to use unified IDs. |

### 6. What Stays the Same

- Two-step safety (dry_run → confirm + EXECUTE)
- Pre-flight checks for cloud jobs
- Job config passthrough detection
- Dry-run template generation (training config → job config template with TODOs)
- Local job subprocess management
- Cloud log streaming via `cluster.get_logs_stream()`
- Registry pruning (7 days, 200 max)
- `list_jobs`, `stop_cluster`, `down_cluster` unchanged in interface
