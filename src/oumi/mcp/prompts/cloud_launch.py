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

# ruff: noqa: E501

"""Cloud job launch guidance resource.

Covers SkyPilot-based cloud job config anatomy, path resolution, file mounts,
storage mounts, and common setup patterns.
Exposed as `guidance://cloud-launch`.
"""

CLOUD_LAUNCH_RESOURCE = """# Cloud Job Launch Guide

## Why You Need a Job Config for Cloud Runs

Cloud jobs run on remote VMs managed by SkyPilot. Unlike local runs where
`oumi train -c config.yaml` is enough, cloud runs need additional information:
- **setup**: How to install Oumi and dependencies on the VM
- **run**: The shell command to execute
- **working_dir**: Which local files to sync to the remote VM
- **file_mounts**: Credential files (HF token, .netrc) to copy
- **storage_mounts**: Persistent cloud storage for outputs (important for spot instances)
- **envs**: Environment variables needed on the VM
- **resources**: Cloud provider, GPU type, disk size, spot/on-demand

## Version Compatibility

The MCP tools document Oumi 0.7 APIs. If the cloud VM installs a different version
(e.g. `pip install oumi[gpu]` pulls 0.1.x), some field names may differ:
- `evaluation_backend` (0.7) vs `evaluation_platform` (0.1.x)

**To avoid mismatches**, pin the version in your setup script:
```bash
pip install 'oumi[gpu]>=0.7'
```
Or use `get_docs()` to check the installed version's API.

## How Path Resolution Works

When you call `run_oumi_job(config_path, command, client_cwd)`:

1. **`client_cwd`** = absolute path to the user's project root
2. **`config_path`** is resolved relative to `client_cwd`
3. For cloud jobs, `client_cwd` becomes the **`working_dir`** in the job config
4. SkyPilot rsyncs `working_dir` to `~/sky_workdir` on the remote VM
5. The `run` command executes from `~/sky_workdir`, so **repo-relative paths** resolve correctly
6. Only **git-tracked files** are synced (when `.gitignore` is present); untracked files are silently skipped

**Always pass `client_cwd`** — without it, the MCP server's own working directory is used
(which is NOT the user's project root), and files will not be found.

## Key Fields

### `setup` (shell script)
Runs once when the VM is provisioned. Install Oumi, download datasets, install extras:
```bash
set -e
pip install uv && uv pip install --system 'oumi[gpu]'
huggingface-cli download <dataset-id> --repo-type dataset --local-dir ./data
uv pip install --system flash-attn
```

### `run` (shell script)
The training command. For multi-GPU, use `oumi distributed torchrun`:
```bash
set -e
oumi train -c ./config.yaml
# Multi-GPU:
# oumi distributed torchrun -m oumi train -c ./config.yaml
```

### `working_dir`
Local directory synced to the remote VM via rsync. **Use `working_dir: .`** (the
default) — `client_cwd` resolves it to the user's project root at launch time.
Do NOT embed absolute local paths like `/Users/you/project` — they make the
config non-portable.

The training config file is placed inside this directory on the VM at `~/sky_workdir/`.

**Important:** Only git-tracked files are synced when a `.gitignore` is present.
If a file is not in git, SkyPilot will silently skip it. Use `file_mounts` for
untracked files (see below).

### Path Conventions for Cloud Jobs

Use **repo-relative paths** for project files (data, configs, output). These
resolve from `working_dir` on the remote VM after sync.

| Path type | Convention | Example |
|-----------|-----------|---------|
| Project files (data, configs) | Repo-relative | `data/pubmed_qa/train.jsonl` |
| Config references | Repo-relative | `configs/train.yaml` |
| Training output | Repo-relative | `output/...` |
| Remote-only output | Remote absolute | `/home/ubuntu/output/...` |
| Local machine paths | **NEVER use these** | `/Users/you/project/data/...` |

**Never** use local machine paths in cloud configs — they do not exist on the
remote VM. The pre-flight check blocks these automatically.

### How Dataset Files Reach the Remote VM

There are 3 ways a dataset file can be available on the VM:

1. **`working_dir` sync** — Git-tracked files inside `working_dir` are automatically
   rsynced to `~/sky_workdir`. Reference them with repo-relative paths
   (e.g. `./data/train.jsonl`). Untracked files are silently skipped.

2. **`file_mounts`** — Explicitly copy local files to the VM. Use this for datasets
   that are NOT git-tracked or are outside the project directory:
   ```yaml
   file_mounts:
     ~/sky_workdir/data/train.jsonl: ./local-data/train.jsonl
   ```
   Then reference as `./data/train.jsonl` in the training config.

3. **`setup` script download** — Download from HuggingFace or cloud storage during VM setup:
   ```bash
   huggingface-cli download my-org/my-dataset --repo-type dataset --local-dir ./data
   ```

### `storage_mounts`
Mount cloud storage buckets for persistent output. Critical for spot instances
where the VM can be preempted:
```yaml
storage_mounts:
  /output:
    source: gs://your-bucket/training-output
    store: gcs
```

### `file_mounts`
Copy local files to the remote VM. Credential files (HF token, .netrc) are
auto-detected and mounted automatically.

Use `file_mounts` for **local dataset files** that are either:
- Outside your `working_dir`, OR
- Not git-tracked (SkyPilot skips untracked files during working_dir sync)

```yaml
file_mounts:
  # Local datasets to remote VM paths
  ~/sky_workdir/data/train.jsonl: ./datasets/train.jsonl
  ~/sky_workdir/data/val.jsonl: ./datasets/val.jsonl
```

Then reference the data in your training config as `./data/train.jsonl`
(relative to `working_dir` = `~/sky_workdir`).

### `envs`
Environment variables set on the remote VM. Local env vars are NOT forwarded
automatically. Set any required API keys or project identifiers here.

**Important:** Do not hardcode secret values directly in config files that may
be committed to version control. Prefer referencing environment variables or
using a secrets manager.

## Example Job Config

```yaml
name: train-llama3-sft
resources:
  cloud: gcp
  accelerators: "A100:8"
  use_spot: false
  disk_size: 500
num_nodes: 1
working_dir: .  # Resolved to client_cwd at launch time

file_mounts:
  ~/.cache/huggingface/token: ~/.cache/huggingface/token
  ~/.netrc: ~/.netrc

storage_mounts:
  /output:
    source: gs://my-bucket/training-output
    store: gcs

envs:
  WANDB_PROJECT: "llama3-sft"

setup: |
  set -e
  pip install uv && uv pip install --system 'oumi[gpu]'
  huggingface-cli download my-org/my-dataset --repo-type dataset --local-dir ./data

run: |
  set -e
  oumi train -c ./config.yaml
```

## How `run_oumi_job` Works

Cloud jobs require a job config (with `resources`, `setup`, `run` keys).
If you pass a training config with `cloud` set to a provider, the tool
returns an error directing you to build a job config first.

### Workflow:
1. **Build** a job config YAML using this guide as reference
2. **Preview**: `run_oumi_job(config_path="job.yaml", command="train", client_cwd=CWD, cloud="gcp")` (dry_run=True by default)
3. **Execute**: `run_oumi_job(config_path="job.yaml", command="train", client_cwd=CWD, cloud="gcp", dry_run=False)` — confirm with the user before running

Local jobs accept training configs directly — no job config needed.

## Common Setup Patterns

### Fine-tuning a gated model (Llama, Gemma):
```bash
set -e
pip install uv && uv pip install --system 'oumi[gpu]'
huggingface-cli whoami  # verify HF auth works
```

### Training with custom dataset from HuggingFace:
```bash
set -e
pip install uv && uv pip install --system 'oumi[gpu]'
huggingface-cli download my-org/my-dataset --repo-type dataset --local-dir ./data
```

### Training with evaluation:
```bash
set -e
pip install uv && uv pip install --system 'oumi[gpu,evaluation]'
```

## Troubleshooting: File Not Found on VM

| Cause | Symptom | Fix |
|-------|---------|-----|
| Missing `client_cwd` | `FileNotFoundError: configs/train.yaml` | Always pass `client_cwd` to `run_oumi_job` |
| File not git-tracked | File exists locally but missing on VM | `git add <file>` and commit, or use `file_mounts` |
| Local absolute path in cloud config | Path does not exist on VM | Use repo-relative paths |
| File outside `working_dir` | Only `working_dir` contents are synced | Use `file_mounts` to copy files from other locations |

**Diagnosis steps:**
1. Check the dry-run output — it shows the resolved `working_dir` and generated job config
2. Run `git status` in the project — untracked files will not sync
3. Verify paths in the training config are repo-relative, not absolute local paths

## Ephemeral Storage

Training outputs on the cluster's local disk are **not preserved** across cluster stops
or recreations. If the cluster is stopped, restarted, or torn down, all local files
(checkpoints, adapters, logs) are lost.

**Before stopping or deleting a cluster:**
1. Use `sky rsync-down <cluster> ~/sky_workdir/<output_dir> ./local_output/` to download artifacts
2. Or configure the training config to save checkpoints to a cloud bucket (S3/GCS) via `storage_mounts`

## Existing Clusters and File Sync

When submitting a job to an **existing** cluster, SkyPilot uses `sky exec` instead of
`sky launch`. This means:
- Local file changes are **NOT re-synced** to the VM
- The VM still has the files from the original `sky launch`

**If you edited files locally and need them on the VM:**
- Use `sky launch` with the same `cluster_name` to force a full re-sync
- Or use explicit `file_mounts` for files that change between submissions
"""
