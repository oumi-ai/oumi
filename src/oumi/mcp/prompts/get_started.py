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

"""Get-started prompt returned by the `get_started()` MCP tool.

This is the first thing an agent sees when connecting to the Oumi MCP server.
It contains the full tool catalog, path rules, resource index, and quickstart
workflows that the agent needs before calling any other tool.
"""

GET_STARTED_CONTENT = """# Oumi MCP - ML Training Config Server

> **You called `get_started()`.** This is the required first step
> before using any other Oumi MCP tool. Read the workflow below
> to understand tool ordering, path rules, and config usage.

## Available Tools

### Discovery
| Tool | Purpose | Example |
|------|---------|---------|
| `list_categories()` | List available models and config types | Start here |
| `search_configs(query, content_match, limit)` | Find training configs | `search_configs(query=["llama3_1", "sft"])` |
| `get_config(path)` | Get a **reference** config (see usage note below) | `get_config("llama3_1/sft/8b_lora")` |
| `validate_config(config, task_type, client_cwd)` | Validate before training | `validate_config("configs/train.yaml", "training", client_cwd="/home/user/project")` |
| `pre_flight_check(config, client_cwd, cloud)` | Catch issues before launch | `pre_flight_check("configs/train.yaml", client_cwd="/home/user/project", cloud="gcp")` |
| `get_docs(query, module, kind)` | Search Oumi Python API docs | `get_docs(["TrainingConfig"])` |
| `list_modules()` | List indexed API modules | `list_modules()` |

### Execution
| Tool | Purpose | Example |
|------|---------|---------|
| `run_oumi_job(config_path, command, client_cwd)` | Execute Oumi command (dry-run by default) | `run_oumi_job("configs/train.yaml", "train", client_cwd="/home/user/project")` |
| `get_job_status(job_id)` | Status snapshot (no streaming) | `get_job_status("train_20260206_...")` |
| `get_job_logs(job_id, lines)` | Tail log snapshot | `get_job_logs("train_20260206_...", lines=200)` |
| `cancel_job(job_id)` | Cancel a running job | `cancel_job("train_20260206_...")` |
| `list_jobs()` | List running and completed jobs | `list_jobs(status="running")` |
| `stop_cluster(cloud, cluster_name)` | Stop cluster (preserves infra, reduces compute cost) | `stop_cluster("gcp", "sky-xxxx")` |
| `down_cluster(cloud, cluster_name, confirm, user_confirmation)` | Delete cluster entirely (irreversible) | See cluster lifecycle section |

### How to Use `get_config` Correctly

Configs returned by `get_config(path)` are **reference recipes** showing the
correct YAML structure, field names, and sensible defaults for a model/task
combination. **Do NOT copy them verbatim.**

Instead:
1. Study the structure and note which fields are relevant to the user's task.
2. Build a NEW config from scratch, adapting only the applicable settings
   (model name, dataset, training params, PEFT config, output dir).
3. Customize values for the user's specific hardware, data, and goals.
4. Omit sections that don't apply (e.g., drop `peft:` for full fine-tuning).

Copying a reference config wholesale leads to wrong datasets, wrong output
paths, and unnecessary settings.

## MCP Resources

### Guidance
| Resource | What it contains | When to use |
|----------|------------------|-------------|
| `guidance://mle-workflow` | End-to-end MLE workflow, decision checkpoints | Full playbook, new project |
| `guidance://mle-train` | Training command usage, sizing heuristics | Planning or running training |
| `guidance://mle-synth` | Synthetic data generation flow | Generating synthetic datasets |
| `guidance://mle-analyze` | Dataset analysis, quality checks | Before training, data audit |
| `guidance://mle-eval` | Evaluation strategies, benchmarks | Benchmarking or comparing runs |
| `guidance://mle-infer` | Inference best practices | Running inference or sanity checks |
| `guidance://cloud-launch` | Cloud job config anatomy, setup patterns | Before launching a cloud training run |
| `guidance://post-training` | Download weights, evaluate, teardown, merge LoRA | After cloud training succeeds |

### Jobs
| Resource | What it contains |
|----------|-----------------|
| `jobs://running` | Currently running jobs (JSON array) |
| `jobs://completed` | Recently finished jobs (JSON array) |
| `jobs://{job_id}/logs` | Full log output for a specific job |

## CRITICAL: Working Directory and Paths

**The MCP server runs in a DIFFERENT directory than the user's project.**
You MUST pass `client_cwd` (absolute path to the project root) to all path-sensitive tools.

**What `client_cwd` does:**
- Resolves relative config paths (e.g. `configs/train.yaml` becomes `/home/user/project/configs/train.yaml`)
- Sets the subprocess working directory for local jobs
- Becomes the `working_dir` synced to the remote VM for cloud jobs

**Example:** If the user's project is at `/home/user/my-project`:
```
validate_config("configs/train.yaml", "training", client_cwd="/home/user/my-project")
run_oumi_job(config_path="configs/train.yaml", command="train", client_cwd="/home/user/my-project")
```

**Paths inside configs:**
- **Local jobs**: absolute or relative to `client_cwd` (resolved at runtime)
- **Cloud jobs**: repo-relative paths only (resolve from `working_dir` on the remote VM)
  - BAD: `/Users/you/data/train.jsonl` (does not exist on the VM)
  - GOOD: `data/train.jsonl` (resolves from synced `working_dir`)

## Cloud Job Workflow

**When a user asks to run a cloud training job, follow these steps in order:**

```
CWD = "/home/user/my-project"  # user's project root — pass as client_cwd everywhere

Step 1: pre_flight_check("configs/train.yaml", client_cwd=CWD, cloud="gcp")
        # Check credentials; use suggested_configs paths with get_config() for reference YAMLs
Step 2: Build a job config YAML using guidance://cloud-launch as reference
        Key fields: resources (cloud, accelerators), working_dir, setup, run, envs, file_mounts
Step 3: run_oumi_job(config_path="job.yaml", command="train", client_cwd=CWD, cloud="gcp")
        # dry-run (default) to verify
Step 4: run_oumi_job(config_path="job.yaml", command="train", client_cwd=CWD, cloud="gcp",
        dry_run=False)
        # Execute for real — CONFIRM with the user before running
Step 5: get_job_status(job_id)    # poll status
Step 6: get_job_logs(job_id, lines=200)  # check logs
Step 7: Cluster teardown — see "Cluster Lifecycle" below
```

**Key fields to customize in your cloud job YAML:**
- `resources.accelerators` — GPU type and count (e.g. `"A10G:1"`, `"A100:8"`)
- `working_dir` — use `.` (default); resolved to `client_cwd` at launch time
- `run` — your oumi command (path relative to `working_dir`)
- `envs` — environment variables needed on the VM (see `guidance://cloud-launch` for details)
- `file_mounts` — credential files auto-included; add local dataset files if not git-tracked

**Tip:** Read `guidance://cloud-launch` for detailed job config field explanations and common
setup patterns (dataset downloads, extra packages, storage mounts).

## Local Quickstart Workflow

1. **Discover models**: `list_categories()` to see model families
2. **Find recipes**: `search_configs(query=["llama3_1", "sft"])`
3. **Study reference**: `get_config("llama3_1/sft/8b_lora")` — read for structure and defaults, do NOT copy verbatim
4. **Build config**: Create a new config for the user's model, dataset, hardware, and goals — use the reference to inform field names and reasonable values
5. **Validate**: `validate_config("configs/train.yaml", "training", client_cwd="/home/user/project")`
6. **Preview**: `run_oumi_job(config_path="configs/train.yaml", command="train", client_cwd="/home/user/project")` (dry-run by default)
7. **Execute**: `run_oumi_job(config_path="configs/train.yaml", command="train", client_cwd="/home/user/project", dry_run=False)` — confirm with the user first
8. **Monitor**: `get_job_status("train_20260206_...")` and `get_job_logs("train_20260206_...", lines=200)`

## Cluster Lifecycle

After a cloud job finishes (or to manage costs):

```
Step 1: get_job_status(job_id)  # see "cluster" and "cloud" fields
Step 2a (pause):  stop_cluster("gcp", "sky-xxxx")
        # Keeps infra, reduces cost, cluster is restartable
Step 2b (delete): down_cluster("gcp", "sky-xxxx", confirm=True, user_confirmation="DOWN")
        # Permanently deletes everything — irreversible
```

- **`stop_cluster`**: Pauses compute. Storage costs may still apply. Cluster is restartable.
- **`down_cluster`**: Permanently deletes the cluster. No more billing. **Irreversible.**
- ALWAYS confirm with the user before calling `down_cluster`.
- Always tear down clusters when training is complete to avoid ongoing storage charges.

## Search Parameters

- **query**: Terms matched against config paths (AND logic, case-insensitive). Paths encode model family, size, task, and technique.
- **content_match**: Substrings matched against YAML file content (AND logic, case-insensitive). Use for values not in the path, e.g. a dataset name or HuggingFace model ID.
- **limit**: Maximum number of results to return (default 20).

## Config Key Settings

When customizing a config, these are the key fields to modify:
- `model.model_name`: HuggingFace model ID
- `data.train.datasets`: Dataset list (see dataset_name rules below)
- `training.output_dir`: Where to save checkpoints
- `training.learning_rate`: Start with recipe default
- `training.per_device_train_batch_size`: Adjust for your GPU memory

## dataset_name: Use Registry Names, NOT Class Names

For local JSONL files, use `dataset_name: "text_sft_jsonl"` with `dataset_path` pointing to the file:
```yaml
data:
  train:
    datasets:
      - dataset_name: "text_sft_jsonl"
        dataset_path: "pubmedqa/train.jsonl"
```
- WRONG: `TextSftJsonLinesDataset`, `TextSftJsonlDataset` (these are Python class names, not registry names)
- For HuggingFace datasets: use the full HF ID (e.g. `yahma/alpaca-cleaned`) as dataset_name
- Use `get_docs(["dataset"])` to search for other registered dataset names

## LoRA/QLoRA: MUST Set `use_peft: True`

When using LoRA or QLoRA, you MUST set BOTH:
1. The `peft:` config block (lora_r, lora_alpha, lora_target_modules, etc.)
2. `training.use_peft: True`

Without `use_peft: True`, the `peft:` block is **silently ignored** and full fine-tuning
runs instead, using ~4x more VRAM and likely causing OOM on smaller GPUs.

## GPU VRAM Quick Reference

| Model Size | Full Fine-Tune | LoRA | QLoRA |
|-----------|---------------|------|-------|
| 3B | 24 GB | 12 GB | 8 GB |
| 7-8B | 60 GB | 20 GB | 14 GB |
| 13B | 100 GB | 32 GB | 20 GB |
| 70B | 400 GB+ | 80 GB | 48 GB |

Common cloud GPUs: A10G (22 GB), L4 (24 GB), A100 (40/80 GB), H100 (80 GB).
"""
