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

"""Oumi MCP - ML Engineer Training Workflow Prompt.

Provides a structured, XML-tagged prompt that guides agents through ML training
workflows using Oumi MCP tools. Designed for clear section parsing and role definition.
"""

GET_STARTED_CONTENT = """# Oumi MCP - ML Training Config Server

> **You called `get_started()` — good.** This is the required first step
> before using any other Oumi MCP tool. Read through the workflow below
> to understand tool ordering, path rules, and config usage guidelines.

## Available Tools

### Discovery
| Tool | Purpose | Example |
|------|---------|---------|
| `list_categories()` | See available models & config types | Start here |
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
| `down_cluster(cloud, cluster_name, confirm, user_confirmation)` | Delete cluster entirely — irreversible | `down_cluster("gcp", "sky-xxxx", confirm=True, user_confirmation="DOWN")` |

### ⚠️  How to use `get_config` correctly

Configs returned by `get_config(path)` are **reference
recipes** — they show you the correct YAML structure, field names, and sensible
defaults for a given model/task combination. **Do NOT copy them verbatim.**

Instead:
1. Study the structure and note which fields are relevant to the user's task.
2. Build a NEW config from scratch, adapting only the settings that apply
   (model name, dataset, training params, PEFT config, output dir).
3. Customize values for the user's specific hardware, data, and goals.
4. Omit sections that don't apply (e.g., drop `peft:` for full fine-tuning).

Copying a reference config wholesale leads to wrong datasets, wrong output
paths, unnecessary settings, and confused users.

## MCP Resources

### Guidance
| Resource | What it contains | Best time to use |
|----------|------------------|-----------------|
| `guidance://mle-workflow` | End-to-end MLE workflow, decision checkpoints | Full playbook, new project |
| `guidance://mle-train` | Training command usage, sizing heuristics | Planning or running training |
| `guidance://mle-synth` | Synthetic data generation flow | Generating synthetic datasets |
| `guidance://mle-analyze` | Dataset analysis, bias/quality checks | Before training, data audit |
| `guidance://mle-eval` | Evaluation strategies, benchmarks | Benchmarking or comparing runs |
| `guidance://mle-infer` | Inference best practices, latency tuning | Running inference or sanity checks |
| `guidance://cloud-launch` | Cloud job config anatomy, setup patterns | Before launching a cloud training run |
| `guidance://post-training` | Download weights, evaluate, teardown, merge LoRA | After cloud training succeeds |

### Jobs
| Resource | What it contains |
|----------|-----------------|
| `jobs://running` | Currently running jobs (JSON array) |
| `jobs://completed` | Recently finished jobs (JSON array) |
| `jobs://{job_id}/logs` | Full log output for a specific job |

## ⚠️  CRITICAL: Working Directory and Paths

**The MCP server runs in a DIFFERENT directory than the user's project.**
You MUST pass `client_cwd` (absolute path to the project root) to all path-sensitive tools.

**What `client_cwd` does:**
- Resolves relative config paths (e.g. `configs/train.yaml` → `/home/user/project/configs/train.yaml`)
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
  - ❌ BAD: `/Users/you/data/train.jsonl` — doesn't exist on the VM
  - ✅ GOOD: `data/train.jsonl` — resolves from synced `working_dir`

## ☁️  Cloud Job Workflow (REQUIRED — follow this order)

**When a user asks to run a cloud training job, ALWAYS follow these steps:**

```
CWD = "/home/user/my-project"  # user's project root — pass as client_cwd everywhere

Step 1: pre_flight_check("configs/train.yaml", client_cwd=CWD, cloud="gcp")
        # → check credentials, then use suggested_configs paths with get_config() for reference YAMLs
Step 2: Build a job config YAML using guidance://cloud-launch as reference
        Key fields: resources (cloud, accelerators), working_dir, setup, run, envs, file_mounts
Step 3: run_oumi_job(config_path="job.yaml", command="train", client_cwd=CWD, cloud="gcp")         # dry-run to verify
Step 4: run_oumi_job(config_path="job.yaml", command="train", client_cwd=CWD, cloud="gcp",
        dry_run=False, skip_preflight=True)
Step 5: get_job_status(job_id)                           # poll status
Step 6: get_job_logs(job_id, lines=200)                  # check logs
Step 7: [when done] stop_cluster("gcp", cluster_name)   # pause OR
         down_cluster("gcp", cluster_name, confirm=True, user_confirmation="DOWN")  # delete
```

**Key fields to customize in your cloud job YAML:**
- `resources.accelerators` — GPU type and count (e.g. `"A10G:1"`, `"A100:8"`)
- `working_dir` — use `.` (default); resolved to `client_cwd` at launch time
- `run` — your oumi command (path relative to `working_dir`)
- `envs` — API keys (WANDB_API_KEY, HF_TOKEN) that won't be forwarded automatically
- `file_mounts` — credential files auto-included; **add local dataset files** if not git-tracked
  - Example: `~/sky_workdir/data/train.jsonl: /Users/you/data/train.jsonl`

**Tip:** Read `guidance://cloud-launch` for detailed job config field explanations and common
setup patterns (dataset downloads, extra packages, storage mounts).

## Local Quickstart Workflow

1. **Discover models**: `list_categories()` -> see model_families
2. **Find recipes**: `search_configs(query=["llama3_1", "sft"])`
3. **Study reference**: `get_config("llama3_1/sft/8b_lora")` — read for structure and defaults, do NOT copy verbatim
4. **Build config**: Create a new config for the user's specific model, dataset, hardware, and goals — use the reference to inform field names and reasonable values
5. **Validate**: `validate_config("configs/train.yaml", "training", client_cwd="/home/user/project")`
6. **Preview**: `run_oumi_job(config_path="configs/train.yaml", command="train", client_cwd="/home/user/project")` -> dry-run (default)
7. **Execute**: `run_oumi_job(config_path="configs/train.yaml", command="train", client_cwd="/home/user/project", dry_run=False)`
8. **Check status**: `get_job_status("train_20260206_...")`
9. **Get logs**: `get_job_logs("train_20260206_...", lines=200)`

## Execution Pattern

```
CWD = "/home/user/project"  # always pass client_cwd

Step 1 (preview):  run_oumi_job(config_path=config, command="train", client_cwd=CWD)                           # dry_run=True
Step 2 (execute):  run_oumi_job(config_path=config, command="train", client_cwd=CWD, dry_run=False)
Step 3 (status):   get_job_status(job_id)
Step 4 (logs):     get_job_logs(job_id, lines=200)
Step 5 (cancel):   cancel_job(job_id)
Step 5b (force):   cancel_job(job_id, force=True)
```

## Cluster Lifecycle

After a cloud job finishes (or to manage costs), use cluster lifecycle tools:

```
Step 1: get_job_status(job_id)                        # → see "cluster" and "cloud" fields
Step 2a (pause):  stop_cluster("gcp", "sky-xxxx")     # keeps infra, lower cost, can restart
Step 2b (delete): down_cluster("gcp", "sky-xxxx", confirm=True, user_confirmation="DOWN")
```

- **`stop_cluster`**: Pauses compute. Storage costs may still apply. Cluster restartable.
- **`down_cluster`**: Permanently deletes everything. No more billing. Irreversible.
- ⚠️  Always `down_cluster` when training is fully done to avoid storage charges.

## Search Parameters

- **query**: Terms matched against config paths (AND logic, case-insensitive). Paths encode model family, size, task, and technique
- **content_match**: Substrings matched against YAML file content (AND logic, case-insensitive). Use for values not in the path, e.g. a dataset name or HuggingFace model ID
- **limit**: Maximum number of results to return (default 20)

## Config Key Settings

When customizing a config, these are the key fields to modify:
- `model.model_name`: HuggingFace model ID
- `data.train.datasets`: Dataset list (see dataset_name below)
- `training.output_dir`: Where to save checkpoints
- `training.learning_rate`: Start with recipe default
- `training.per_device_train_batch_size`: Adjust for your GPU memory

## ⚠️  dataset_name: Use Registry Names, NOT Class Names

For local JSONL files, use `dataset_name: "text_sft_jsonl"` with `dataset_path` pointing to the file:
```yaml
data:
  train:
    datasets:
      - dataset_name: "text_sft_jsonl"
        dataset_path: "pubmedqa/train.jsonl"
```
- WRONG: `TextSftJsonLinesDataset`, `TextSftJsonlDataset` — these are Python class names, not registry names
- For HuggingFace datasets: use the full HF ID (e.g. `yahma/alpaca-cleaned`) as dataset_name
- Use `get_docs(["dataset"])` to search for other registered dataset names

## ⚠️  LoRA/QLoRA: MUST set `use_peft: True`

When using LoRA or QLoRA, you MUST set BOTH:
1. The `peft:` config block (lora_r, lora_alpha, lora_target_modules, etc.)
2. `training.use_peft: True`

Without `use_peft: True`, the `peft:` block is **silently ignored** and full fine-tuning runs
instead — using ~4x more VRAM and likely OOMing on smaller GPUs (e.g. A10G with 8B model).

## GPU VRAM Quick Reference

| Model Size | Full Fine-Tune | LoRA | QLoRA |
|-----------|---------------|------|-------|
| 3B | 24 GB | 12 GB | 8 GB |
| 7-8B | 60 GB | 20 GB | 14 GB |
| 13B | 100 GB | 32 GB | 20 GB |
| 70B | 400 GB+ | 80 GB | 48 GB |

Common cloud GPUs: A10G (22 GB), L4 (24 GB), A100 (40/80 GB), H100 (80 GB).
"""

MLE_WORKFLOW_RESOURCE = """# Oumi ML Engineering Workflow Guidance

This resource provides an end-to-end ML engineering workflow for LLM training with Oumi.
Use it to reason like an MLE: start with requirements, validate data, pick recipes via
MCP tools, customize configs, validate, and evaluate before iterating.

## Principles
- Requirements first: Clarify success criteria before implementation
- Data-centric: Data quality determines model quality
- Iterative: Start small, validate assumptions, then scale
- Tool-first: Use MCP tools to find and adapt existing configs rather than building from scratch

## When to use each tool
0. get_started: ALWAYS call this FIRST — returns the full tool catalog, path rules, and workflow
1. search_configs: Find training recipes by model, task, or keyword
2. get_config: Study a reference config for structure and defaults — do NOT copy verbatim
3. validate_config: ALWAYS validate before training
4. run_oumi_job: Execute after validation passes

## Decision guidelines
- Model < 10B: Full fine-tuning viable
- Model 10B-30B: Use LoRA (r=16)
- Model > 30B: Use QLoRA (4-bit)

## Critical rules
- Never skip validation before run_oumi_job
- Always check GPU memory requirements

## MCP tools overview
- list_categories(): Discover available model families and config categories
- search_configs(query, content_match, limit): Find training/eval/inference recipes
- get_config(path): Retrieve YAML and metadata for a recipe
- validate_config(config, task_type, client_cwd): Validate before running

## Workflow
### Phase 1: Requirements Gathering
Establish clarity before designing anything.
- Task: What should the model do?
- Data: Format, size, quality status
- Compute: GPU type and count, time budget
- Success criteria: Metrics and targets
- Constraints: Model size limits, latency requirements, deployment target

### Phase 2: Recipe Selection
Call list_categories() then search_configs() to find matching recipes.

Task mapping (examples):
- Instruction following: sft (TRL_SFT)
- Domain adaptation: pretrain (OUMI)
- Preference alignment: dpo (TRL_DPO)
- Reward optimization: grpo (TRL_GRPO)

PEFT selection:
- Use LoRA/QLoRA when GPU memory is limited or rapid iteration is needed
- Use full fine-tuning when maximum quality and compute are available
- CRITICAL: When using LoRA/QLoRA, you MUST set `training.use_peft: True` in
  addition to providing the `peft:` config block. Without `use_peft: True`,
  the peft block is silently ignored and full fine-tuning runs instead — causing OOM.

### Phase 3: Data Validation
- Consistent schema across examples
- Manual review of 50+ samples
- Duplicates removed
- No data leakage between train/val/test splits
- Token lengths within model context
- Labels/outputs are correct

Red flags (fix before training):
- Too few examples (less than 500 for narrow tasks or 5000 for general capability)
- More than 5% duplicates
- Significant class imbalance (greater than 10:1)
- P95 token length exceeds max context
- More than 10% low-quality examples

### Phase 4: Config Customization
1. get_config("path") — use as a REFERENCE only, not a template to copy
2. Build a new config from scratch, adapting only the relevant settings for the user's
   specific model, dataset, hardware, and goals
3. Save the new config
4. validate_config("config.yaml", "training", client_cwd="/path/to/project")

Key settings to customize:
- model.model_name
- data.train.datasets (see dataset_name reference below)
- training.output_dir
- training.learning_rate
- training.per_device_train_batch_size
- training.gradient_accumulation_steps
- training.use_peft: True (REQUIRED when using LoRA/QLoRA — peft block alone is NOT enough)

dataset_name reference — use these EXACT registry names (NOT Python class names):
- `text_sft_jsonl` — Local JSONL file with SFT conversations (most common for custom data)
  Requires: `dataset_path: "path/to/train.jsonl"` pointing to the JSONL file
- For HuggingFace datasets: use the full HF ID as dataset_name (e.g. `yahma/alpaca-cleaned`)
- WRONG: `TextSftJsonLinesDataset`, `TextSftJsonlDataset` — these are class names, not registry names
- Use `get_docs(["dataset"])` to search for other registered dataset names

### Phase 5: Evaluation Strategy
- During training: monitor train/val loss
- Post-training benchmarks: lm-eval-harness or similar
- Task-specific metrics on held-out test set
- Qualitative review of 50-100 samples

Success criteria:
- Val loss near or below train loss
- Primary metric exceeds baseline
- No regression on general capabilities
- 90%+ manual review quality

### Phase 6: Iteration and Troubleshooting
Common issues:
- Loss NaN/Inf: lower LR, check data
- OOM: reduce batch size, increase grad accum
- Slow training: data loading bottleneck
- Tokenizer mismatch: verify tokenizer/model alignment
- Overfitting: reduce epochs, add regularization, add data

Pivot guidance:
- Stop when success criteria met or diminishing returns
- Continue with a clear hypothesis for improvement
- Pivot if data quality is the bottleneck after multiple iterations

## Quick reference
VRAM rough guidance:
- 3B: 24GB full / 12GB LoRA / 8GB QLoRA
- 7B: 60GB full / 20GB LoRA / 14GB QLoRA
- 13B: 100GB full / 32GB LoRA / 20GB QLoRA
- 70B: 400GB+ full / 80GB LoRA / 48GB QLoRA
"""

TRAIN_COMMAND_RESOURCE = """<resource>
<title>MLE Train Guide (Oumi)</title>
<purpose>Plan and run training to change model behavior or adapt to domain data.</purpose>

<usage>
<cli>oumi train -c path/to/train.yaml</cli>
<override>oumi train -c path/to/train.yaml --training.max_steps 20</override>
</usage>

<workflow>
<step name="intent">
<what>Clarify the user's objective, success criteria, and constraints.</what>
<why>Task type, data requirements, and model choices flow directly from intent.</why>
<oumi>Map intent to `task` (sft/dpo/grpo/pretrain) and find recipes with `search_configs`.</oumi>
</step>

<step name="data">
<what>Confirm dataset format, size, and quality.</what>
<why>Quality issues propagate into model behavior.</why>
<oumi>Set `data.train.datasets` and run `oumi analyze` before training.</oumi>
</step>

<step name="model">
<what>Select a base model and parameter size.</what>
<why>Model size drives VRAM needs, runtime, and expected capability.</why>
<oumi>Set `model.model_name` and choose LoRA/QLoRA vs full fine-tuning.</oumi>
</step>

<step name="compute">
<what>Match batch size and accumulation to GPU memory.</what>
<why>Effective batch size controls stability and throughput.</why>
<oumi>Tune `training.per_device_train_batch_size` and `training.gradient_accumulation_steps`.</oumi>
</step>

<step name="run">
<what>Validate and execute the config.</what>
<why>Schema errors waste GPU time.</why>
<oumi>Run `validate_config(..., \"training\")` then `oumi train`.</oumi>
</step>
</workflow>

<key_config>
<field name="model.model_name">HF model ID or local path.</field>
<field name="data.train.datasets">Dataset list. Use registry names for dataset_name (e.g. "text_sft_jsonl" for local JSONL, or a HuggingFace ID). NOT Python class names.</field>
<field name="training.learning_rate">Start from recipe; tune if unstable or slow.</field>
<field name="training.max_steps">Limit steps for quick iteration.</field>
<field name="training.output_dir">Per-run output directory for checkpoints/logs.</field>
<field name="training.per_device_train_batch_size">Tune to avoid OOM.</field>
<field name="training.gradient_accumulation_steps">Increase if batch size is small.</field>
<field name="training.use_peft">MUST be True when using LoRA/QLoRA. Without this, the peft block is silently ignored and full fine-tuning runs — causing OOM on smaller GPUs.</field>
</key_config>

<common_overrides>
<override>--training.max_steps</override>
<override>--training.learning_rate</override>
<override>--training.output_dir</override>
<override>--data.train.datasets</override>
</common_overrides>

<outputs>
<item>Checkpoints and logs under `training.output_dir`.</item>
<item>Trainer state and metrics for monitoring convergence.</item>
</outputs>
</resource>
"""

SYNTH_COMMAND_RESOURCE = """<resource>
<title>MLE Synthesis Guide (Oumi)</title>
<purpose>Generate synthetic training data when real data is scarce, noisy, or needs coverage.</purpose>

<usage>
<cli>oumi synth -c path/to/synth.yaml</cli>
<override>oumi synth -c path/to/synth.yaml --num_samples 1000</override>
</usage>

<workflow>
<step name="intent">
<what>Define what the synthetic data should teach the model.</what>
<why>Generation targets must align with downstream training goals.</why>
<oumi>Encode intent in `strategy_params.generated_attributes` prompt templates.</oumi>
</step>

<step name="attributes">
<what>Define which attributes should vary across samples.</what>
<why>Attributes control diversity and coverage.</why>
<oumi>Specify `sampled_attributes` (topic, difficulty, persona, style).</oumi>
</step>

<step name="templates">
<what>Describe how to generate outputs from attributes.</what>
<why>Templates enforce schema consistency and label quality.</why>
<oumi>Use `generated_attributes` with `instruction_messages` prompts.</oumi>
</step>

<step name="scale">
<what>Run a small sample and review before scaling up.</what>
<why>Early review prevents low-quality data at scale.</why>
<oumi>Start with low `num_samples`, inspect JSONL, then increase.</oumi>
</step>
</workflow>

<key_config>
<field name="strategy">Synthesis strategy (typically `GENERAL`).</field>
<field name="num_samples">Total samples to generate.</field>
<field name="output_path">JSONL output file path.</field>
<field name="strategy_params.sampled_attributes">Attribute variations for diversity.</field>
<field name="strategy_params.generated_attributes">Prompt templates and expected outputs.</field>
<field name="inference_config">LLM model + generation parameters.</field>
</key_config>

<outputs>
<item>JSONL dataset at `output_path` for use in training.</item>
</outputs>
</resource>
"""

ANALYZE_COMMAND_RESOURCE = """<resource>
<title>MLE Dataset Analysis Guide (Oumi)</title>
<purpose>Profile datasets, compute metrics, and flag outliers before training.</purpose>

<usage>
<cli>oumi analyze --config path/to/analyze.yaml</cli>
</usage>

<workflow>
<step name="intent">
<what>Validate data quality and suitability for training.</what>
<why>Bad data causes instability and poor generalization.</why>
<oumi>Run analyze before training and after synthesis.</oumi>
</step>

<step name="coverage">
<what>Check distribution coverage and length ranges.</what>
<why>Outliers and long examples cause truncation and bias.</why>
<oumi>Use length/token analyzers and inspect percentiles.</oumi>
</step>

<step name="quality">
<what>Find duplicates, empty samples, and label issues.</what>
<why>Duplicates and noise reduce effective data and increase overfitting.</why>
<oumi>Export results and filter problematic rows.</oumi>
</step>
</workflow>

<key_config>
<field name="dataset_path|dataset_name">Input dataset.</field>
<field name="sample_count">Limit for quick scans.</field>
<field name="output_path">Directory for analysis outputs.</field>
<field name="format">csv/json/parquet export format.</field>
<field name="analyzers">Built-in or custom analyzers.</field>
</key_config>

<outputs>
<item>`analysis_summary.json` plus message/conversation analysis files.</item>
</outputs>
</resource>
"""

EVAL_COMMAND_RESOURCE = """<resource>
<title>MLE Evaluation Guide (Oumi)</title>
<purpose>Benchmark model performance on standard or custom evaluation tasks.</purpose>

<usage>
<cli>oumi evaluate -c path/to/eval.yaml</cli>
<override>oumi evaluate -c path/to/eval.yaml --model.model_name my/model</override>
</usage>

<workflow>
<step name="intent">
<what>Define which behaviors to measure and what baseline to beat.</what>
<why>Evaluation tasks must match user goals.</why>
<oumi>Configure `tasks` with the correct backend and task names.</oumi>
</step>

<step name="comparability">
<what>Keep evaluation settings consistent across runs.</what>
<why>Different generation settings invalidate comparisons.</why>
<oumi>Lock `generation` parameters and use a per-run `output_dir`.</oumi>
</step>

<step name="scale">
<what>Plan for multi-GPU or sharded eval on larger models.</what>
<why>Large models may not fit on one GPU.</why>
<oumi>Set `model.shard_for_eval` and use distributed launch if needed.</oumi>
</step>
</workflow>

<key_config>
<field name="model.model_name">Model to evaluate.</field>
<field name="tasks">Benchmarks and backends (lm_harness, alpaca_eval, custom).</field>
<field name="generation">Batch size and decoding parameters.</field>
<field name="output_dir">Where results and metadata are stored.</field>
</key_config>

<outputs>
<item>Metrics and run metadata in `output_dir` (e.g., task_result.json).</item>
</outputs>

<caveats>
<item>lm_harness evaluation tasks may have version-specific compatibility with Oumi.
If evaluation fails with dtype or model constructor errors, verify the installed oumi
version matches task expectations, or fall back to a direct evaluation script.</item>
<item>Config structure is consistent across model versions within a family (e.g. Qwen2.5
and Qwen3 use the same config shape). If search_configs() doesn't return your exact model
version, use a config from the same family as a template.</item>
</caveats>
</resource>
"""

CLOUD_LAUNCH_RESOURCE = """# Cloud Job Launch Guide

## Why You Need a Job Config for Cloud Runs

Cloud jobs run on remote VMs managed by SkyPilot. Unlike local runs where `oumi train -c config.yaml`
is enough, cloud runs need additional information:
- **setup**: How to install Oumi and dependencies on the fresh VM
- **run**: The exact shell command to execute
- **working_dir**: Which local files to sync to the remote VM
- **file_mounts**: Credential files (HF token, .netrc) to copy
- **storage_mounts**: Persistent cloud storage for outputs (important for spot instances)
- **envs**: Environment variables (API keys, project names)
- **resources**: Cloud provider, GPU type, disk size, spot/on-demand

## Version Compatibility

The MCP tools document Oumi 0.7 APIs. If the cloud VM installs a different version
(e.g. `pip install oumi[gpu]` installs 0.1.x), some field names may differ:
- `evaluation_backend` (0.7) → `evaluation_platform` (0.1.x)

**To avoid mismatches:** pin the version in your setup script:
```bash
pip install 'oumi[gpu]>=0.7'
```
Or use `get_docs()` to check the installed version's API.

## How Path Resolution Works

When you call `run_oumi_job(config_path, command, client_cwd)`:

1. **`client_cwd`** = absolute path to the user's project root (e.g. `/home/user/my-project`)
2. **`config_path`** is resolved relative to `client_cwd` (e.g. `configs/train.yaml` → `/home/user/my-project/configs/train.yaml`)
3. For cloud jobs, `client_cwd` becomes the **`working_dir`** in the job config
4. SkyPilot rsyncs `working_dir` to `~/sky_workdir` on the remote VM
5. The `run` command `cd`s into `~/sky_workdir`, so **repo-relative paths** resolve correctly
6. Only **git-tracked files** are synced (when `.gitignore` is present); untracked files are silently skipped

**Always pass `client_cwd`** — without it, the MCP server's own working directory is used (which is NOT your project root), and files won't be found.

## GPU Sizing for Cloud Jobs

| Model Size | Full Fine-Tune | LoRA | QLoRA | Recommended GPU |
|-----------|---------------|------|-------|----------------|
| 3B | 24 GB | 12 GB | 8 GB | A10G (22 GB) or L4 (24 GB) |
| 7-8B | 60 GB | 20 GB | 14 GB | LoRA: A10G; FFT: A100 (40 GB) |
| 13B | 100 GB | 32 GB | 20 GB | LoRA: A100 (40 GB); FFT: A100:2+ |
| 70B | 400 GB+ | 80 GB | 48 GB | A100:8 or H100:4+ |

**CRITICAL: LoRA requires `training.use_peft: True`** in your training config.
The `peft:` config block alone is NOT enough — without `use_peft: True`, Oumi silently
runs full fine-tuning, which uses ~4x more VRAM and will OOM on smaller GPUs.

## Key Fields

### `setup` (shell script)
Runs once when the VM is provisioned. Install Oumi, download datasets, install extras:
```bash
set -e
pip install uv && uv pip install --system 'oumi[gpu]'
# Download datasets
huggingface-cli download <dataset-id> --repo-type dataset --local-dir ./data
# Install additional packages
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
config non-portable and may not exist on another machine.

The training config file is placed inside this directory on the VM at
`~/sky_workdir/`.

**Important:** Only git-tracked files are synced when a `.gitignore` is present.
If a file isn't in git, SkyPilot will silently skip it. Use `file_mounts` for
untracked files (see below).

### Path conventions for cloud jobs

Use **repo-relative paths** for project files (data, configs, output). These
resolve from `working_dir` on the remote VM after sync.

| Path type | Convention | Example |
|-----------|-----------|---------|
| Project files (data, configs) | Repo-relative | `data/pubmed_qa/train.jsonl` |
| Config references | Repo-relative | `configs/train.yaml` |
| Training output | Repo-relative | `output/...` |
| Remote-only output | Remote absolute | `/home/ubuntu/output/...` |
| Local machine paths | **NEVER** | ~~`/Users/you/project/data/...`~~ |

**NEVER** use local machine paths like `/Users/.../` or `/home/yourname/...`
in cloud configs — they don't exist on the remote VM. The pre-flight check
will block these automatically.

### How dataset files reach the remote VM

There are 3 ways a dataset file can be available on the VM:

1. **`working_dir` sync** — Files inside `working_dir` that are **git-tracked** are
   automatically rsynced to `~/sky_workdir` on the VM. Reference them as repo-relative
   paths (e.g. `./data/train.jsonl`). Untracked files are silently skipped.

2. **`file_mounts`** — Explicitly copy local files to the VM. Use this for datasets
   that are NOT git-tracked or are outside your project directory:
   ```yaml
   file_mounts:
     ~/sky_workdir/data/train.jsonl: /Users/you/datasets/train.jsonl
   ```
   Then reference as `./data/train.jsonl` in your training config.

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
  # Credentials (auto-detected, but can override)
  ~/.cache/huggingface/token: ~/.cache/huggingface/token
  ~/.netrc: ~/.netrc
  # Local datasets → remote VM paths
  ~/sky_workdir/data/train.jsonl: /Users/you/datasets/train.jsonl
  ~/sky_workdir/data/val.jsonl: /Users/you/datasets/val.jsonl
```

Then reference the data in your training config as `./data/train.jsonl`
(relative to `working_dir` = `~/sky_workdir`).

### `envs`
Environment variables set on the remote VM. Local env vars are NOT forwarded:
```yaml
envs:
  WANDB_API_KEY: "your-key"
  WANDB_PROJECT: "my-project"
  HF_TOKEN: "hf_..."
```

## Example Complete Job Config

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
  WANDB_API_KEY: "..."
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
2. **Preview**: `run_oumi_job(config_path="job.yaml", command="train", client_cwd=CWD, cloud="gcp")` (dry_run=True)
3. **Execute**: `run_oumi_job(config_path="job.yaml", command="train", client_cwd=CWD, cloud="gcp", dry_run=False, skip_preflight=True)`

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
| `working_dir` pointed to wrong directory (no `client_cwd`) | `FileNotFoundError: configs/train.yaml` | Always pass `client_cwd` to `run_oumi_job` |
| File not git-tracked | File exists locally but missing on VM | `git add <file>` and commit, or use `file_mounts` |
| Local absolute path in cloud config | `/Users/you/data/...` doesn't exist on VM | Use repo-relative paths (e.g. `data/...`) |
| File outside `working_dir` | Only `working_dir` contents are synced | Use `file_mounts` to copy files from other locations |

**Diagnosis steps:**
1. Check the dry-run output — it shows the resolved `working_dir` and generated job config
2. Run `git status` in your project — untracked files won't sync
3. Verify paths in your training config are repo-relative, not absolute local paths

## Important: Ephemeral Storage

Training outputs on the cluster's local disk are **not preserved** across cluster stops
or recreations. If the cluster is stopped, restarted, or torn down, all local files
(checkpoints, adapters, logs) are lost.

**Before stopping or deleting a cluster:**
1. Use `sky rsync-down <cluster> ~/sky_workdir/<output_dir> ./local_output/` to download artifacts
2. Or configure your training config to save checkpoints to a cloud bucket (S3/GCS) via `storage_mounts`

## Important: Existing Clusters and File Sync

When submitting a job to an **existing** cluster, SkyPilot uses `sky exec` instead of
`sky launch`. This means:
- Local file changes are **NOT re-synced** to the VM
- The VM still has the files from the original `sky launch`

**If you edited files locally and need them on the VM:**
- Use `sky launch` with the same `cluster_name` to force a full re-sync
- Or use explicit `file_mounts` for files that change between submissions
"""

POST_TRAINING_RESOURCE = """# Post-Training Guide — What To Do After Cloud Training Succeeds

## Priority Order

When training completes on a cloud VM, act quickly — the cluster is billing you.

1. **Download model weights** (do this first)
2. **Run evaluation on the cluster** (optional, while cluster is still up)
3. **Tear down the cluster**
4. **Merge LoRA adapter locally** (if applicable)
5. **Push to HuggingFace Hub** (optional)

---

## Step 1: Download Model Weights

The MCP has no file-transfer tool. Use SkyPilot CLI directly in the user's terminal:

```bash
# Download the output directory from the remote VM
sky rsync-down <cluster-name> ~/sky_workdir/<output_dir> ./output/

# Example:
sky rsync-down sky-abc-user ~/sky_workdir/output/llama8b-sft ./output/
```

**LoRA adapters are small** (~5-50 MB depending on rank and target modules).
Full fine-tuned models are the size of the base model (e.g. ~16 GB for 8B in bf16).

The output directory typically contains:
- `adapter_model.safetensors` + `adapter_config.json` (LoRA)
- OR `model-*.safetensors` + config files (full fine-tune)
- `trainer_state.json`, `training_args.bin` (training metadata)

## Step 2: Run Evaluation (Optional, On-Cluster)

While the cluster is still running, evaluate the fine-tuned model to get benchmark
scores. Use `run_oumi_job` with `command: "evaluate"`:

- Point `model.model_name` at the output path on the remote VM
- Use the same cluster (it already has the model weights)
- Common benchmarks: MMLU, HellaSwag, ARC, or task-specific evals

This avoids downloading the full model just to evaluate it.

## Step 3: Tear Down the Cluster

Once weights are downloaded, stop billing immediately:

| Action | MCP Tool | SkyPilot CLI | Effect |
|--------|----------|-------------|--------|
| **Stop** (pause) | `stop_cluster(cloud, cluster_name)` | `sky stop <cluster>` | Stops compute billing, keeps disk. Can restart later. |
| **Down** (delete) | `down_cluster(cloud, cluster_name, confirm, user_confirmation)` | `sky down <cluster>` | Deletes everything — VM, disk, all files. Irreversible. |

**Use `stop`** if you might want to run more jobs on the same cluster.
**Use `down`** if you're done — this is cheaper (no disk storage fees).

## Step 4: Merge LoRA Adapter (Local)

If you trained with LoRA/QLoRA, merge the adapter into the base model for easier deployment:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(base, "./output/llama8b-sft")
merged = model.merge_and_unload()
merged.save_pretrained("./output/llama8b-sft-merged")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.save_pretrained("./output/llama8b-sft-merged")
```

Or use Oumi CLI:
```bash
oumi merge -c merge_config.yaml
```

## Step 5: Push to HuggingFace Hub (Optional)

```bash
huggingface-cli upload <your-org>/<model-name> ./output/llama8b-sft-merged
# Or just the adapter:
huggingface-cli upload <your-org>/<model-name>-lora ./output/llama8b-sft
```

## Quick Reference: What the MCP Can and Cannot Do Post-Training

| Task | MCP Support | How |
|------|------------|-----|
| Check job status | Yes | `get_job_status(job_id)` |
| View training logs | Yes | `get_job_logs(job_id)` |
| Run evaluation | Yes | `run_oumi_job(config_path=..., command="evaluate", client_cwd=...)` |
| Run inference | Yes | `run_oumi_job(config_path=..., command="infer", client_cwd=...)` |
| Stop/delete cluster | Yes | `stop_cluster(cloud, cluster_name)` / `down_cluster(cloud, cluster_name, confirm, user_confirmation)` |
| Download files | No | Use `sky rsync-down` in terminal |
| Merge LoRA adapter | No | Use `peft` or `oumi merge` locally |
| Push to HF Hub | No | Use `huggingface-cli upload` in terminal |
"""

INFER_COMMAND_RESOURCE = """<resource>
<title>MLE Inference Guide (Oumi)</title>
<purpose>Run inference for quick checks or batch generation.</purpose>

<usage>
<cli>oumi infer -c path/to/infer.yaml</cli>
<interactive>oumi infer -c path/to/infer.yaml --interactive</interactive>
</usage>

<workflow>
<step name="intent">
<what>Decide between quick checks and structured batch outputs.</what>
<why>Interactive mode is best for debugging; batch mode is for datasets and evaluation.</why>
<oumi>Use `--interactive` or set `input_path`/`output_path` for batch JSONL.</oumi>
</step>

<step name="alignment">
<what>Ensure prompts and templates match the model’s chat format.</what>
<why>Mismatched templates degrade responses.</why>
<oumi>Confirm tokenizer and chat template for the selected `model.model_name`.</oumi>
</step>

<step name="decoding">
<what>Set generation parameters appropriate for your task.</what>
<why>Temperature and max tokens change output quality and length.</why>
<oumi>Configure `generation.max_new_tokens` and `generation.temperature`.</oumi>
</step>
</workflow>

<key_config>
<field name="model.model_name">Model used for inference.</field>
<field name="generation">max_new_tokens, temperature, batch_size.</field>
<field name="input_path">Batch JSONL input (optional).</field>
<field name="output_path">Batch JSONL output (optional).</field>
</key_config>

<outputs>
<item>Console responses (interactive) or JSONL outputs (batch).</item>
</outputs>

<output_format>
<description>Each line in predictions.jsonl contains the full conversation history
with the model's response appended as an assistant turn:</description>
<example>{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}</example>
<note>The output is NOT just the raw prediction — it includes the full message history.
Parse the last assistant message to extract the model's response.</note>
</output_format>

<caveats>
<item>Config structure is consistent across model versions within a family. If
search_configs() doesn't return your exact model version (e.g. Qwen2.5), use a config
from the same family (e.g. Qwen3) as a template — the structure is identical.</item>
</caveats>
</resource>
"""
