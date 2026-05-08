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

"""Training command guidance resource.

Covers planning and running `oumi train`, including task mapping, data prep,
model selection, compute sizing, and key config fields.
Exposed as `guidance://mle-train`.
"""

TRAIN_COMMAND_RESOURCE = """# Training Guide

Plan and run training to change model behavior or adapt to domain data.

## Usage

```bash
oumi train -c path/to/train.yaml
oumi train -c path/to/train.yaml --training.max_steps 20   # CLI override
```

## Workflow

### 1. Clarify Intent
- Map the user's objective to a task type: `sft`, `dpo`, `grpo`, or `pretrain`.
- Define success criteria and constraints up front.
- Find matching recipes with `search_configs`.

### 2. Prepare Data
- Confirm dataset format, size, and quality.
- Set `data.train.datasets` and run `oumi analyze` before training.
- Quality issues propagate directly into model behavior.

### 3. Select Model
- Choose a base model and parameter size.
- Model size drives VRAM needs, runtime, and expected capability.
- Set `model.model_name` and decide between LoRA/QLoRA vs full fine-tuning.

### 4. Size Compute
- Match batch size and gradient accumulation to GPU memory.
- Effective batch size = `per_device_train_batch_size` x `gradient_accumulation_steps` x GPU count.
- Tune `training.per_device_train_batch_size` and `training.gradient_accumulation_steps`.

### 5. Validate and Run
- Always run `validate_config(..., "training")` before training.
- Schema errors waste GPU time — catch them early.
- Execute with `oumi train -c config.yaml`.

## Key Config Fields

| Field | Purpose |
|-------|---------|
| `model.model_name` | HuggingFace model ID or local path |
| `data.train.datasets` | Dataset list. Use registry names (e.g. `text_sft_jsonl` for local JSONL, or a HuggingFace ID). NOT Python class names. |
| `training.learning_rate` | Start from recipe default; tune if unstable or slow |
| `training.max_steps` | Limit steps for quick iteration |
| `training.output_dir` | Per-run output directory for checkpoints and logs |
| `training.per_device_train_batch_size` | Tune to avoid OOM |
| `training.gradient_accumulation_steps` | Increase if batch size is small |
| `training.use_peft` | **MUST be True** when using LoRA/QLoRA. Without this, the `peft:` block is silently ignored and full fine-tuning runs instead. |

## Common CLI Overrides

- `--training.max_steps`
- `--training.learning_rate`
- `--training.output_dir`
- `--data.train.datasets`

## Outputs

- Checkpoints and logs under `training.output_dir`.
- Trainer state and metrics for monitoring convergence.
"""
