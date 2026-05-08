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

"""End-to-end ML engineering workflow guidance resource.

Provides the full MLE playbook: requirements gathering, recipe selection,
data validation, config customization, evaluation, and iteration.
Exposed as `guidance://mle-workflow`.
"""

MLE_WORKFLOW_RESOURCE = """# Oumi ML Engineering Workflow Guidance

End-to-end ML engineering workflow for LLM training with Oumi. Start with
requirements, validate data, pick recipes via MCP tools, customize configs,
validate, and evaluate before iterating.

## Principles
- Requirements first: Clarify success criteria before implementation
- Data-centric: Data quality determines model quality
- Iterative: Start small, validate assumptions, then scale
- Tool-first: Use MCP tools to find and adapt existing configs rather than building from scratch

## Tool Ordering
0. `get_started()`: ALWAYS call first — returns the full tool catalog, path rules, and workflow
1. `search_configs()`: Find training recipes by model, task, or keyword
2. `get_config()`: Study a reference config for structure and defaults — do NOT copy verbatim
3. `validate_config()`: ALWAYS validate before training
4. `run_oumi_job()`: Execute after validation passes

## Decision Guidelines
- Model < 10B params: Full fine-tuning is viable
- Model 10B-30B params: Use LoRA (r=16)
- Model > 30B params: Use QLoRA (4-bit)

## Critical Rules
- Never skip validation before `run_oumi_job`
- Always check GPU memory requirements (see VRAM table in `get_started()` output)
- When using LoRA/QLoRA, MUST set `training.use_peft: True` in addition to the `peft:` block

## Workflow

### Phase 1: Requirements Gathering
Establish clarity before designing anything.
- Task: What should the model do?
- Data: Format, size, quality status
- Compute: GPU type and count, time budget
- Success criteria: Metrics and targets
- Constraints: Model size limits, latency requirements, deployment target

### Phase 2: Recipe Selection
Call `list_categories()` then `search_configs()` to find matching recipes.

Task mapping:
- Instruction following: sft (TRL_SFT)
- Domain adaptation: pretrain (OUMI)
- Preference alignment: dpo (TRL_DPO)
- Reward optimization: grpo (TRL_GRPO)

PEFT selection:
- Use LoRA/QLoRA when GPU memory is limited or rapid iteration is needed
- Use full fine-tuning when maximum quality is required and compute is available

### Phase 3: Data Validation
- Consistent schema across examples
- Manual review of 50+ samples
- Duplicates removed
- No data leakage between train/val/test splits
- Token lengths within model context window
- Labels and outputs are correct

Red flags (fix before training):
- Too few examples (< 500 for narrow tasks, < 5000 for general capability)
- More than 5% duplicates
- Significant class imbalance (> 10:1)
- P95 token length exceeds max context
- More than 10% low-quality examples

### Phase 4: Config Customization
1. `get_config("path")` — use as a REFERENCE only, not a template to copy
2. Build a new config from scratch, adapting only the relevant settings
3. Save the new config
4. `validate_config("config.yaml", "training", client_cwd="/path/to/project")`

Key settings to customize:
- `model.model_name`
- `data.train.datasets` (use registry names like `text_sft_jsonl`, NOT Python class names)
- `training.output_dir`
- `training.learning_rate`
- `training.per_device_train_batch_size`
- `training.gradient_accumulation_steps`
- `training.use_peft: True` (required for LoRA/QLoRA)

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
- Loss NaN/Inf: lower learning rate, check data quality
- OOM: reduce batch size, increase gradient accumulation
- Slow training: data loading bottleneck
- Tokenizer mismatch: verify tokenizer/model alignment
- Overfitting: reduce epochs, add regularization, increase data

When to pivot:
- Stop when success criteria are met or returns are diminishing
- Continue with a clear hypothesis for improvement
- Pivot if data quality is the bottleneck after multiple iterations
"""
