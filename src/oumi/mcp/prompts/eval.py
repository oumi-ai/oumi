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

"""Evaluation guidance resource.

Covers running `oumi evaluate` to benchmark model performance on standard
or custom evaluation tasks.
Exposed as `guidance://mle-eval`.
"""

EVAL_COMMAND_RESOURCE = """# Evaluation Guide

Benchmark model performance on standard or custom evaluation tasks.

## Usage

```bash
oumi evaluate -c path/to/eval.yaml
oumi evaluate -c path/to/eval.yaml --model.model_name my/model   # CLI override
```

## Workflow

### 1. Define Evaluation Goals
- Specify which behaviors to measure and what baseline to beat.
- Evaluation tasks must match the user's training goals.
- Configure `tasks` with the correct backend and task names.

### 2. Ensure Comparability
- Keep evaluation settings consistent across runs.
- Different generation settings invalidate comparisons.
- Lock `generation` parameters and use a per-run `output_dir`.

### 3. Plan for Scale
- Large models may not fit on a single GPU for evaluation.
- Set `model.shard_for_eval` and use distributed launch if needed.

## Key Config Fields

| Field | Purpose |
|-------|---------|
| `model.model_name` | Model to evaluate |
| `tasks` | Benchmarks and backends (lm_harness, alpaca_eval, custom) |
| `generation` | Batch size and decoding parameters |
| `output_dir` | Where results and metadata are stored |

## Outputs

- Metrics and run metadata in `output_dir` (e.g., `task_result.json`).

## Caveats

- **lm_harness compatibility:** Evaluation tasks may have version-specific compatibility with Oumi. If evaluation fails with dtype or model constructor errors, verify the installed Oumi version matches task expectations, or fall back to a direct evaluation script.
- **Cross-version configs:** Config structure is consistent across model versions within a family (e.g. Qwen2.5 and Qwen3 use the same config shape). If `search_configs()` does not return your exact model version, use a config from the same family as a template.
"""
