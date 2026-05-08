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

"""Inference guidance resource.

Covers running `oumi infer` for interactive sanity checks and batch generation,
including prompt alignment and output format details.
Exposed as `guidance://mle-infer`.
"""

INFER_COMMAND_RESOURCE = """# Inference Guide

Run inference for quick sanity checks or batch generation.

## Usage

```bash
oumi infer -c path/to/infer.yaml
oumi infer -c path/to/infer.yaml --interactive   # interactive mode
```

## Workflow

### 1. Choose Mode
- **Interactive mode** (`--interactive`): Best for quick debugging and spot-checking outputs.
- **Batch mode** (`input_path` / `output_path`): Best for structured dataset generation and evaluation.

### 2. Align Prompts
- Ensure prompts and templates match the model's expected chat format.
- Mismatched chat templates degrade response quality.
- Confirm tokenizer and chat template for the selected `model.model_name`.

### 3. Set Generation Parameters
- Configure `generation.max_new_tokens` and `generation.temperature` for the task.
- Lower temperature for factual tasks; higher for creative generation.

## Key Config Fields

| Field | Purpose |
|-------|---------|
| `model.model_name` | Model used for inference |
| `generation` | `max_new_tokens`, `temperature`, `batch_size` |
| `input_path` | Batch JSONL input (optional) |
| `output_path` | Batch JSONL output (optional) |

## Output Format

Each line in `predictions.jsonl` contains the full conversation history with the model's response appended as an assistant turn:

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}
```

The output is **not** just the raw prediction — it includes the full message history. Parse the last assistant message to extract the model's response.

## Outputs

- Console responses (interactive mode) or JSONL file (batch mode).

## Caveats

- **Cross-version configs:** Config structure is consistent across model versions within a family. If `search_configs()` does not return your exact model version (e.g. Qwen2.5), use a config from the same family (e.g. Qwen3) as a template — the structure is identical.
"""
