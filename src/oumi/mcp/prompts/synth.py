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

"""Synthetic data generation guidance resource.

Covers planning and running `oumi synth` to generate training data when real
data is scarce, noisy, or needs broader coverage.
Exposed as `guidance://mle-synth`.
"""

SYNTH_COMMAND_RESOURCE = """# Synthetic Data Generation Guide

Generate synthetic training data when real data is scarce, noisy, or needs broader coverage.

## Usage

```bash
oumi synth -c path/to/synth.yaml
oumi synth -c path/to/synth.yaml --num_samples 1000   # CLI override
```

## Workflow

### 1. Define Intent
- Specify what the synthetic data should teach the model.
- Generation targets must align with downstream training goals.
- Encode intent in `strategy_params.generated_attributes` prompt templates.

### 2. Design Attributes
- Define which attributes should vary across samples (topic, difficulty, persona, style).
- Attributes control diversity and coverage of the generated dataset.
- Specify via `sampled_attributes` in the config.

### 3. Build Templates
- Define how to generate outputs from attribute combinations.
- Templates enforce schema consistency and label quality.
- Use `generated_attributes` with `instruction_messages` prompts.

### 4. Scale Incrementally
- Start with a small `num_samples`, inspect the JSONL output, then increase.
- Early review prevents low-quality data at scale.

## Key Config Fields

| Field | Purpose |
|-------|---------|
| `strategy` | Synthesis strategy (typically `GENERAL`) |
| `num_samples` | Total samples to generate |
| `output_path` | JSONL output file path |
| `strategy_params.sampled_attributes` | Attribute variations for diversity |
| `strategy_params.generated_attributes` | Prompt templates and expected outputs |
| `inference_config` | LLM model and generation parameters |

## Outputs

- JSONL dataset at `output_path`, ready for use in training.
"""
