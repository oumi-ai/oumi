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

"""Dataset analysis guidance resource.

Covers running `oumi analyze` to profile datasets, compute quality metrics,
and flag outliers before training.
Exposed as `guidance://mle-analyze`.
"""

ANALYZE_COMMAND_RESOURCE = """# Dataset Analysis Guide

Profile datasets, compute metrics, and flag outliers before training.

## Usage

```bash
oumi analyze --config path/to/analyze.yaml
```

## Workflow

### 1. Validate Data Quality
- Run analysis before training and after synthetic data generation.
- Poor data causes training instability and weak generalization.

### 2. Check Distribution Coverage
- Inspect length distributions and token count percentiles.
- Outliers and overly long examples cause truncation and bias.
- Use length/token analyzers to identify problematic ranges.

### 3. Find Quality Issues
- Detect duplicates, empty samples, and label inconsistencies.
- Duplicates and noise reduce effective data size and increase overfitting.
- Export results and filter problematic rows before training.

## Key Config Fields

| Field | Purpose |
|-------|---------|
| `dataset_path` or `dataset_name` | Input dataset |
| `sample_count` | Limit for quick scans |
| `output_path` | Directory for analysis outputs |
| `format` | Export format: csv, json, or parquet |
| `analyzers` | Built-in or custom analyzers to run |

## Outputs

- `analysis_summary.json` plus per-message and per-conversation analysis files.
"""
