# Analysis Configuration

{py:class}`~oumi.analyze.config.TypedAnalyzeConfig` controls how Oumi analyzes datasets. See {doc}`analyze` for usage examples.

## Core Settings

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_name` | `str` | Conditional | `None` | HuggingFace dataset name |
| `dataset_path` | `str` | Conditional | `None` | Path to local JSONL file |
| `split` | `str` | No | `"train"` | Dataset split to analyze |
| `subset` | `str` | No | `None` | Dataset subset/config name |
| `sample_count` | `int` | No | `None` | Max samples to analyze (`None` = all) |
| `output_path` | `str` | No | `"."` | Directory for output files |

Provide either `dataset_name` (HuggingFace Hub) or `dataset_path` (local JSONL file):

Both sources must already be in Oumi conversation format (each row / line: `{"messages": [{"role": "...", "content": "..."}]}`). Rows that fail to parse are skipped with a warning.

::::{tab-set}
:::{tab-item} HuggingFace Dataset

```yaml
dataset_name: <org>/<repo>
split: train
sample_count: 1000
```

:::
:::{tab-item} Local File

```yaml
dataset_path: /path/to/data.jsonl
```

:::
::::

## Analyzers

Configure analyzers as a list with `id`, `instance_id`, and optional `params`:

```yaml
analyzers:
  - id: length
    instance_id: Length
    params:
      tokenizer_name: cl100k_base
  - id: quality
    instance_id: Quality
  - id: turn_stats
    instance_id: TurnStats
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | Yes | — | Analyzer identifier (e.g., `length`, `quality`, `turn_stats`) |
| `instance_id` | `str` | No | Same as `id` | Label used as result key and metric path prefix |
| `params` | `dict` | No | `{}` | Analyzer-specific parameters |

The `instance_id` is used in metric paths for tests (e.g., `Length.total_tokens`) and as the column prefix in output DataFrames.

:::{note}
Each analyzer must have a unique `instance_id`. If omitted, it defaults to the `id` value.
:::

### `length` Analyzer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer_name` | `str` | `cl100k_base` | Tokenizer name (tiktoken encoding or HuggingFace model ID) |

Tiktoken encodings: `cl100k_base` (GPT-4), `p50k_base`, `o200k_base`.
HuggingFace models: any valid model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

### `quality` Analyzer Parameters

The `quality` analyzer has no configurable parameters. It always checks for non-alternating turns, missing user messages, misplaced system messages, empty messages, and invalid serialized values.

### `turn_stats` Analyzer Parameters

The `turn_stats` analyzer has no configurable parameters. It computes turn counts by role, system message presence, and first/last turn roles.

## Tests

Tests validate analysis results against configurable thresholds. Metrics are referenced as `"{instance_id}.{field_name}"`.

```yaml
tests:
  - id: max_tokens
    type: threshold
    metric: Length.total_tokens
    operator: ">"
    value: 10000
    max_percentage: 5.0
    severity: high
    title: "Token count exceeds limit"
```

### Common Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | Yes | — | Unique test identifier |
| `type` | `str` | Yes | — | Test type (`threshold`) |
| `metric` | `str` | Yes | — | Metric path (e.g., `Length.total_tokens`) |
| `severity` | `str` | No | `medium` | Failure severity: `high`, `medium`, or `low` |
| `title` | `str` | No | `""` | Human-readable title shown in results |
| `description` | `str` | No | `""` | Description of what the test checks |

### Threshold Tests

Check if a metric exceeds a threshold across the dataset.

| Field | Type | Description |
|-------|------|-------------|
| `operator` | `str` | Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=` |
| `value` | `number` | Value to compare against |
| `max_percentage` | `float` | At most this % of samples can match the condition |
| `min_percentage` | `float` | At least this % of samples must match the condition |

```yaml
# At most 5% of conversations can have > 10K tokens
- id: max_tokens
  type: threshold
  metric: Length.total_tokens
  operator: ">"
  value: 10000
  max_percentage: 5.0

# At most 5% of conversations can have non-alternating turns
- id: non_alternating
  type: threshold
  metric: Quality.has_non_alternating_turns
  operator: "=="
  value: true
  max_percentage: 5.0
```

## Output Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | `"."` | Directory for output files |
| `generate_report` | `bool` | `false` | Generate HTML report |
| `report_title` | `str` | `None` | Custom title for the report |

Override the output directory via CLI:

```bash
oumi analyze --config config.yaml --output /custom/path --format parquet
```

## Complete Example

```yaml
dataset_path: /path/to/data.jsonl
sample_count: 1000
output_path: ./analysis_output

analyzers:
  - id: length
    instance_id: Length
    params:
      tokenizer_name: cl100k_base
  - id: quality
    instance_id: Quality
  - id: turn_stats
    instance_id: TurnStats

tests:
  - id: max_tokens
    type: threshold
    metric: Length.total_tokens
    operator: ">"
    value: 10000
    max_percentage: 5.0
    severity: high
    title: "Token count exceeds 10K"
  - id: empty_turns
    type: threshold
    metric: Quality.has_empty_turns
    operator: "=="
    value: true
    max_percentage: 5.0
    severity: high
    title: "Conversations with empty turns"
```

## See Also

- {doc}`analyze` - Main analysis guide
- {py:class}`~oumi.analyze.config.TypedAnalyzeConfig` - Configuration API reference
- {py:class}`~oumi.analyze.config.AnalyzerConfig` - Analyzer configuration
