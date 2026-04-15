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

::::{tab-set}
:::{tab-item} HuggingFace Dataset

```yaml
dataset_name: argilla/databricks-dolly-15k-curated-en
split: train
sample_count: 1000
```

:::
:::{tab-item} Local File

```yaml
dataset_path: /path/to/data.jsonl
```

Local files must be in JSONL format with Oumi conversation structure (each line: `{"messages": [{"role": "...", "content": "..."}]}`).

:::
::::

## Analyzers

Configure analyzers as a list with `type`, optional `display_name`, and optional `params`:

```yaml
analyzers:
  - type: length
    display_name: Length
    params:
      tokenizer_name: cl100k_base
  - type: quality
    display_name: Quality
  - type: turn_stats
    display_name: TurnStats
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `str` | Yes | — | Analyzer identifier (e.g., `length`, `quality`, `turn_stats`) |
| `display_name` | `str` | No | Same as `type` | Label used as result key and metric path prefix |
| `params` | `dict` | No | `{}` | Analyzer-specific parameters |

The `display_name` is used in metric paths for tests (e.g., `Length.total_tokens`) and as the column prefix in output DataFrames.

:::{note}
Each analyzer must have a unique `display_name`. If omitted, it defaults to the `type` value.
:::

### `length` Analyzer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer_name` | `str` | `cl100k_base` | Tokenizer name (tiktoken encoding or HuggingFace model ID) |

Tiktoken encodings: `cl100k_base` (GPT-4), `p50k_base`, `o200k_base`.
HuggingFace models: any valid model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

### `quality` Analyzer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check_turn_pattern` | `bool` | `true` | Check for alternating user/assistant turns |
| `check_empty_content` | `bool` | `true` | Check for empty messages |
| `check_invalid_values` | `bool` | `true` | Check for serialized NaN/null/None |
| `check_truncation` | `bool` | `true` | Check for truncated messages |
| `check_refusals` | `bool` | `true` | Check for policy refusal patterns |
| `check_tags` | `bool` | `true` | Check for unbalanced tags |
| `context_4k_threshold` | `int` | `4096` | Token threshold for 4K context check |
| `context_8k_threshold` | `int` | `8192` | Token threshold for 8K context check |

### `turn_stats` Analyzer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_system_in_counts` | `bool` | `false` | Include system messages in total turn counts |

## Tests

Tests validate analysis results against configurable thresholds. Metrics are referenced as `"{display_name}.{field_name}"`.

```yaml
tests:
  - id: max_tokens
    type: threshold
    metric: Length.total_tokens
    operator: ">"
    value: 10000
    max_percentage: 5.0
    severity: high
    display_name: "Token count exceeds limit"
```

### Common Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | Yes | — | Unique test identifier |
| `type` | `str` | Yes | — | Test type: `threshold`, `percentage`, or `range` |
| `metric` | `str` | Yes | — | Metric path (e.g., `Length.total_tokens`) |
| `severity` | `str` | No | `medium` | Failure severity: `high`, `medium`, or `low` |
| `display_name` | `str` | No | `""` | Human-readable title shown in results |
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

### Percentage Tests

Check what percentage of samples match a condition string.

| Field | Type | Description |
|-------|------|-------------|
| `condition` | `str` | Condition to evaluate (e.g., `"== True"`, `"> 0.5"`) |
| `max_percentage` | `float` | Maximum allowed matching percentage |
| `min_percentage` | `float` | Minimum required matching percentage |

```yaml
# At most 5% should have empty turns
- id: empty_turns
  type: percentage
  metric: Quality.has_empty_turns
  condition: "== True"
  max_percentage: 5.0
```

### Range Tests

Check if metric values fall within a range.

| Field | Type | Description |
|-------|------|-------------|
| `min_value` | `float` | Minimum allowed value |
| `max_value` | `float` | Maximum allowed value |
| `max_percentage` | `float` | Maximum % of samples allowed outside the range (default: 0%) |

```yaml
# Tokens should be between 10 and 8192
- id: token_range
  type: range
  metric: Length.total_tokens
  min_value: 10
  max_value: 8192
  max_percentage: 10.0
```

## Custom Metrics

Define custom Python functions to compute additional metrics:

```yaml
custom_metrics:
  - id: word_to_char_ratio
    scope: conversation
    description: "Ratio of words to characters"
    output_schema:
      - name: ratio
        type: float
        description: "Words divided by characters"
    function: |
      def compute(conversation):
          chars = sum(len(m.content) for m in conversation.messages)
          words = sum(len(m.content.split()) for m in conversation.messages)
          return {"ratio": words / chars if chars > 0 else 0.0}
```

:::{warning}
Custom metrics with `function` fields execute arbitrary Python code. Only load configurations from trusted sources. Configs with custom code require `allow_custom_code=True` when loading programmatically.
:::

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
  - type: length
    display_name: Length
    params:
      tokenizer_name: cl100k_base
  - type: quality
    display_name: Quality
  - type: turn_stats
    display_name: TurnStats

tests:
  - id: max_tokens
    type: threshold
    metric: Length.total_tokens
    operator: ">"
    value: 10000
    max_percentage: 5.0
    severity: high
    display_name: "Token count exceeds 10K"
  - id: empty_turns
    type: percentage
    metric: Quality.has_empty_turns
    condition: "== True"
    max_percentage: 5.0
    severity: high
    display_name: "Conversations with empty turns"
  - id: token_range
    type: range
    metric: Length.total_tokens
    min_value: 10
    max_value: 8192
    max_percentage: 15.0
    severity: low
    display_name: "Tokens within context window"
```

## See Also

- {doc}`analyze` - Main analysis guide
- {py:class}`~oumi.analyze.config.TypedAnalyzeConfig` - Configuration API reference
- {py:class}`~oumi.analyze.config.AnalyzerConfig` - Analyzer configuration
