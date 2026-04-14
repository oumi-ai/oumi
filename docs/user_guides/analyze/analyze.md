# Dataset Analysis

```{toctree}
:maxdepth: 2
:caption: Dataset Analysis
:hidden:

analyze_config
```

Oumi's dataset analysis framework helps you understand training data before and after fine-tuning. Compute metrics, identify quality issues, compare datasets, and validate data with configurable tests.

**Key capabilities:**

- **Profile datasets**: Token counts, length distributions, turn statistics
- **Quality control**: Empty messages, truncation, policy refusals, invalid values
- **Validate data**: Configurable tests with threshold, percentage, and range checks
- **Export results**: CSV, JSON, or Parquet output with statistical summaries

## Quick Start

::::{tab-set-code}
:::{code-block} bash
oumi analyze -c configs/examples/analyze/analyze.yaml
:::
:::{code-block} python
from oumi.analyze import run_typed_analysis, TypedAnalyzeConfig, AnalyzerConfig

config = TypedAnalyzeConfig(
    dataset_path="data/dataset_examples/oumi_format.jsonl",
    analyzers=[
        AnalyzerConfig(type="length", display_name="Length"),
        AnalyzerConfig(type="quality", display_name="Quality"),
    ],
)

results = run_typed_analysis(config)
:::
::::

Results are saved to the configured `output_path` (default: current directory) including per-conversation metrics, test results, and statistical summaries.

## Configuration

A minimal YAML configuration:

```yaml
dataset_path: data/dataset_examples/oumi_format.jsonl

analyzers:
  - type: length
    display_name: Length
    params:
      tokenizer_name: cl100k_base
```

For complete configuration options including tests, custom metrics, and tokenizer settings, see {doc}`analyze_config`.

## Available Analyzers

### Length Analyzer (`length`)

Computes token and message count metrics using a configurable tokenizer.

| Metric | Description |
|--------|-------------|
| `total_tokens` | Total tokens across all messages |
| `avg_tokens_per_message` | Average tokens per message |
| `num_messages` | Number of messages in the conversation |
| `user_total_tokens` | Total tokens in user messages |
| `assistant_total_tokens` | Total tokens in assistant messages |
| `system_total_tokens` | Total tokens in system messages |

:::{tip}
Configure the tokenizer via `params.tokenizer_name`. Supports tiktoken encodings (e.g., `cl100k_base`) and HuggingFace model IDs (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
:::

### Quality Analyzer (`quality`)

Fast, non-LLM quality checks for data validation.

| Metric | Description |
|--------|-------------|
| `has_non_alternating_turns` | Consecutive same-role messages exist (excluding system) |
| `has_no_user_message` | Conversation contains no user message |
| `has_system_message_not_at_start` | System message appears after position 0 |
| `has_empty_turns` | Any message has empty or whitespace-only content |
| `empty_turn_count` | Number of empty/whitespace-only messages |
| `has_invalid_values` | Contains serialized `NaN`, `null`, `None`, `undefined` |
| `invalid_value_patterns` | List of invalid value patterns found |

### Turn Stats Analyzer (`turn_stats`)

Conversation structure and turn count metrics.

| Metric | Description |
|--------|-------------|
| `num_turns` | Total number of turns (messages) |
| `num_user_turns` | Number of user turns |
| `num_assistant_turns` | Number of assistant turns |
| `num_tool_turns` | Number of tool turns |
| `has_system_message` | Whether the conversation has a system message |
| `first_turn_role` | Role of the first message |
| `last_turn_role` | Role of the last message |

Use `oumi analyze --list-metrics` to see all available metrics and their descriptions.

## Working with Results

### Output Files

| File | Description |
|------|-------------|
| `analysis.{format}` | Per-conversation metrics (one row per conversation) |
| `test_results.json` | Test pass/fail details (if tests configured) |
| `summary.json` | Statistical summary (mean, std, min, max) |

### Exporting

::::{tab-set-code}
:::{code-block} bash

# Export to CSV (default)
oumi analyze -c config.yaml

# Export to JSON
oumi analyze -c config.yaml --format json

# Export to Parquet
oumi analyze -c config.yaml --format parquet

# Override output directory
oumi analyze -c config.yaml --output ./my_results
:::
::::

### Programmatic Access

```python
from oumi.analyze import run_typed_analysis, TypedAnalyzeConfig

config = TypedAnalyzeConfig.from_yaml("config.yaml")
results = run_typed_analysis(config)

# results is a dict mapping analyzer display_name to list of result models
for length_result in results["Length"]:
    print(f"Tokens: {length_result.total_tokens}")

# Convert to DataFrame
from oumi.analyze import to_analysis_dataframe

df = to_analysis_dataframe(results)
print(df.describe())
```

## Analyzing HuggingFace Datasets

Analyze any HuggingFace Hub dataset directly:

::::{tab-set-code}
:::{code-block} yaml

# hf_analyze.yaml
dataset_name: argilla/databricks-dolly-15k-curated-en
split: train
sample_count: 100
output_path: ./analysis_output/dolly

analyzers:
  - type: length
    display_name: Length
    params:
      tokenizer_name: cl100k_base
  - type: quality
    display_name: Quality
:::
:::{code-block} python
from oumi.analyze import run_typed_analysis, TypedAnalyzeConfig, AnalyzerConfig

config = TypedAnalyzeConfig(
    dataset_name="argilla/databricks-dolly-15k-curated-en",
    split="train",
    sample_count=100,
    analyzers=[
        AnalyzerConfig(type="length", display_name="Length"),
        AnalyzerConfig(type="quality", display_name="Quality"),
    ],
)
results = run_typed_analysis(config)
:::
::::

## Data Validation with Tests

Configure tests to automatically validate your dataset against quality thresholds:

```yaml
analyzers:
  - type: length
    display_name: Length
  - type: quality
    display_name: Quality

tests:
  - id: max_tokens
    type: threshold
    metric: Length.total_tokens
    operator: ">"
    value: 10000
    max_percentage: 5.0
    severity: high
    display_name: "Token count exceeds 10K"

  - id: no_empty_turns
    type: percentage
    metric: Quality.has_empty_turns
    condition: "== False"
    min_percentage: 95.0
    severity: high
    display_name: "No empty turns"

  - id: token_range
    type: range
    metric: Length.total_tokens
    min_value: 10
    max_value: 8192
    max_percentage: 10.0
    severity: low
    display_name: "Tokens within context window"
```

Metrics are referenced as `"{display_name}.{field_name}"` (e.g., `Length.total_tokens`, `Quality.has_empty_turns`).

See {doc}`analyze_config` for full test configuration options.

## API Reference

- {py:class}`~oumi.analyze.config.TypedAnalyzeConfig` - Configuration class
- {py:class}`~oumi.analyze.config.AnalyzerConfig` - Analyzer configuration
- {py:class}`~oumi.analyze.pipeline.AnalysisPipeline` - Analysis pipeline
- {py:class}`~oumi.analyze.base.ConversationAnalyzer` - Base class for analyzers
- {py:class}`~oumi.analyze.analyzers.length.LengthAnalyzer` - Length metrics
- {py:class}`~oumi.analyze.analyzers.quality.DataQualityAnalyzer` - Quality checks
- {py:class}`~oumi.analyze.analyzers.turn_stats.TurnStatsAnalyzer` - Turn statistics
