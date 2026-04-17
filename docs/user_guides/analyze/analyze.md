# Dataset Analysis

```{toctree}
:maxdepth: 2
:caption: Dataset Analysis
:hidden:

analyze_config
```

Oumi's dataset analysis framework helps you understand you datasets. Compute metrics, identify quality issues and validate data with configurable tests.

**Key capabilities:**

- **Profile datasets**: Token counts, length distributions, turn statistics
- **Quality control**: Turn alternation, empty messages, invalid values
- **Validate data**: Configurable threshold tests with percentage tolerances
- **Export results**: CSV, JSON, or Parquet output with statistical summaries

## Quick Start

::::{tab-set-code}
:::{code-block} bash
oumi analyze --config configs/examples/analyze/analyze.yaml
:::
:::{code-block} python
from oumi.analyze import TypedAnalyzeConfig, AnalyzerConfig
from oumi.cli.analyze import run_typed_analysis

config = TypedAnalyzeConfig(
    dataset_path="data/dataset_examples/oumi_format.jsonl",
    analyzers=[
        AnalyzerConfig(id="length", instance_id="Length"),
        AnalyzerConfig(id="quality", instance_id="Quality"),
    ],
)

results = run_typed_analysis(config)
:::
::::

Results are saved to the output directory (default: current directory) including per-conversation metrics, test results, and statistical summaries. Set it via `--output` on the CLI or `output_path` in the YAML config.

:::{tip}
You can use `-c` as a shorthand for `--config` in all CLI examples.
:::

## Configuration

A minimal YAML configuration:

```yaml
dataset_path: data/dataset_examples/oumi_format.jsonl

analyzers:
  - id: length
    instance_id: Length
    params:
      tokenizer_name: cl100k_base
```

For complete configuration options including tests, custom metrics, and tokenizer settings, see {doc}`analyze_config`.

## Available Analyzers

### Length Analyzer (`length`)

Computes token and message count metrics using a configurable tokenizer.


| Metric                   | Description                            |
| ------------------------ | -------------------------------------- |
| `total_tokens`           | Total tokens across all messages       |
| `avg_tokens_per_message` | Average tokens per message             |
| `num_messages`           | Number of messages in the conversation |
| `user_total_tokens`      | Total tokens in user messages          |
| `assistant_total_tokens` | Total tokens in assistant messages     |
| `system_total_tokens`    | Total tokens in system messages        |


:::{tip}
Configure the tokenizer via `params.tokenizer_name`. Supports tiktoken encodings (e.g., `cl100k_base`) and HuggingFace model IDs (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
:::

### Quality Analyzer (`quality`)

Fast, non-LLM quality checks for data validation.


| Metric                            | Description                                             |
| --------------------------------- | ------------------------------------------------------- |
| `has_non_alternating_turns`       | Consecutive same-role messages exist (excluding system) |
| `has_no_user_message`             | Conversation contains no user message                   |
| `has_system_message_not_at_start` | System message appears after position 0                 |
| `has_empty_turns`                 | Any message has empty or whitespace-only content        |
| `empty_turn_count`                | Number of empty/whitespace-only messages                |
| `has_invalid_values`              | Contains serialized `NaN`, `null`, `None`, `undefined`  |
| `invalid_value_patterns`          | List of invalid value patterns found                    |


### Turn Stats Analyzer (`turn_stats`)

Conversation structure and turn count metrics.


| Metric                | Description                                   |
| --------------------- | --------------------------------------------- |
| `num_turns`           | Total number of turns (messages)              |
| `num_user_turns`      | Number of user turns                          |
| `num_assistant_turns` | Number of assistant turns                     |
| `num_tool_turns`      | Number of tool turns                          |
| `has_system_message`  | Whether the conversation has a system message |
| `first_turn_role`     | Role of the first message                     |
| `last_turn_role`      | Role of the last message                      |


Use `oumi analyze --list-metrics` to see all available metrics and their descriptions.

## Working with Results

### Output Files


| File                | Description                                         |
| ------------------- | --------------------------------------------------- |
| `analysis.{format}` | Per-conversation metrics (one row per conversation) |
| `test_results.json` | Test pass/fail details (if tests configured)        |
| `summary.json`      | Statistical summary (mean, std, min, max)           |


### Exporting

::::{tab-set-code}
:::{code-block} bash

# Export to CSV (default)

oumi analyze --config config.yaml

# Export to JSON

oumi analyze --config config.yaml --format json

# Export to Parquet

oumi analyze --config config.yaml --format parquet

# Override output directory

oumi analyze --config config.yaml --output ./my_results
:::
::::

### Programmatic Access

```python
from oumi.analyze import TypedAnalyzeConfig
from oumi.cli.analyze import run_typed_analysis

config = TypedAnalyzeConfig.from_yaml("config.yaml")
output = run_typed_analysis(config)

# Analyzer results keyed by instance_id
for length_result in output["results"]["Length"]:
    print(f"Tokens: {length_result.total_tokens}")

# Pre-built DataFrame (one row per conversation)
df = output["dataframe"]
print(df.describe())

# Test summary (if tests were configured)
if output["test_summary"]:
    summary = output["test_summary"]
    print(f"{summary.passed_tests}/{summary.total_tests} passed")
```

## Analyzing HuggingFace Datasets

Analyze any HuggingFace Hub dataset directly:

Rows must already be in Oumi conversation format
(each row: `{"messages": [{"role": "...", "content": "..."}]}`). Rows that
don't parse are skipped with a warning. To analyze instruction-style datasets
(e.g. `prompt`/`response` fields), pre-convert them to Oumi JSONL first and
use `dataset_path`.

::::{tab-set-code}
:::{code-block} yaml

# hf_analyze.yaml

dataset_name: <org>/<repo>
split: train
sample_count: 100
output_path: ./analysis_output

analyzers:
  - id: length
    instance_id: Length
    params:
      tokenizer_name: cl100k_base
  - id: quality
    instance_id: Quality
:::
:::{code-block} python
from oumi.analyze import TypedAnalyzeConfig, AnalyzerConfig
from oumi.cli.analyze import run_typed_analysis

config = TypedAnalyzeConfig(
    dataset_name="<org>/<repo>",
    split="train",
    sample_count=100,
    analyzers=[
        AnalyzerConfig(id="length", instance_id="Length"),
        AnalyzerConfig(id="quality", instance_id="Quality"),
    ],
)
results = run_typed_analysis(config)
:::
::::

## Data Validation with Tests

Configure tests to automatically validate your dataset against quality thresholds:

```yaml
analyzers:
  - id: length
    instance_id: Length
  - id: quality
    instance_id: Quality

tests:
  - id: max_tokens
    type: threshold
    metric: Length.total_tokens
    operator: ">"
    value: 10000
    max_percentage: 5.0
    severity: high
    title: "Token count exceeds 10K"

  - id: no_empty_turns
    type: threshold
    metric: Quality.has_empty_turns
    operator: "=="
    value: true
    max_percentage: 5.0
    severity: high
    title: "Conversations with empty turns"
```

Metrics are referenced as `"{instance_id}.{field_name}"` (e.g., `Length.total_tokens`, `Quality.has_empty_turns`).

See {doc}`analyze_config` for full test configuration options.

## Writing Custom Analyzers

Create a custom analyzer by subclassing one of the base classes and registering it:

```python
from pydantic import BaseModel, Field
from oumi.analyze.base import ConversationAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.core.types.conversation import Conversation


class QuestionMetrics(BaseModel):
    num_questions: int = Field(description="Count of '?' in all messages")
    density: float = Field(description="Questions per message")


@register_sample_analyzer("questions")
class QuestionAnalyzer(ConversationAnalyzer[QuestionMetrics]):
    def analyze(self, conversation: Conversation) -> QuestionMetrics:
        total = sum(
            m.content.count("?")
            for m in conversation.messages
            if isinstance(m.content, str)
        )
        return QuestionMetrics(
            num_questions=total,
            density=total / max(len(conversation.messages), 1),
        )
```

Then reference it in YAML the same way as built-ins:

```yaml
analyzers:
  - id: questions
    instance_id: Questions
```

Base classes for different scopes:

| Base Class | Scope | `analyze()` Input |
|---|---|---|
| {py:class}`~oumi.analyze.base.MessageAnalyzer` | Per message | `Message` |
| {py:class}`~oumi.analyze.base.ConversationAnalyzer` | Per conversation | `Conversation` |
| {py:class}`~oumi.analyze.base.DatasetAnalyzer` | Entire dataset | `list[Conversation]` |
| {py:class}`~oumi.analyze.base.PreferenceAnalyzer` | Preference pairs | `(Conversation, Conversation)` |

## API Reference

- {py:class}`~oumi.analyze.config.TypedAnalyzeConfig` - Configuration class
- {py:class}`~oumi.analyze.config.AnalyzerConfig` - Analyzer configuration
- {py:class}`~oumi.analyze.pipeline.AnalysisPipeline` - Analysis pipeline
- {py:class}`~oumi.analyze.base.ConversationAnalyzer` - Base class for conversation-level analyzers
- {py:class}`~oumi.analyze.base.MessageAnalyzer` - Base class for message-level analyzers
- {py:class}`~oumi.analyze.base.DatasetAnalyzer` - Base class for dataset-level analyzers
- {py:class}`~oumi.analyze.analyzers.length.LengthAnalyzer` - Length metrics
- {py:class}`~oumi.analyze.analyzers.quality.DataQualityAnalyzer` - Quality checks
- {py:class}`~oumi.analyze.analyzers.turn_stats.TurnStatsAnalyzer` - Turn statistics
- {py:class}`~oumi.analyze.testing.engine.TestEngine` - Test engine (in-memory)
- {py:class}`~oumi.analyze.testing.batch_engine.BatchTestEngine` - Incremental test engine
