# Dataset Analysis

```{toctree}
:maxdepth: 2
:caption: Dataset Analysis
:hidden:

analyze_config
```

Oumi's dataset analysis framework helps you understand training data before and after fine-tuning. Compute metrics, identify outliers, compare datasets, and create filtered subsets.

**Key capabilities:**

- **Profile datasets**: Understand text length distributions, token counts, and statistics
- **Quality control**: Identify outliers, empty samples, or problematic data
- **Compare datasets**: Analyze multiple datasets with consistent metrics
- **Filter data**: Create filtered subsets based on analysis results

## Quick Start

::::{tab-set-code}
:::{code-block} bash
oumi analyze --config configs/examples/analyze/analyze.yaml
:::
:::{code-block} python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams

config = AnalyzeConfig(
    dataset_path="data/dataset_examples/oumi_format.jsonl",
    is_multimodal=False,
    analyzers=[SampleAnalyzerParams(id="length")],
)

analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()
print(analyzer.analysis_summary)
:::
::::

Oumi outputs results to `./analysis_output/` including per-message metrics, conversation aggregates, and statistical summaries.

## Configuration

A minimal configuration for a local file:

```yaml
dataset_path: data/dataset_examples/oumi_format.jsonl
is_multimodal: false
analyzers:
  - id: length
```

For complete configuration options including dataset sources, output settings, tokenizer configuration, and validation rules, see {doc}`analyze_config`.

## Available Analyzers

### Length Analyzer

The built-in `length` analyzer computes text length metrics:

| Metric | Description |
|--------|-------------|
| `char_count` | Number of characters |
| `word_count` | Number of words (space-separated) |
| `sentence_count` | Number of sentences (split on `.!?`) |
| `token_count` | Number of tokens (requires tokenizer) |

:::{tip}
Enable token counting by adding `tokenizer_config` to your configuration. See {doc}`analyze_config` for setup details.
:::

### Data Quality Analyzer

The built-in `quality` analyzer ({py:class}`~oumi.analyze.analyzers.quality.DataQualityAnalyzer`) catches five common data issues without running any model inference. It's meant as a cheap, first-pass sanity check before training or fine-tuning.

| Field                              | What it flags                                                             |
|------------------------------------|---------------------------------------------------------------------------|
| `has_non_alternating_turns`        | Consecutive same-role messages (`user`, `user`, …) in non-system turns     |
| `has_no_user_message`              | Conversation has no `user` message at all (including empty conversations) |
| `has_system_message_not_at_start`  | A `system` message appears anywhere other than position 0                  |
| `has_empty_turns` / `empty_turn_count` | Any message whose content is empty or whitespace-only                  |
| `has_invalid_values` / `invalid_value_patterns` | Strings like `NaN`, `null`, `None`, `undefined` leaked into content |

```yaml
analyzers:
  - id: quality
```

Because the output is typed ({py:class}`~oumi.analyze.analyzers.quality.DataQualityMetrics`), quality fields can be referenced by later **tests** using dotted metric paths (see [Testing Framework](#testing-framework)), e.g. `quality.has_no_user_message`.

### Turn Stats Analyzer

The built-in `turn_stats` analyzer ({py:class}`~oumi.analyze.analyzers.turn_stats.TurnStatsAnalyzer`) reports conversation shape: `num_turns`, `num_user_turns`, `num_assistant_turns`, `has_system_message`, `first_turn_role`, `last_turn_role`. Useful for finding malformed or single-sided conversations.

```yaml
analyzers:
  - id: turn_stats
```

## Typed Analyzer Framework

All built-in analyzers above (`length`, `quality`, `turn_stats`) are implemented in the **typed analyzer framework** ({py:class}`~oumi.analyze.base.BaseAnalyzer`). Each analyzer declares a pydantic result model, which gives you:

- **Auto-generated JSON schemas** for result documentation and validation.
- **Typed access** to analyzer output in Python (fields are proper attributes, not dict keys).
- **Metric paths** for the testing framework — `{analyzer_id}.{field_name}`, or `{instance_id}.{field_name}` when you run multiple instances of the same analyzer.

### Defining a Typed Analyzer

```python
from pydantic import BaseModel, Field
from oumi.analyze.base import ConversationAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.core.types.conversation import Conversation


class QuestionMetrics(BaseModel):
    num_questions: int = Field(description="Count of '?' characters")
    density: float = Field(description="Questions per message")


@register_sample_analyzer("questions")
class QuestionAnalyzer(ConversationAnalyzer[QuestionMetrics]):
    _result_model = QuestionMetrics

    @classmethod
    def get_config_schema(cls) -> dict:
        return {"properties": {}}

    def analyze(self, conversation: Conversation) -> QuestionMetrics:
        total = sum(m.content.count("?") for m in conversation.messages)
        return QuestionMetrics(
            num_questions=total,
            density=total / max(len(conversation.messages), 1),
        )
```

Point the config at your typed analyzer the same way as built-ins:

```yaml
analyzers:
  - id: questions
    instance_id: questions            # required for typed analyzers
```

When you need two configurations of the same analyzer (e.g. two `length` analyzers with different tokenizers), give each one a unique `instance_id`.

### Custom Metrics (No Code Registration Required)

For quick one-offs you don't want to package as an analyzer, declare a `custom_metrics` block directly in YAML:

```yaml
custom_metrics:
  - id: word_to_char_ratio
    scope: conversation              # message | conversation | dataset
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

```{warning}
Custom metric `function` strings are compiled and run as arbitrary Python. Only load configs from sources you trust.
```

## Testing Framework

The typed framework also ships a **testing** layer that evaluates analyzer output against thresholds and produces a pass/fail summary — useful for CI, regression detection, and "fail the run if more than 5% of conversations are missing a user message".

### Defining Tests

```yaml
tests:
  - id: max_words
    type: threshold
    metric: length.total_words        # <analyzer_id_or_instance_id>.<field>
    operator: ">"
    value: 10000
    max_percentage: 5.0               # fail if >5% of conversations match

  - id: no_missing_user_msg
    type: threshold
    metric: quality.has_no_user_message
    operator: "=="
    value: true
    max_percentage: 0.0               # fail if any conversation is missing a user
```

Each test compares a metric to a `value` using `operator`, then checks whether the flagged fraction exceeds `max_percentage` (or falls below `min_percentage`).

### Running Tests Incrementally with BatchTestEngine

For large datasets where full analyzer output won't fit in memory, use {py:class}`~oumi.analyze.testing.batch_engine.BatchTestEngine`. It accumulates only lightweight counters and per-test affected conversation IDs as batches stream through, then returns a `TestSummary` at the end:

```python
from oumi.analyze.testing.batch_engine import BatchTestEngine

engine = BatchTestEngine(config.tests)

for batch_results, batch_conversation_ids in stream_batches():
    engine.process_batch(batch_results, batch_conversation_ids)

summary = engine.finalize()
print(f"{summary.passed_tests}/{summary.total_tests} passed "
      f"({summary.pass_rate}%)")

# IDs of conversations that caused test failures, per test:
affected = engine.get_affected_conversation_ids()
```

Use the standard `TestEngine` (same module) when the full dataset fits in memory; use `BatchTestEngine` when it doesn't.

## Working with Results

### Analysis Summary

Access summary statistics after running analysis:

```python
summary = analyzer.analysis_summary

# Dataset overview
print(f"Dataset: {summary['dataset_overview']['dataset_name']}")
print(f"Samples: {summary['dataset_overview']['conversations_analyzed']}")

# Message-level statistics
for analyzer_name, metrics in summary['message_level_summary'].items():
    for metric_name, stats in metrics.items():
        print(f"{metric_name}: mean={stats['mean']}, std={stats['std']}")
```

### DataFrames

Access raw analysis data as pandas DataFrames:

```python
message_df = analyzer.message_df        # One row per message
conversation_df = analyzer.conversation_df  # One row per conversation
full_df = analyzer.analysis_df          # Merged view
```

The `conversation_df` includes:

- `conversation_index`: Index of the conversation in the dataset
- `conversation_id`: Unique identifier for the conversation
- `num_messages`: Number of messages in the conversation
- `conversation_text_content`: Full conversation rendered as text (formatted as "ROLE: content" for each message)

### Querying and Filtering

Filter results using pandas query syntax:

```python
# Find long messages
long_messages = analyzer.query("text_content_length_word_count > 10")

# Find short conversations
short_convos = analyzer.query_conversations("conversation_text_content_length_char_count < 100")

# Create filtered dataset
filtered_dataset = analyzer.filter("text_content_length_word_count < 100")
```

## Supported Dataset Formats

| Format | Description | Example |
|--------|-------------|---------|
| **oumi** | Multi-turn conversations with roles | SFT, instruction-following |
| **alpaca** | Instruction/input/output format | Stanford Alpaca |
| **DPO** | Preference pairs (chosen/rejected) | Preference learning |
| **KTO** | Binary feedback format | Human feedback |
| **Pretraining** | Raw text | C4, The Pile |

### Analyzing HuggingFace Datasets

Analyze any HuggingFace Hub dataset directly:

::::{tab-set-code}
:::{code-block} yaml

# hf_analyze.yaml

dataset_name: argilla/databricks-dolly-15k-curated-en
split: train
sample_count: 100
output_path: ./analysis_output/dolly
analyzers:

- id: length
:::
:::{code-block} python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams

config = AnalyzeConfig(
    dataset_name="argilla/databricks-dolly-15k-curated-en",
    split="train",
    sample_count=100,
    analyzers=[SampleAnalyzerParams(id="length")],
)
analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()
:::
::::

## Exporting Results

::::{tab-set-code}
:::{code-block} bash

# Export to CSV (default)

oumi analyze --config configs/examples/analyze/analyze.yaml

# Export to Parquet

oumi analyze --config configs/examples/analyze/analyze.yaml --format parquet

# Override output directory

oumi analyze --config configs/examples/analyze/analyze.yaml --output ./my_results
:::
::::

**Output files:**

| File | Description |
|------|-------------|
| `message_analysis.{format}` | Per-message metrics |
| `conversation_analysis.{format}` | Per-conversation aggregated metrics |
| `analysis_summary.json` | Statistical summary |

## Creating Custom Analyzers

You can create custom analyzers to compute domain-specific metrics for your datasets. Custom analyzers extend the `SampleAnalyzer` base class and are registered using the `@register_sample_analyzer` decorator.

For example, to build a question detector analyzer:

```python
import re
from typing import Optional
import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("questions")
class QuestionAnalyzer(SampleAnalyzer):
    """Counts questions in text fields."""

    def _count_questions(self, text: str) -> int:
        """Count question marks in text. Replace with your own logic."""
        return len(re.findall(r"\?", text))

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        result_df = df.copy()

        # Find text columns using the schema
        text_columns = [
            col for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        for column in text_columns:
            result_df[f"{column}_question_count"] = (
                df[column].astype(str).apply(self._count_questions)
            )

        return result_df
```

Use your analyzer by referencing its registered ID:

::::{tab-set-code}
:::{code-block} yaml
analyzers:

- id: questions
:::
:::{code-block} python

# Import your analyzer module to trigger registration

import my_analyzers  # noqa: F401

config = AnalyzeConfig(
    dataset_path="data/my_dataset.jsonl",
    is_multimodal=False,
    analyzers=[SampleAnalyzerParams(id="questions")],
)
:::
::::

**Key points:**

- Register with a unique ID via `@register_sample_analyzer("id")`
- Use `schema` to find text columns (`ContentType.TEXT`)
- Prefix output columns with the source column name (e.g., `{column}_question_count`)

## API Reference

- {py:class}`~oumi.core.configs.AnalyzeConfig` - Configuration class
- {py:class}`~oumi.core.analyze.dataset_analyzer.DatasetAnalyzer` - Main analyzer class
- {py:class}`~oumi.core.analyze.sample_analyzer.SampleAnalyzer` - Base class for analyzers
- {py:class}`~oumi.core.analyze.length_analyzer.LengthAnalyzer` - Built-in length analyzer
