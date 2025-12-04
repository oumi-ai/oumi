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

### Querying and Filtering

Filter results using pandas query syntax:

```python
# Find long messages
long_messages = analyzer.query("text_content_length_word_count > 10")

# Find short conversations
short_convos = analyzer.query_conversations("text_content_length_char_count < 100")

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
