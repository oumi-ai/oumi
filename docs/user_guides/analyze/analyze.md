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
    dataset_format="oumi",
    is_multimodal=False,
    analyzers=[SampleAnalyzerParams(id="length")],
)

analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()
print(analyzer.analysis_summary)
:::
::::

Oumi outputs results to `./analysis_output/basic/` including per-message metrics, conversation aggregates, and statistical summaries.

## Configuration

A minimal configuration for a local file:

```yaml
dataset_path: data/dataset_examples/oumi_format.jsonl
dataset_format: oumi
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

:::{note}
Filtering requires map-style datasets. Streaming/iterable datasets cannot be filtered by index.
:::

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

dataset_name: databricks/dolly-15k
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
    dataset_name="databricks/dolly-15k",
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
:::{code-block} python
import json

# Export DataFrames

analyzer.message_df.to_csv("message_analysis.csv", index=False)
analyzer.conversation_df.to_parquet("conversation_analysis.parquet")

# Export summary

with open("summary.json", "w") as f:
    json.dump(analyzer.analysis_summary, f, indent=2)
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

### Basic Structure

```python
from typing import Optional
import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("my_analyzer")  # Register with unique ID
class MyAnalyzer(SampleAnalyzer):
    """Custom analyzer that computes domain-specific metrics."""

    def __init__(self, *, my_option: bool = True):
        """Initialize with configuration options.

        Args:
            my_option: Example parameter passed from config
        """
        self.my_option = my_option

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields and return metrics.

        Args:
            df: Input DataFrame with text fields
            schema: Column schema dict identifying column types

        Returns:
            DataFrame with added analysis columns
        """
        result_df = df.copy()

        # Find text columns using the schema
        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT
            and col in df.columns
        ]

        # Compute metrics for each text column
        for column in text_columns:
            if self.my_option:
                # Add your custom metric computation here
                result_df[f"{column}_my_metric"] = (
                    df[column].astype(str).apply(self._compute_metric)
                )

        return result_df

    def _compute_metric(self, text: str) -> float:
        """Compute your custom metric."""
        # Your metric logic here
        return len(text.split()) / max(len(text), 1)
```

### Using Your Custom Analyzer

Once registered, use your analyzer in configs by its ID:

```yaml
# my_config.yaml
dataset_path: data/my_dataset.jsonl
dataset_format: oumi
is_multimodal: false

analyzers:
  - id: my_analyzer  # Your registered ID
    params:
      my_option: true  # Passed to __init__
```

Or in Python:

```python
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

# Import your analyzer module to trigger registration
import my_analyzers  # noqa: F401

config = AnalyzeConfig(
    dataset_path="data/my_dataset.jsonl",
    dataset_format="oumi",
    is_multimodal=False,
    analyzers=[
        SampleAnalyzerParams(id="my_analyzer", params={"my_option": True})
    ],
)

analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()
```

### Example: Question Detector

Here's a practical example that detects questions in text:

```python
import re
from typing import Optional
import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("questions")
class QuestionAnalyzer(SampleAnalyzer):
    """Analyzer that detects and counts questions in text."""

    def __init__(self, *, count_questions: bool = True, has_question: bool = True):
        self.count_questions = count_questions
        self.has_question = has_question

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        result_df = df.copy()

        text_columns = [
            col for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        for column in text_columns:
            if self.count_questions:
                result_df[f"{column}_question_count"] = (
                    df[column].astype(str).apply(
                        lambda t: len(re.findall(r'\?', t))
                    )
                )

            if self.has_question:
                result_df[f"{column}_has_question"] = (
                    df[column].astype(str).str.contains(r'\?', regex=True)
                )

        return result_df
```

### Key Points

- **Registration**: Use `@register_sample_analyzer("id")` with a unique ID
- **Schema**: Use the `schema` parameter to identify text columns via `ContentType.TEXT`
- **Column naming**: Prefix output columns with the source column name for clarity
- **Parameters**: Constructor parameters are passed from `params` in the config
- **Import**: Ensure your analyzer module is imported before creating the config

## Troubleshooting

````{dropdown} Common Issues
**"Dataset not found in registry"**

HuggingFace datasets not registered in Oumi load directly from the Hub. Verify internet access and dataset name.

**"Tokenizer required for token_count"**

Provide a `tokenizer_config` to compute token counts:

```yaml
tokenizer_config:
  model_name: openai-community/gpt2

analyzers:
  - id: length
    params:
      token_count: true
```

**"Filtering not supported for iterable datasets"**

Streaming datasets cannot be filtered by index. Use `query()` to get filtered indices:

```python
filtered_df = analyzer.query("text_content_length_word_count > 100")
valid_indices = filtered_df.conversation_index.unique().tolist()
```
````

## API Reference

- {py:class}`~oumi.core.configs.AnalyzeConfig` - Configuration class
- {py:class}`~oumi.core.analyze.dataset_analyzer.DatasetAnalyzer` - Main analyzer class
- {py:class}`~oumi.core.analyze.sample_analyzer.SampleAnalyzer` - Base class for analyzers
- {py:class}`~oumi.core.analyze.length_analyzer.LengthAnalyzer` - Built-in length analyzer
