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
oumi analyze --config configs/examples/analyze/basic_analyze.yaml
:::
:::{code-block} python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams

config = AnalyzeConfig(
    dataset_source=DatasetSource.CONFIG,
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
dataset_source: CONFIG
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

dataset_source: CONFIG
dataset_name: databricks/dolly-15k
split: train
sample_count: 100
output_path: ./analysis_output/dolly
analyzers:

- id: length
:::
:::{code-block} python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams

config = AnalyzeConfig(
    dataset_source=DatasetSource.CONFIG,
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

oumi analyze --config configs/examples/analyze/basic_analyze.yaml

# Export to Parquet

oumi analyze --config configs/examples/analyze/basic_analyze.yaml --format parquet

# Override output directory

oumi analyze --config configs/examples/analyze/basic_analyze.yaml --output ./my_results
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
