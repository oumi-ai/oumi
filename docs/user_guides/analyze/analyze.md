# Dataset Analysis

```{toctree}
:maxdepth: 2
:caption: Dataset Analysis
:hidden:

analyze_config
```

## Overview

Oumi provides a powerful dataset analysis framework that helps you understand your training data before and after fine-tuning. The analysis tools compute various metrics about your datasets, enabling you to:

- **Profile datasets**: Understand text length distributions, token counts, and other statistics
- **Quality control**: Identify outliers, empty samples, or problematic data points
- **Compare datasets**: Analyze multiple datasets with consistent metrics
- **Filter data**: Create filtered subsets based on analysis results

Key features include:

- **Plugin architecture**: Extensible analyzer system with built-in and custom analyzers
- **Multi-format support**: Works with conversation, DPO, KTO, pretraining, and custom datasets
- **HuggingFace integration**: Analyze any dataset from HuggingFace Hub directly
- **Export options**: Save results to CSV, JSON, or Parquet formats
- **CLI and Python API**: Use from command line or programmatically

## Quick Start

### Using the CLI

Analyze the included example dataset:

```bash
oumi analyze --config configs/examples/analyze/basic_analyze.yaml
```

This analyzes `data/dataset_examples/oumi_format.jsonl` and outputs results to `./analysis_output/basic/`.

To include token counts (requires downloading a tokenizer):

```bash
oumi analyze --config configs/examples/analyze/analyze_with_tokens.yaml
```

Export results to a different format:

```bash
oumi analyze --config configs/examples/analyze/basic_analyze.yaml --output ./my_results --format parquet
```

### Using the Python API

```python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams

# Analyze the included example dataset
config = AnalyzeConfig(
    dataset_source=DatasetSource.CONFIG,
    dataset_path="data/dataset_examples/oumi_format.jsonl",
    dataset_format="oumi",
    is_multimodal=False,
    analyzers=[
        SampleAnalyzerParams(
            id="length",
            params={
                "char_count": True,
                "word_count": True,
                "sentence_count": True,
            }
        )
    ],
)

# Create analyzer and run
analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()

# Access results
print(analyzer.analysis_summary)
df = analyzer.message_df  # Pandas DataFrame with results
```

## Configuration

### Minimal Configuration

A minimal analysis configuration for a local file:

```yaml
dataset_source: CONFIG
dataset_path: data/dataset_examples/oumi_format.jsonl
dataset_format: oumi
is_multimodal: false
analyzers:
  - id: length
```

### Full Configuration Options

```yaml
# Required: How to load the dataset (CONFIG or DIRECT)
dataset_source: CONFIG

# Dataset specification - local file
dataset_path: data/dataset_examples/oumi_format.jsonl
dataset_format: oumi  # Required with dataset_path: "oumi" or "alpaca"
is_multimodal: false  # Required with dataset_path

# OR - HuggingFace dataset (instead of dataset_path)
# dataset_name: "tatsu-lab/alpaca"
# split: train
# sample_count: 1000  # Limit samples to analyze (null = all)

# Output settings
output_path: "./analysis_results"  # Where to save results

# Analyzers to run
analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
      token_count: false  # Requires tokenizer_config

# Optional: Tokenizer for token counting
tokenizer_config:
  model_name: openai-community/gpt2
```

For detailed configuration options, see {doc}`analyze_config`.

## Available Analyzers

### Length Analyzer

The built-in `length` analyzer computes text length metrics:

| Metric | Description |
|--------|-------------|
| `char_count` | Number of characters in text |
| `word_count` | Number of words (space-separated tokens) |
| `sentence_count` | Number of sentences (split on `.!?`) |
| `token_count` | Number of tokens (requires tokenizer) |

**Configuration:**

```yaml
analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
      token_count: true  # Requires tokenizer_config
      include_special_tokens: true  # Include special tokens in count
```

**With tokenizer for token counting:**

```yaml
tokenizer_config:
  model_name: openai-community/gpt2

analyzers:
  - id: length
    params:
      token_count: true
```

## Working with Results

### Analysis Summary

After running analysis, access the summary statistics:

```python
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

summary = analyzer.analysis_summary

# Dataset overview
print(f"Dataset: {summary['dataset_overview']['dataset_name']}")
print(f"Samples analyzed: {summary['dataset_overview']['conversations_analyzed']}")

# Message-level statistics
for analyzer_name, metrics in summary['message_level_summary'].items():
    for metric_name, stats in metrics.items():
        print(f"{metric_name}: mean={stats['mean']}, std={stats['std']}")
```

### DataFrames

Access raw analysis data as pandas DataFrames:

```python
# Message-level metrics (one row per message)
message_df = analyzer.message_df

# Conversation-level metrics (one row per conversation)
conversation_df = analyzer.conversation_df

# Merged view
full_df = analyzer.analysis_df
```

### Querying Results

Filter results using pandas query syntax:

```python
# Find long messages
long_messages = analyzer.query("text_content_length_word_count > 10")

# Find short conversations
short_convos = analyzer.query_conversations("text_content_length_char_count < 100")
```

### Filtering Datasets

Create filtered datasets based on analysis:

```python
# Get dataset with only short messages
filtered_dataset = analyzer.filter("text_content_length_word_count < 100")

# Use filtered dataset for training
print(f"Filtered from {len(analyzer.dataset)} to {len(filtered_dataset)} samples")
```

```{note}
Filtering is only supported for map-style datasets. Streaming/iterable datasets cannot be filtered by index.
```

## Supported Dataset Types

The analyze feature works with multiple dataset formats:

| Format | Description | Example |
|--------|-------------|---------|
| **Conversation (oumi)** | Multi-turn conversations with roles | SFT, instruction-following datasets |
| **Alpaca** | Instruction/input/output format | Stanford Alpaca, many instruction datasets |
| **DPO** | Preference pairs (chosen/rejected) | Preference learning datasets |
| **KTO** | Binary feedback format | Human feedback datasets |
| **Pretraining** | Raw text | C4, The Pile, etc. |
| **HuggingFace Hub** | Any HF dataset | Loaded directly via `datasets` library |

### Analyzing Local Files

Analyze the included example datasets:

**Oumi format (multi-turn conversations):**

```bash
oumi analyze --config configs/examples/analyze/basic_analyze.yaml
```

**Alpaca format (instruction/input/output):**

```bash
oumi analyze --config configs/examples/analyze/analyze_local_dataset.yaml
```

### Analyzing HuggingFace Datasets

You can analyze any HuggingFace Hub dataset directly. Create a config file `hf_analyze.yaml`:

```yaml
dataset_source: CONFIG
dataset_name: databricks/dolly-15k
split: train
sample_count: 100
output_path: ./analysis_output/dolly
analyzers:
  - id: length
```

Then run:

```bash
oumi analyze --config hf_analyze.yaml
```

Or use the Python API:

```python
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
print(analyzer.analysis_summary)
```

## Exporting Results

### CLI Export

The CLI automatically exports results when `output_path` is set:

```bash
# Export to CSV (default)
oumi analyze --config configs/examples/analyze/basic_analyze.yaml

# Export to Parquet
oumi analyze --config configs/examples/analyze/basic_analyze.yaml --format parquet

# Export to JSON
oumi analyze --config configs/examples/analyze/basic_analyze.yaml --format json

# Override output directory
oumi analyze --config configs/examples/analyze/basic_analyze.yaml --output ./my_results
```

**Output files:**

| File | Description |
|------|-------------|
| `message_analysis.{format}` | Per-message metrics |
| `conversation_analysis.{format}` | Per-conversation aggregated metrics |
| `analysis_summary.json` | Statistical summary |

### Python API Export

```python
import json

# Export DataFrames
analyzer.message_df.to_csv("message_analysis.csv", index=False)
analyzer.conversation_df.to_parquet("conversation_analysis.parquet")

# Export summary
with open("summary.json", "w") as f:
    json.dump(analyzer.analysis_summary, f, indent=2)
```

## Example Workflows

### Analyze with Token Counting

```bash
oumi analyze --config configs/examples/analyze/analyze_with_tokens.yaml
```

This uses GPT-2 tokenizer to count tokens in addition to characters, words, and sentences.

### Custom Analysis Script

```python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams

# Analyze local dataset with token counting
config = AnalyzeConfig(
    dataset_source=DatasetSource.CONFIG,
    dataset_path="data/dataset_examples/oumi_format.jsonl",
    dataset_format="oumi",
    is_multimodal=False,
    output_path="./my_analysis",
    tokenizer_config={"model_name": "openai-community/gpt2"},
    analyzers=[
        SampleAnalyzerParams(
            id="length",
            params={
                "char_count": True,
                "word_count": True,
                "sentence_count": True,
                "token_count": True,
            }
        )
    ],
)

analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()

# Print summary
summary = analyzer.analysis_summary
print(f"Analyzed {summary['dataset_overview']['conversations_analyzed']} conversations")
print(f"Total messages: {summary['dataset_overview']['total_messages']}")

# Export to CSV
analyzer.message_df.to_csv("my_analysis/messages.csv", index=False)
```

### Filter Dataset by Length

```python
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

# Filter out very short responses (< 50 characters)
quality_dataset = analyzer.filter("text_content_length_char_count >= 50")
print(f"Kept {len(quality_dataset)} of {len(analyzer.dataset)} samples")
```

## Troubleshooting

### Common Issues

**"Dataset not found in registry"**

If you're using a HuggingFace dataset that's not registered in Oumi, it will be loaded directly from the Hub. Make sure you have internet access and the dataset name is correct.

**"Tokenizer required for token_count"**

To compute token counts, you must provide a `tokenizer_config`:

```yaml
tokenizer_config:
  model_name: openai-community/gpt2

analyzers:
  - id: length
    params:
      token_count: true
```

**"Filtering not supported for iterable datasets"**

Streaming datasets cannot be filtered by index. Use the `query()` method to get filtered indices, then process manually:

```python
# Get indices that match criteria
filtered_df = analyzer.query("text_content_length_word_count > 100")
valid_indices = filtered_df.conversation_index.unique().tolist()

# Process manually
for idx in valid_indices:
    # Your processing logic
    pass
```

## API Reference

- {py:class}`~oumi.core.configs.AnalyzeConfig` - Configuration class
- {py:class}`~oumi.core.analyze.dataset_analyzer.DatasetAnalyzer` - Main analyzer class
- {py:class}`~oumi.core.analyze.sample_analyzer.SampleAnalyzer` - Base class for analyzers
- {py:class}`~oumi.core.analyze.length_analyzer.LengthAnalyzer` - Built-in length analyzer
