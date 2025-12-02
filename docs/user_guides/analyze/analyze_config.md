# Analysis Configuration

This page provides a detailed reference for all configuration options available in `AnalyzeConfig`.

## Configuration Reference

### Core Settings

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_source` | `DatasetSource` | Yes | - | How to load the dataset: `config` or `direct` |
| `dataset_name` | `str` | Conditional | `None` | Dataset name (HuggingFace Hub or registered) |
| `dataset_path` | `str` | Conditional | `None` | Path to local dataset file |
| `split` | `str` | No | `"train"` | Dataset split to analyze |
| `subset` | `str` | No | `None` | Dataset subset/config name |
| `sample_count` | `int` | No | `None` | Max samples to analyze (None = all) |

### Dataset Source

The `dataset_source` field determines how the dataset is loaded:

```yaml
# Load from configuration (HuggingFace Hub or registered dataset)
dataset_source: CONFIG
dataset_name: "tatsu-lab/alpaca"

# OR use direct mode in Python API
dataset_source: DIRECT
# Then pass dataset to DatasetAnalyzer constructor
```

**`CONFIG` mode:**

- Loads dataset based on `dataset_name` or `dataset_path`
- Supports HuggingFace Hub datasets
- Supports locally registered Oumi datasets

**`DIRECT` mode:**

- Dataset is passed directly to `DatasetAnalyzer.__init__()`
- Useful when you already have a dataset loaded in memory

### Dataset Specification

When `dataset_source: CONFIG`, you must provide one of:

#### Option 1: Named dataset (HuggingFace or registered)

```yaml
dataset_source: CONFIG
dataset_name: "databricks/dolly-15k"
split: train
subset: null  # Optional subset name
```

#### Option 2: Local file

```yaml
dataset_source: CONFIG
dataset_path: data/dataset_examples/oumi_format.jsonl
dataset_format: oumi  # Required: "oumi" or "alpaca"
is_multimodal: false  # Required: true or false
```

### Output Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | `"."` | Directory for output files |

```yaml
output_path: "./analysis_results"
```

When using the CLI, you can override this with `--output`:

```bash
oumi analyze --config configs/examples/analyze/basic_analyze.yaml --output /custom/path
```

### Analyzers Configuration

The `analyzers` field is a list of analyzer configurations:

```yaml
analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
      token_count: false
```

Each analyzer has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | Yes | Unique analyzer identifier (must be registered) |
| `params` | `dict` | No | Analyzer-specific parameters |

#### Built-in Analyzers

**`length` - Text Length Analyzer**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `char_count` | `bool` | `true` | Compute character count |
| `word_count` | `bool` | `true` | Compute word count |
| `sentence_count` | `bool` | `true` | Compute sentence count |
| `token_count` | `bool` | `false` | Compute token count (requires tokenizer) |
| `include_special_tokens` | `bool` | `true` | Include special tokens in token count |

```yaml
analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
      token_count: true
      include_special_tokens: true
```

### Tokenizer Configuration

Required when using `token_count: true`:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | `str` | Yes | HuggingFace model/tokenizer name |
| `tokenizer_kwargs` | `dict` | No | Additional tokenizer arguments |
| `trust_remote_code` | `bool` | No | Allow remote code execution |

```yaml
tokenizer_config:
  model_name: openai-community/gpt2
  tokenizer_kwargs:
    use_fast: true
  trust_remote_code: false
```

### Multimodal Settings

For vision-language datasets:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_multimodal` | `bool` | `None` | Whether dataset is multimodal |
| `processor_name` | `str` | `None` | Processor name for VL datasets |
| `processor_kwargs` | `dict` | `{}` | Processor arguments |
| `trust_remote_code` | `bool` | `false` | Allow remote code |

```yaml
dataset_path: "/path/to/vl_data.jsonl"
dataset_format: oumi
is_multimodal: true
processor_name: "llava-hf/llava-1.5-7b-hf"
trust_remote_code: true
```

## Complete Example Configurations

All examples below can be run directly from the Oumi repository root.

### Basic Local Dataset Analysis

```yaml
# configs/examples/analyze/basic_analyze.yaml
# Run: oumi analyze --config configs/examples/analyze/basic_analyze.yaml
dataset_source: CONFIG
dataset_path: data/dataset_examples/oumi_format.jsonl
dataset_format: oumi
is_multimodal: false
output_path: ./analysis_output/basic

analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
```

### Analysis with Token Counting

```yaml
# configs/examples/analyze/analyze_with_tokens.yaml
# Run: oumi analyze --config configs/examples/analyze/analyze_with_tokens.yaml
dataset_source: CONFIG
dataset_path: data/dataset_examples/oumi_format.jsonl
dataset_format: oumi
is_multimodal: false
output_path: ./analysis_output/with_tokens

tokenizer_config:
  model_name: openai-community/gpt2

analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
      token_count: true
      include_special_tokens: true
```

### Alpaca Format Dataset

```yaml
# configs/examples/analyze/analyze_local_dataset.yaml
# Run: oumi analyze --config configs/examples/analyze/analyze_local_dataset.yaml
dataset_source: CONFIG
dataset_path: data/dataset_examples/alpaca_format.jsonl
dataset_format: alpaca
is_multimodal: false
output_path: ./analysis_output/alpaca

analyzers:
  - id: length
```

### HuggingFace Hub Dataset

```yaml
# Create this file as hf_analyze.yaml and run: oumi analyze --config hf_analyze.yaml
dataset_source: CONFIG
dataset_name: databricks/dolly-15k
split: train
sample_count: 100
output_path: ./analysis_output/dolly

analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
      sentence_count: true
```

## Validation Rules

The configuration is validated at initialization with these rules:

1. **`dataset_source` is required** and must be `CONFIG` or `DIRECT`

2. **When `dataset_source: CONFIG`:**
   - Either `dataset_name` or `dataset_path` must be provided

3. **When `dataset_path` is provided:**
   - `dataset_format` is required (`oumi` or `alpaca`)
   - `is_multimodal` is required (`true` or `false`)

4. **When `is_multimodal: true`:**
   - `dataset_format` must be `oumi`
   - `processor_name` is required

5. **`sample_count`** must be greater than 0 if specified

6. **Analyzer IDs** must be unique (no duplicates)

7. **Token counting** requires `tokenizer_config` with `model_name`

## Python API

```python
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams

# Create configuration for local dataset
config = AnalyzeConfig(
    dataset_source=DatasetSource.CONFIG,
    dataset_path="data/dataset_examples/oumi_format.jsonl",
    dataset_format="oumi",
    is_multimodal=False,
    output_path="./results",
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

# Or load from YAML
config = AnalyzeConfig.from_yaml("configs/examples/analyze/basic_analyze.yaml")

# Validate (automatic on creation, but can be explicit)
config.finalize_and_validate()
```

## CLI Options

The `oumi analyze` command accepts these options:

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to YAML configuration file (required) |
| `--output` | `-o` | Override output directory |
| `--format` | `-f` | Output format: `csv`, `json`, or `parquet` |
| `--verbose` | `-v` | Enable verbose logging |
| `--log-level` | `-log` | Set log level: DEBUG, INFO, WARNING, ERROR |

```bash
# Basic usage with included example config
oumi analyze -c configs/examples/analyze/basic_analyze.yaml

# With all options
oumi analyze \
  --config configs/examples/analyze/basic_analyze.yaml \
  --output ./my_results \
  --format parquet \
  --verbose \
  --log-level DEBUG
```

## See Also

- {doc}`analyze` - Main analysis guide
- {py:class}`~oumi.core.configs.AnalyzeConfig` - API reference
- {py:class}`~oumi.core.configs.params.base_params.SampleAnalyzerParams` - Analyzer params
