# Analysis Configuration

{py:class}`~oumi.core.configs.AnalyzeConfig` controls how Oumi analyzes datasets. See {doc}`analyze` for usage examples.

## Core Settings

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_source` | `DatasetSource` | Yes | - | How to load the dataset: `config` or `direct` |
| `dataset_name` | `str` | Conditional | `None` | Dataset name (HuggingFace Hub or registered) |
| `dataset_path` | `str` | Conditional | `None` | Path to local dataset file |
| `split` | `str` | No | `"train"` | Dataset split to analyze |
| `subset` | `str` | No | `None` | Dataset subset/config name |
| `sample_count` | `int` | No | `None` | Max samples to analyze (None = all) |

## Dataset Source

::::{tab-set-code}
:::{code-block} yaml

# CONFIG mode: load from HuggingFace Hub or registered dataset

dataset_source: CONFIG
dataset_name: "tatsu-lab/alpaca"
:::
:::{code-block} python

# DIRECT mode: pass dataset to DatasetAnalyzer

from oumi.analyze import DatasetAnalyzer
analyzer = DatasetAnalyzer(config, dataset=my_dataset)
:::
::::

**`CONFIG` mode** — Loads dataset from `dataset_name` (HuggingFace Hub or registered) or `dataset_path` (local file).

**`DIRECT` mode** — Pass a dataset already loaded in memory to `DatasetAnalyzer.__init__()`.

### Dataset Specification

When using `CONFIG` mode, provide either a named dataset or local file path:

::::{tab-set}
:::{tab-item} Named Dataset

```yaml
dataset_source: CONFIG
dataset_name: "databricks/dolly-15k"
split: train
subset: null  # Optional
```

:::
:::{tab-item} Local File

```yaml
dataset_source: CONFIG
dataset_path: data/dataset_examples/oumi_format.jsonl
dataset_format: oumi  # Required: "oumi" or "alpaca"
is_multimodal: false  # Required
```

:::
::::

## Output Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | `"."` | Directory for output files |

::::{tab-set-code}
:::{code-block} yaml
output_path: "./analysis_results"
:::
:::{code-block} bash
oumi analyze --config config.yaml --output /custom/path
:::
::::

## Analyzers

Configure analyzers as a list with `id` and optional `params`:

```yaml
analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | Yes | Analyzer identifier (must be registered) |
| `params` | `dict` | No | Analyzer-specific parameters |

### `length` Analyzer

Computes text length metrics:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `char_count` | `bool` | `true` | Character count |
| `word_count` | `bool` | `true` | Word count |
| `sentence_count` | `bool` | `true` | Sentence count |
| `token_count` | `bool` | `false` | Token count (requires tokenizer) |
| `include_special_tokens` | `bool` | `true` | Include special tokens in count |

## Tokenizer Configuration

Required when `token_count: true`:

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
```

## Multimodal Settings

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
```

:::{note}
Multimodal datasets require `dataset_format: oumi` and a valid `processor_name`.
:::

## Example Configurations

Run examples from the Oumi repository root.

````{dropdown} Basic Local Dataset
```yaml
# configs/examples/analyze/basic_analyze.yaml
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
````

````{dropdown} With Token Counting
```yaml
# configs/examples/analyze/analyze_with_tokens.yaml
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
      token_count: true
      include_special_tokens: true
```
````

````{dropdown} HuggingFace Hub Dataset
```yaml
dataset_source: CONFIG
dataset_name: databricks/dolly-15k
split: train
sample_count: 100
output_path: ./analysis_output/dolly

analyzers:
  - id: length
```
````

## Validation Rules

Oumi validates configuration at initialization:

| Rule | Requirement |
|------|-------------|
| `dataset_source` | Required: `CONFIG` or `DIRECT` |
| `CONFIG` mode | Requires `dataset_name` or `dataset_path` |
| Local file (`dataset_path`) | Requires `dataset_format` and `is_multimodal` |
| Multimodal | Requires `dataset_format: oumi` and `processor_name` |
| `sample_count` | Must be > 0 if specified |
| Analyzer IDs | Must be unique |
| Token counting | Requires `tokenizer_config.model_name` |

## CLI Reference

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to YAML config (required) |
| `--output` | `-o` | Override output directory |
| `--format` | `-f` | Output format: `csv`, `json`, `parquet` |
| `--verbose` | `-v` | Enable verbose logging |
| `--log-level` | `-log` | Log level: DEBUG, INFO, WARNING, ERROR |

```bash
oumi analyze -c configs/examples/analyze/basic_analyze.yaml \
  --output ./my_results --format parquet
```

## Python API

```python
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams

# Create configuration
config = AnalyzeConfig(
    dataset_source=DatasetSource.CONFIG,
    dataset_path="data/dataset_examples/oumi_format.jsonl",
    dataset_format="oumi",
    is_multimodal=False,
    analyzers=[
        SampleAnalyzerParams(id="length", params={"char_count": True})
    ],
)

# Or load from YAML
config = AnalyzeConfig.from_yaml("configs/examples/analyze/basic_analyze.yaml")
```

## See Also

- {doc}`analyze` - Main analysis guide
- {py:class}`~oumi.core.configs.AnalyzeConfig` - API reference
- {py:class}`~oumi.core.configs.params.base_params.SampleAnalyzerParams` - Analyzer params
