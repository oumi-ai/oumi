# Analysis Configuration

{py:class}`~oumi.core.configs.AnalyzeConfig` controls how Oumi analyzes datasets. See {doc}`analyze` for usage examples.

## Core Settings

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_name` | `str` | Conditional | `None` | Dataset name (HuggingFace Hub or registered) |
| `dataset_path` | `str` | Conditional | `None` | Path to local dataset file |
| `split` | `str` | No | `"train"` | Dataset split to analyze |
| `subset` | `str` | No | `None` | Dataset subset/config name |
| `sample_count` | `int` | No | `None` | Max samples to analyze (None = all) |

## Dataset Specification

Provide either a named dataset or local file path:

::::{tab-set}
:::{tab-item} Named Dataset

```yaml
dataset_name: "argilla/databricks-dolly-15k-curated-en"
split: train
subset: null  # Optional
```

:::
:::{tab-item} Local File

```yaml
dataset_path: data/dataset_examples/oumi_format.jsonl
is_multimodal: false  # Required
```

:::
::::

:::{tip}
You can also pass a pre-loaded dataset directly to `DatasetAnalyzer`:

```python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
analyzer = DatasetAnalyzer(config, dataset=my_dataset)
```

:::

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
is_multimodal: true
processor_name: "llava-hf/llava-1.5-7b-hf"
```

:::{note}
Multimodal datasets require a valid `processor_name`.
:::

## Example Configuration

Run the example from the Oumi repository root:

```bash
oumi analyze --config configs/examples/analyze/analyze.yaml
```

The example config at `configs/examples/analyze/analyze.yaml` demonstrates all available options with detailed comments explaining each setting.

## See Also

- {doc}`analyze` - Main analysis guide
- {py:class}`~oumi.core.configs.AnalyzeConfig` - API reference
- {py:class}`~oumi.core.configs.params.base_params.SampleAnalyzerParams` - Analyzer params
