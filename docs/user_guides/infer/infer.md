# Inference

```{toctree}
:maxdepth: 2
:caption: Inference
:hidden:

inference_engines
vllm_gcp
```

## Quick Start

To run local interactive inference using the Oumi CLI:

```{code-block} bash
oumi infer -i -c configs/recipes/llama3_2/inference/1b_infer.yaml
```

## Inference Configuration

Create an inference configuration file (e.g., `infer_config.yaml`) with the following structure.

For a complete list of configuration options, refer to the {py:class}`~oumi.core.configs.InferenceConfig` class:

```{code-block} yaml
model:
  model_name: "meta-llama/Meta-Llama-3.1-8B"

generation:
  max_new_tokens: 100
  temperature: 0.7
  top_p: 0.9
  batch_size: 1

engine: "VLLM"  # or "LLAMACPP", "NATIVE", etc.
```

```{note}
Adjust the parameters according to your specific inference needs. For a complete list of configuration options, refer to the {doc}`Configuration Guide <../../get_started/configuration>` and {py:class}`~oumi.core.configs.InferenceConfig` class.
```

## Input Data

Oumi supports several input formats for inference:

1. JSONL files

- Prepare a JSONL file with your inputs, where each line is a JSON object containing your input data.
- See {doc}`../../datasets/local_datasets` for more details.

2. Interactive console input

- To run inference interactively, use the `oumi infer` command with the `-i` flag.

```{code-block} bash
oumi infer -c infer_config.yaml -i
```

## Supported Inference Engines

Oumi supports multiple inference engines:

```{include} ../../api/summary/inference_engines.md
```

```{seealso}
For detailed information on each inference engine, see the {doc}`Inference Engines documentation <inference_engines>`.
```

## Advanced Topics

### Inference with Quantized Models

```{code-block} yaml
model:
  model_name: "model.gguf"

engine: "llamacpp"

generation:
  temperature: 0.7
  batch_size: 1
```

```{warning}
Ensure the selected inference engine supports the specific quantization method used in your model.
```

### Multi-modal Inference

For models that support multi-modal inputs (e.g., text and images):

```python
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import InferenceConfig, ModelParams

vllmModelParams = ModelParams(model_name="llava-hf/llava-1.5-7b-hf")
engine = VLLMInferenceEngine(vllmModelParams)
input_conversation = [] # Add your inputs here
inference_config = InferenceConfig()
outputConversations = engine.infer_online(input=input_conversation, inference_config=inference_config)
```

### Distributed Inference

For large-scale inference across multiple GPUs or machines, see the following tutorials
for inference with Llama 3.1 70B on {doc}`GCP <vllm_gcp>` and
{doc}`Polaris <vllm_polaris>`.

## Troubleshooting

For more help, consult the {doc}`troubleshooting guide <../../faq/troubleshooting>` or open an issue on the [Oumi GitHub repository](https://github.com/oumi-ai/oumi/issues).
