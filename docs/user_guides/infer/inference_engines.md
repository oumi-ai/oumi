# Inference Engines

Oumi's inference API provides a unified interface for multiple inference engines through the `InferenceEngine` class.

In this guide, we'll go through each supported engine, what they are best for, and how to get started using them.


## Introduction

Before digging into specific engines, let's look at the basic patterns for initializing both local and remote inference engines.

These patterns will be consistent across all engine types, making it easy to switch between them as your needs change.

**Local Inference**

Let's start with a basic example of how to use the `VLLMInferenceEngine` to run inference on a local model.

```python
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import ModelParams

# Local inference with vLLM
engine = VLLMInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
    )
)
```

**Using the CLI**

You can also specify configuration in YAML, and use the CLI to run inference:

```bash
oumi infer --engine VLLM --model.model_name meta-llama/Llama-3.2-1B-Instruct
```

Checkout the {doc}`cli_reference` for more information on how to use the CLI.


**Cloud APIs**

Remote inference engines (i.e. API based) require a `RemoteParams` object to be passed in.

The `RemoteParams` object contains the API URL and any necessary API keys. For example, here is to use Claude Sonnet 3.5:

```python
from oumi.inference import AnthropicInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = AnthropicInferenceEngine(
    model_params=ModelParams(
        model_name="claude-3-5-sonnet-20240620",
    ),
    remote_params=RemoteParams(
        api_url="https://api.anthropic.com/v1/messages",
        api_key_env_varname="ANTHROPIC_API_KEY",
    )
)
```

**Supported Parameters**

Each inference engine supports a different set of parameters (for example, different generation parameters, or specific model kwargs).

Make sure to check the {doc}`configuration` for an exhaustive list of supported parameters, and the reference page for the specific engine you are using to find the parameters it supports.

For example, the supported parameters for the `VLLMInferenceEngine` can be found here {py:meth}`~oumi.inference.VLLMInferenceEngine.get_supported_params`.


## Local Inference

This next section covers setting up and optimizing local inference engines for running models directly on your machine, whether you're running on a laptop or a server with multiple GPUs.

Local inference is ideal for running your own fine-tuned models, and in general for development, testing, and scenarios where you need complete control over your inference environment.

### Hardware Recommendations

The following tables provide a rough estimate of the memory requirements for different model sizes using both BF16 and Q4 quantization.

The actual memory requirements might vary based on the specific quantization implementation and additional optimizations used.

Also note that Q4 quantization typically comes with some degradation in model quality, though the impact varies by model architecture and task.

**BF16 / FP16 (16-bit)**
| Model Size | GPU VRAM              | Notes |
|------------|----------------------|--------|
| 1B         | ~2 GB                | Can run on most modern GPUs |
| 3B         | ~6 GB                | Can run on mid-range GPUs |
| 7B         | ~14 GB               | Can run on consumer GPUs like RTX 3090 or RX 7900 XTX |
| 13B        | ~26 GB               | Requires high-end GPU or multiple GPUs |
| 33B        | ~66 GB               | Requires enterprise GPUs or multi-GPU setup |
| 70B        | ~140 GB              | Typically requires multiple A100s or H100s |

**Q4 (4-bit)**

| Model Size | GPU VRAM             | Notes |
|------------|----------------------|--------|
| 1B         | ~0.5 GB              | Can run on most integrated GPUs |
| 3B         | ~1.5 GB              | Can run on entry-level GPUs |
| 7B         | ~3.5 GB              | Can run on most gaming GPUs |
| 13B        | ~6.5 GB              | Can run on mid-range GPUs |
| 33B        | ~16.5 GB             | Can run on high-end consumer GPUs |
| 70B        | ~35 GB               | Can run on professional GPUs |


### vLLM Engine

[vLLM](https://github.com/vllm-project/vllm) is a high-performance inference engine that implements state-of-the-art serving techniques like PagedAttention for optimal memory usage and throughput.

vLLM is our recommended choice for production deployments on GPUs.

**Installation**

First, make sure to install the vLLM package:
```bash
pip install vllm
```

**Basic Usage**

```python
engine = VLLMInferenceEngine(
    ModelParams(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
)
```

**Tensor Parallel Inference**

For multi-GPU setups, you can leverage tensor parallelism:

```python
# Tensor parallel inference
model_params = ModelParams(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_kwargs={
            "tensor_parallel_size": 2,        # Set to number of GPUs
            "gpu_memory_utilization": 1.0,    # Memory usage
            "enable_prefix_caching": True,    # Enable prefix caching
        }
)
```

**Resources**

- [vLLM Documentation](https://vllm.readthedocs.io/en/latest/)
### LlamaCPP Engine

For scenarios where GPU resources are limited or unavailable, the [LlamaCPP engine](https://github.com/ggerganov/llama.cpp) provides an excellent alternative.

Built on the highly optimized llama.cpp library, this engine excels at CPU inference and quantized models, making it particularly suitable for edge deployment and resource-constrained environments. ls even on modest hardware.

LlamaCPP is a great choice for CPU inference and inference with quantized models.


**Installation**

```bash
pip install llama-cpp-python
```

**Basic Usage**

```python
engine = LlamaCppInferenceEngine(
    ModelParams(
        model_name="model.gguf",
        model_kwargs={
            "n_gpu_layers": 0,     # CPU only
            "n_ctx": 2048,         # Context window
            "n_batch": 512,        # Batch size
            "low_vram": True       # Memory optimization
        }
    )
)
```

**Resources**
- [llama.cpp Python Documentation](https://llama-cpp-python.readthedocs.io/en/latest/)
- [llama.cpp GitHub Project](https://github.com/ggerganov/llama.cpp)

### Native Engine

The Native engine uses HuggingFace's [🤗 Transformers](https://huggingface.co/docs/transformers/index) library directly, providing maximum compatibility and ease of use.

While it may not offer the same performance optimizations as vLLM or LlamaCPP, its simplicity and compatibility make it an excellent choice for prototyping and testing.


**Basic Usage**

```python
engine = NativeTextInferenceEngine(
    ModelParams(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "float16"
        }
    )
)
```

**4-bit Quantization**

For memory-constrained environments, 4-bit quantization is available:

```python
model_params = ModelParams(
    model_kwargs={
        "load_in_4bit": True,
    }
)
```

### Remote VLLM

[vLLM](https://github.com/vllm-project/vllm) can be deployed as a server, providing high-performance inference capabilities over HTTP. This section covers different deployment scenarios and configurations.

#### Server Setup

1. **Basic Server** - Suitable for development and testing:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000
```

2. **Multi-GPU Server** - For large models requiring multiple GPUs:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000
```


#### Client Configuration

The client can be configured with different reliability and performance options:

```python
# Basic client with timeout and retry settings
engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
    ),
    remote_params=RemoteParams(
        api_url="http://localhost:8000",
    )
)
```

## Cloud APIs

While local inference offers control and flexibility, cloud APIs provide access to state-of-the-art models and scalable infrastructure without the need to manage your own hardware.


### Anthropic

[Claude](https://www.anthropic.com/claude) is Anthropic's advanced language model, available through their API.

**Basic Usage**

```python
from oumi.inference import AnthropicInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = AnthropicInferenceEngine(
    model_params=ModelParams(
        model_name="claude-3-5-sonnet-20240620"
    ),
    remote_params=RemoteParams(
        api_url="https://api.anthropic.com/v1/messages",
        api_key_env_varname="ANTHROPIC_API_KEY",
    )
)
```

**Resources**

- [Anthropic API Documentation](https://docs.anthropic.com/en/api/getting-started)
- [Available Models](https://docs.anthropic.com/en/docs/about-claude/models)


### Google Cloud

Google Cloud provides multiple pathways for accessing their AI models, either through the Vertex AI platform or directly via the Gemini API.

#### Vertex AI

**Installation**

```bash
pip install "oumi[gcp]"
```

**Basic Usage**

```python
engine = GoogleVertexInferenceEngine(
    model_params=ModelParams(
        model_name="google/gemini-1.5-pro"
    ),
    remote_params=RemoteParams(
        api_url="https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi/chat/completions",
    )
)
```

**Resources**
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs) for Google Cloud AI services

#### Gemini API

**Basic Usage**
```python
engine = RemoteInferenceEngine(
    model_params=ModelParams(
        model_name="gemini-1.5-flash"
    ),
    remote_params=RemoteParams(
        api_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        api_key_env_varname="GEMINI_API_KEY",
    )
)
```

**Resources**
- [Gemini API Documentation](https://ai.google.dev/docs) for Gemini API details

### OpenAI

[OpenAI's models](https://platform.openai.com/), including GPT-4, represent some of the most widely used and capable AI systems available.

**Basic Usage**

```python
engine = RemoteInferenceEngine(
    model_params=ModelParams(
        model_name="gpt-4o-mini"
    ),
    remote_params=RemoteParams(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key_env_varname="OPENAI_API_KEY",
    )
)
```

**Resources**

- [OpenAI API Documentation](https://platform.openai.com/docs) for OpenAI API details

### Parasail.io

[Parasail.io](https://parasail.io) offers a cloud-native inference platform that combines the flexibility of self-hosted models with the convenience of cloud infrastructure.

This service is particularly useful when you need to run open source models in a managed environment.

**Basic Usage**

Here's how to configure Oumi for Parasail.io:

```python
from oumi.inference import RemoteInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = RemoteInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct"
    ),
    remote_params=RemoteParams(
        api_url="https://api.parasail.io/v1/chat/completions",
        api_key_env_varname="PARASAIL_API_KEY",
    )
)
```

**Resources**

- [Parasail.io Documentation](https://docs.parasail.io)

## See Also

- [Configuration Guide](configuration.md) for detailed config options
- [Common Workflows](common_workflows.md) for usage examples
