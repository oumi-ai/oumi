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

Checkout the {doc}`inference_cli` for more information on how to use the CLI.

**Cloud APIs**

Remote inference engines (i.e. API based) require a `RemoteParams` object to be passed in.

The `RemoteParams` object contains the API URL and any necessary API keys. For example, here is to use Claude Sonnet 3.5:

```{testcode}
from oumi.inference import AnthropicInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = AnthropicInferenceEngine(
    model_params=ModelParams(
        model_name="claude-3-5-sonnet-20240620",
    ),
    remote_params=RemoteParams(
        api_key_env_varname="ANTHROPIC_API_KEY",
    )
)
```

**Supported Parameters**

Each inference engine supports a different set of parameters (for example, different generation parameters, or specific model kwargs).

Make sure to check the {doc}`configuration` for an exhaustive list of supported parameters, and the reference page for the specific engine you are using to find the parameters it supports.

For example, the supported parameters for the `VLLMInferenceEngine` can be found in {py:meth}`~oumi.inference.VLLMInferenceEngine.get_supported_params`.

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
# Alternatively, install all Oumi GPU dependencies, which takes care of installing a
# vLLM version compatible with your current Oumi version.
pip install oumi[gpu]
```

**Basic Usage**

```python
engine = VLLMInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )
)
```

**Tensor Parallel Inference**

For multi-GPU setups, you can leverage tensor parallelism:

```python
# Tensor parallel inference
model_params = ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
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
        model_name="meta-llama/Llama-3.2-1B-Instruct",
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

### Remote vLLM

[vLLM](https://github.com/vllm-project/vllm) can be deployed as a server, providing high-performance inference capabilities over HTTP. This section covers different deployment scenarios and configurations.

#### Server Setup

1. **Basic Server** - Suitable for development and testing:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 6864
```

2. **Multi-GPU Server** - For large models requiring multiple GPUs:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --port 6864 \
    --tensor-parallel-size 4

```

#### Client Configuration

The client can be configured with different reliability and performance options similar to any other remote engine:

```{testcode}
# Basic client with timeout and retry settings
engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct"
    ),
    remote_params=RemoteParams(
        api_url="http://localhost:6864",
        max_retries=3,      # Maximum number of retries
        num_workers=10,    # Number of parallel threads
    )
)
```

### Remote SGLang

[SGLang](https://sgl-project.github.io/) is another model server, providing high-performance LLM inference capabilities.

#### Server Setup

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 6864 \
    --disable-cuda-graph \
    --mem-fraction-static=0.99
```

Please refer to [SGLang documentation](https://sgl-project.github.io/backend/server_arguments.html) for more advanced configuration options.

#### Client Configuration

The client can be configured with different reliability and performance options similar to any other remote engines:

```{testcode}
engine = SGLangInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct"
    ),
    remote_params=RemoteParams(
        api_url="http://localhost:6864"
    )
)
```

To run inference interactively, use the `oumi infer` command with the `-i` flag.

```
oumi infer -c configs/recipes/llama3_1/inference/8b_sglang_infer.yaml -i
```

## Cloud APIs

While local inference offers control and flexibility, cloud APIs provide access to state-of-the-art models and scalable infrastructure without the need to manage your own hardware.

### Anthropic

[Claude](https://www.anthropic.com/claude) is Anthropic's advanced language model, available through their API.

**Basic Usage**

```{testcode}
from oumi.inference import AnthropicInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = AnthropicInferenceEngine(
    model_params=ModelParams(
        model_name="claude-3-5-sonnet-20240620"
    ),
    remote_params=RemoteParams(
        api_key_env_varname="ANTHROPIC_API_KEY",
    )
)
```

**Supported Models**

The Anthropic models available via this API as of late Jan'2025 are listed below. For an up-to-date list, please visit [this page](https://docs.anthropic.com/en/docs/about-claude/models).

| Anthropic Model                       | API Model Name            |
|---------------------------------------|---------------------------|
| Claude 3.5 Sonnet (most intelligent)  | claude-3-5-sonnet-latest  |
| Claude 3.5 Haiku (fastest)            | claude-3-5-haiku-latest   |
| Claude 3.0 Opus                       | claude-3-opus-latest      |
| Claude 3.0 Sonnet                     | claude-3-sonnet-20240229  |
| Claude 3.0 Haiku                      | claude-3-haiku-20240307   |

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

```{testcode}
from oumi.inference import GoogleVertexInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = GoogleVertexInferenceEngine(
    model_params=ModelParams(
        model_name="google/gemini-1.5-pro"
    ),
    remote_params=RemoteParams(
        api_url="https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi/chat/completions",
    )
)
```

**Supported Models**

The most popular Google Vertex AI models available via this API (as of late Jan'2025) are listed below. For a full list, including specialized and 3rd party models, please visit [this page](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).

| Gemini Model                          | API Model Name                   |
|---------------------------------------|----------------------------------|
| Gemini 2.0 Flash Thinking Mode        | google/gemini-2.0-flash-thinking-exp-01-21 |
| Gemini 2.0 Flash                      | google/gemini-2.0-flash-exp      |
| Gemini 1.5 Flash                      | google/gemini-1.5-flash-002      |
| Gemini 1.5 Pro                        | google/gemini-1.5-pro-002        |
| Gemini 1.0 Pro Vision                 | google/gemini-1.0-pro-vision-001 |

| Gemma Model                           | API Model Name                   |
|---------------------------------------|----------------------------------|
| Gemma 2 2B IT                         | google/gemma2-2b-it              |
| Gemma 2 9B IT                         | google/gemma2-9b-it              |
| Gemma 2 27B IT                        | google/gemma2-27b-it             |
| Code Gemma 2B                         | google/codegemma-2b              |
| Code Gemma 7B                         | google/codegemma-7b              |
| Code Gemma 7B IT                      | google/codegemma-7b-it           |


**Resources**

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs) for Google Cloud AI services

#### Gemini API

**Basic Usage**

```{testcode}
from oumi.inference import GoogleGeminiInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = GoogleGeminiInferenceEngine(
    model_params=ModelParams(
        model_name="gemini-1.5-flash"
    ),
    remote_params=RemoteParams(
        api_key_env_varname="GEMINI_API_KEY",
    )
)
```

**Supported Models**

The Gemini models available via this API as of late Jan'2025 are listed below. For an up-to-date list, please visit [this page](https://ai.google.dev/gemini-api/docs/models/gemini).

| Model Name                            | API Model Name            |
|---------------------------------------|---------------------------|
| Gemini 2.0 Flash (experimental)       | gemini-2.0-flash-exp      |
| Gemini 1.5 Flash                      | gemini-1.5-flash          |
| Gemini 1.5 Flash-8B                   | gemini-1.5-flash-8b       |
| Gemini 1.5 Pro                        | gemini-1.5-pro            |
| Gemini 1.0 Pro (deprecated)           | gemini-1.0-pro            |
| AQA                                   | aqa                       |

**Resources**

- [Gemini API Documentation](https://ai.google.dev/docs) for Gemini API details

### OpenAI

[OpenAI's models](https://platform.openai.com/), including GPT-4, represent some of the most widely used and capable AI systems available.

**Basic Usage**

```python
from oumi.inference import OpenAIInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = OpenAIInferenceEngine(
    model_params=ModelParams(
        model_name="gpt-4o-mini"
    ),
    remote_params=RemoteParams(
        api_key_env_varname="OPENAI_API_KEY",
    )
)
```

**Supported Models**

The most popular models available via the OpenAI API as of late Jan'2025 are listed below. For a full list please visit [this page](https://platform.openai.com/docs/models)

| OpenAI Model                          | API Model Name            |
|---------------------------------------|---------------------------|
| GPT 4o (flagship model)               | gpt-4o                    |
| GPT 4o mini (fast and affordable)     | gpt-4o-mini               |
| o1 (reasoning model)                  | o1                        |
| o1 mini (reasoning and affordable)    | o1-mini                   |
| GPT-4 Turbo                           | gpt-4-turbo               |
| GPT-4                                 | gpt-4                     |

**Resources**

- [OpenAI API Documentation](https://platform.openai.com/docs) for OpenAI API details

### Together

[Together](https://together.xyz) offers remote inference for 100+ models through serverless endpoints.

**Basic Usage**

```{testcode}
from oumi.inference import TogetherInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = TogetherInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct"
    ),
    remote_params=RemoteParams(
        api_key_env_varname="TOGETHER_API_KEY",
    )
)
```

The models available via this API can be found at [together.ai](https://www.together.ai/).

### Lambda Inference API

[Lambda Inference API](https://lambda.ai) enables you to use large language models (LLMs) without the need to set up a server. No limits are placed on the rate of requests.

**Basic Usage**

```{testcode}
from oumi.inference import LambdaInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = LambdaInferenceEngine(
    model_params=ModelParams(
        model_name="llama-4-scout-17b-16e-instruct"
    ),
)
```

**Supported Models**

The full list of models available via this API can be found at [docs.lambda.ai](https://docs.lambda.ai/public-cloud/lambda-inference-api/#listing-models).


**Resources**

- [Lambda AI API Documentation](https://docs.lambda.ai/public-cloud/lambda-inference-api)

### DeepSeek

[DeepSeek](https://deepseek.com) allows to access the DeepSeek models (Chat, Code, and Reasoning) through the DeepSeek AI Platform.

**Basic Usage**

```{testcode}
from oumi.inference import DeepSeekInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = DeepSeekInferenceEngine(
    model_params=ModelParams(
        model_name="deepseek-chat"
    ),
    remote_params=RemoteParams(
        api_key_env_varname="DEEPSEEK_API_KEY",
    )
)
```

**Supported Models**

The DeepSeek models available via this API as of late Jan'2025 are listed below. For an up-to-date list, please visit [this page](https://api-docs.deepseek.com/quick_start/pricing).

| DeepSeek Model                        | API Model Name            |
|---------------------------------------|---------------------------|
| DeepSeek-V3                           | deepseek-chat             |
| DeepSeek-R1 (reasoning with CoT)      | deepseek-reasoner         |

### SambaNova

[SambaNova](https://www.sambanova.ai/) offers an extreme-speed inference platform on cloud infrastructure with wide variety of models.

This service is particularly useful when you need to run open source models in a managed environment.

**Basic Usage**

```{testcode}
from oumi.inference import SambanovaInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = SambanovaInferenceEngine(
    model_params=ModelParams(
        model_name="Meta-Llama-3.1-405B-Instruct"
    ),
    remote_params=RemoteParams(
        api_key_env_varname="SAMBANOVA_API_KEY",
    )
)
```

** Reference **
- [SambaNova's Documentation](https://docs.sambanova.ai/cloud/docs/get-started/overview)

### Parasail.io

[Parasail.io](https://parasail.io) offers a cloud-native inference platform that combines the flexibility of self-hosted models with the convenience of cloud infrastructure.

This service is particularly useful when you need to run open source models in a managed environment.

**Basic Usage**

Here's how to configure Oumi for Parasail.io:

```{testcode}
from oumi.inference import ParasailInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = ParasailInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct"
    ),
    remote_params=RemoteParams(
        api_key_env_varname="PARASAIL_API_KEY",
    )
)
```

The models available via this API can be found at [docs.parasail.io](https://docs.parasail.io/).

**Resources**

- [Parasail.io Documentation](https://docs.parasail.io)

## See Also

- [Configuration Guide](configuration.md) for detailed config options
- [Common Workflows](common_workflows.md) for usage examples
