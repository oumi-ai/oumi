# Local Inference

Local inference is ideal for running your own fine-tuned models, and in general development, testing, and scenarios where you need complete control over your inference environment.

This guide covers setting up and optimizing local inference engines for running models directly on your machine, whether you're running on a laptop or a server with multiple GPUs.

## Quick Setup

Before diving into specific engines, let's start with a basic setup. Choose the appropriate installation based on your intended inference engine:

1. Install dependencies:

```bash
# For VLLM engine
pip install vllm

# For SGlang engine
pip install sglang

# For LlamaCPP engine
pip install llama-cpp-python

# For Native engine, all the dependencies are installed with the oumi package
```

2. Basic usage:

```python
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import ModelParams

engine = VLLMInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        device_map="auto"
    )
)
```

## Engine Selection Guide

Oumi integrates with multiple inference engines, each optimized for different use cases and hardware configurations.

Below, we'll explore each engine's strengths and provide detailed configuration examples.

### VLLM Engine

[vLLM](https://github.com/vllm-project/vllm) is a high-performance inference engine that implements state-of-the-art serving techniques like PagedAttention for optimal memory usage and throughput.

Best for:

- Production deployments
- High throughput requirements
- GPU inference
- Memory optimization


```python
# High-performance configuration
engine = VLLMInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        model_kwargs={
            "tensor_parallel_size": 2,        # Multi-GPU
            "gpu_memory_utilization": 0.9,    # Memory usage
            "max_num_batched_tokens": 8192,   # Batch size
            "max_num_seqs": 256,             # Concurrent sequences
            "block_size": 16                 # KV cache block size
        }
    )
)
```

For multi-GPU setups, you can leverage tensor parallelism:

```python
# Tensor parallel inference
model_params = ModelParams(
    model_kwargs={
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 2
    }
)
```

To optimize memory usage, enable KV cache:

```python
# Enable KV cache
model_params = ModelParams(
    model_kwargs={
        "use_cache": True,
        "max_cache_size": "20GiB"
    }
)
```

### LlamaCPP Engine

[llama.cpp](https://github.com/ggerganov/llama.cpp) is a lightweight inference engine that excels at CPU inference and quantized models. It's particularly useful for edge deployment and resource-constrained environments.

Best for:

- CPU inference
- Edge deployment
- Quantized models
- Limited resources


```python
# CPU optimized configuration
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

### Native Engine

The Native engine uses HuggingFace's [Transformers](https://huggingface.co/docs/transformers/index) library directly, providing maximum compatibility and ease of use.

Best for:

- Development and testing
- Custom models
- Simple use cases
- Debugging

```python
# Basic configuration
engine = NativeTextInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "float16"
        }
    )
)
```

For memory-constrained environments, 4-bit quantization is available:

```python
# 4-bit quantization
model_params = ModelParams(
    model_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16"
    }
)
```

## Hardware Requirements

Before deploying models, ensure your hardware meets the minimum requirements. These requirements vary based on model size and quantization:

| Model Size | GPU Memory | CPU Memory | Storage |
|------------|------------|------------|----------|
| 7B         | 8GB        | 16GB       | 15GB     |
| 13B        | 16GB       | 32GB       | 30GB     |
| 70B        | 80GB       | 128GB      | 140GB    |

Note: These requirements assume using FP16 precision. Using quantization (4-bit, 8-bit) can significantly reduce memory requirements.

## See Also

- {doc}`configuration` for detailed config options
- {doc}`common_workflows` for usage examples
- {doc}`remote_inference` for distributed inference setups
- [vLLM Documentation](https://vllm.readthedocs.io/) for advanced vLLM configurations
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp/blob/master/README.md) for detailed CPU inference options
