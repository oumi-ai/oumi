# Inference Engine Comparison

This guide compares the different inference engines available in Oumi to help you choose the right one for your use case.

## Quick Comparison

| Engine | Best For | GPU Required | Throughput | Latency | Ease of Use |
|--------|----------|--------------|------------|---------|-------------|
| **vLLM** | Production GPU deployment | Yes | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Native** | Prototyping & debugging | Optional | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **LlamaCPP** | CPU & edge devices | No | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Remote vLLM** | Distributed serving | Server-side | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **SGLang** | High-throughput serving | Yes | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Detailed Comparison

### vLLM (Local)

**Best for**: Production deployments on GPU servers

```python
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import ModelParams

engine = VLLMInferenceEngine(
    ModelParams(model_name="meta-llama/Llama-3.1-8B-Instruct")
)
```

**Pros**:

- ✅ Highest throughput with PagedAttention
- ✅ Efficient memory management
- ✅ Continuous batching for concurrent requests
- ✅ Tensor parallelism for large models
- ✅ LoRA adapter support
- ✅ Prefix caching

**Cons**:

- ❌ Requires NVIDIA GPU
- ❌ Larger memory footprint than quantized alternatives
- ❌ Additional installation required (`pip install vllm`)

**When to use**:

- Serving models in production
- High-throughput batch processing
- Multi-GPU deployments
- When latency and throughput matter

---

### Native (HuggingFace Transformers)

**Best for**: Development, prototyping, and maximum compatibility

```python
from oumi.inference import NativeTextInferenceEngine
from oumi.core.configs import ModelParams

engine = NativeTextInferenceEngine(
    ModelParams(model_name="meta-llama/Llama-3.1-8B-Instruct")
)
```

**Pros**:

- ✅ Maximum model compatibility
- ✅ Works with CPU or GPU
- ✅ No additional dependencies
- ✅ Easy to debug and customize
- ✅ Supports all HuggingFace features
- ✅ 4-bit/8-bit quantization support

**Cons**:

- ❌ Lower throughput than vLLM
- ❌ No continuous batching
- ❌ Less optimized memory usage

**When to use**:

- Quick prototyping and testing
- Debugging model issues
- Using models not supported by other engines
- When you need maximum flexibility

---

### LlamaCPP

**Best for**: CPU inference, edge devices, and resource-constrained environments

```python
from oumi.inference import LlamaCppInferenceEngine
from oumi.core.configs import ModelParams

engine = LlamaCppInferenceEngine(
    ModelParams(
        model_name="model.gguf",
        model_kwargs={
            "n_gpu_layers": 0,  # CPU only
            "n_ctx": 2048,
        }
    )
)
```

**Pros**:

- ✅ Excellent CPU performance
- ✅ Runs on machines without GPUs
- ✅ Native quantization support (GGUF format)
- ✅ Low memory footprint
- ✅ Runs on edge devices (Raspberry Pi, etc.)
- ✅ Metal support for Apple Silicon

**Cons**:

- ❌ Requires GGUF model format
- ❌ Lower throughput than GPU engines
- ❌ Limited model architecture support

**When to use**:

- No GPU available
- Edge deployment
- Running on laptops or consumer hardware
- Using quantized models (Q4, Q8)

---

### Remote vLLM

**Best for**: Distributed deployments with a central model server

```python
from oumi.inference import RemoteVLLMInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(model_name="meta-llama/Llama-3.1-8B-Instruct"),
    remote_params=RemoteParams(
        api_url="http://your-server:8000",
        max_retries=3,
    )
)
```

**Pros**:

- ✅ Separate compute from application logic
- ✅ Share one model across multiple clients
- ✅ Scale server independently
- ✅ OpenAI-compatible API
- ✅ LoRA adapter support

**Cons**:

- ❌ Network latency overhead
- ❌ Requires server setup and management
- ❌ Additional infrastructure complexity

**When to use**:

- Multiple applications sharing one model
- Separating inference from application tier
- Team environments with shared resources
- Kubernetes/microservices deployments

---

### SGLang

**Best for**: High-throughput serving with advanced optimizations

```python
from oumi.inference import SGLangInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = SGLangInferenceEngine(
    model_params=ModelParams(model_name="meta-llama/Llama-3.1-8B-Instruct"),
    remote_params=RemoteParams(api_url="http://localhost:6864")
)
```

**Pros**:

- ✅ RadixAttention for efficient prefix caching
- ✅ High throughput for structured generation
- ✅ Optimized for long conversations
- ✅ Good for constrained decoding

**Cons**:

- ❌ Requires server setup
- ❌ Smaller community than vLLM
- ❌ More complex configuration

**When to use**:

- Structured output generation (JSON, code)
- Long multi-turn conversations
- When prefix caching provides significant benefit
- High-throughput serving with complex prompts

## Decision Flowchart

```text
Start
  │
  ├─ Do you need a cloud API (OpenAI, Anthropic, etc.)?
  │   └─ Yes → Use Cloud API engines
  │
  ├─ Do you have a GPU?
  │   ├─ No → Use LlamaCPP
  │   │
  │   └─ Yes → Continue...
  │       │
  │       ├─ Is this for production?
  │       │   ├─ Yes → Use vLLM (local) or Remote vLLM
  │       │   └─ No → Use Native for prototyping
  │       │
  │       ├─ Do you need distributed serving?
  │       │   └─ Yes → Use Remote vLLM or SGLang
  │       │
  │       └─ Is throughput critical?
  │           └─ Yes → Use vLLM or SGLang
  │
  └─ Default: Start with Native, upgrade to vLLM for production
```

## Memory Requirements

Approximate GPU memory needed for different model sizes:

### BF16 (Full Precision)

| Model Size | vLLM | Native | Notes |
|------------|------|--------|-------|
| 1B | ~4 GB | ~4 GB | Fits on most GPUs |
| 7B | ~16 GB | ~16 GB | RTX 3090/4090 |
| 13B | ~28 GB | ~30 GB | A100 40GB |
| 70B | ~150 GB | ~150 GB | Multiple A100s |

### Quantized (4-bit)

| Model Size | LlamaCPP (Q4) | Native (4-bit) | Notes |
|------------|---------------|----------------|-------|
| 1B | ~1 GB | ~2 GB | Any GPU |
| 7B | ~4 GB | ~6 GB | RTX 3060+ |
| 13B | ~8 GB | ~10 GB | RTX 3080+ |
| 70B | ~40 GB | ~45 GB | A100 40GB |

## Feature Comparison

| Feature | vLLM | Native | LlamaCPP | Remote vLLM | SGLang |
|---------|------|--------|----------|-------------|--------|
| Continuous batching | ✅ | ❌ | ❌ | ✅ | ✅ |
| Tensor parallelism | ✅ | ✅ | ❌ | ✅ | ✅ |
| LoRA adapters | ✅ | ✅ | ✅ | ✅ | ✅ |
| 4-bit quantization | ❌ | ✅ | ✅ | ❌ | ❌ |
| GGUF format | ❌ | ❌ | ✅ | ❌ | ❌ |
| Prefix caching | ✅ | ❌ | ❌ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vision models | ✅ | ✅ | ✅ | ✅ | ✅ |
| CPU support | ❌ | ✅ | ✅ | N/A | ❌ |

## Performance Benchmarks

Typical performance characteristics (results vary by hardware and model):

| Engine | Tokens/sec (7B model) | Time to First Token |
|--------|----------------------|---------------------|
| vLLM (A100) | 100-150 | 50-100ms |
| Native (A100) | 30-50 | 100-200ms |
| LlamaCPP (CPU) | 10-30 | 200-500ms |
| LlamaCPP (GPU) | 50-80 | 100-200ms |
| Remote vLLM | 80-130 | 100-200ms (+ network) |
| SGLang | 100-150 | 50-100ms |

```{note}
These are approximate values. Actual performance depends on hardware, model, batch size, and generation parameters.
```

## Migration Guide

### From Native to vLLM

```python
# Before (Native)
from oumi.inference import NativeTextInferenceEngine
engine = NativeTextInferenceEngine(model_params)

# After (vLLM)
from oumi.inference import VLLMInferenceEngine
engine = VLLMInferenceEngine(model_params)
# API remains the same!
```

### From Local to Remote

```python
# Before (Local vLLM)
from oumi.inference import VLLMInferenceEngine
engine = VLLMInferenceEngine(model_params)

# After (Remote vLLM)
from oumi.inference import RemoteVLLMInferenceEngine
engine = RemoteVLLMInferenceEngine(
    model_params=model_params,
    remote_params=RemoteParams(api_url="http://server:8000")
)
# API remains the same!
```

## See Also

- {doc}`inference_engines` - Detailed engine documentation
- {doc}`configuration` - Configuration options
- {doc}`common_workflows` - Common inference patterns
- {doc}`/faq/inference` - Inference FAQ
