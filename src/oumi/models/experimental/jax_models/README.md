<!--
Copyright 2025 - Oumi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# JAX Integration for High-Performance Large Language Model Inference

**⚠️ Experimental Feature**: This integration is currently under active development and should be considered experimental. The implementation is based on open-source model examples from Google's [JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples) repository.

## Introduction

This document describes Oumi's integration with [JAX](https://jax.readthedocs.io/), Google's machine learning framework for high-performance numerical computing. Our implementation leverages minimal yet performant LLM implementations from the [JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples) repository, bringing research-grade JAX optimizations to Oumi's inference pipeline.

## What is JAX?

[JAX](https://jax.readthedocs.io/) is a Python library developed by Google Research that provides:

- **XLA Compilation**: Just-in-time compilation of Python functions to optimized kernels
- **Automatic Differentiation**: Forward and reverse-mode autodiff for gradient computation
- **Vectorization**: Automatic batching and SIMD operations
- **Parallelization**: Built-in support for multi-device and distributed computing

JAX excels at large language model inference through XLA optimization, advanced sharding capabilities, and unified hardware support across CPU, GPU, and TPU.

## What is JAX LLM Examples?

The [JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples) project provides:

> "Minimal yet performant LLM examples in pure JAX"

This Google-maintained repository contains reference implementations of state-of-the-art language models, focusing on:

- **Pure JAX implementations** - No PyTorch dependencies
- **Performance optimization** - Designed for production inference speeds
- **Multi-device support** - Tensor parallelism and distributed computation
- **Research-grade quality** - Implements latest architectural innovations

## Goals and Current Status

### Project Goals

The JAX LLM Examples project aims to provide:

1. **Concise, self-contained codebases** for major LLM architectures
2. **Competitive inference performance** with optimized kernels
3. **Accessible starting points** for JAX-based LLM development
4. **Multi-host cluster support** for large-scale deployment

### Current Achievements

Based on the upstream repository, JAX LLM Examples has achieved:

#### **Llama 3 Implementation**
- Pure JAX implementation with Grouped Query Attention (GQA)
- Tensor parallelism via JAX's `shard_map`
- INT8 quantization
- Custom TPU ragged decode attention kernel

#### **Llama 4 Implementation**
- Pure JAX implementation with MoE (Mixture of Experts) and gated routing
- Two model variants: Scout and Maverick
- NoPE (No Position Embedding) layers at configurable intervals
- QK normalization support
- Expert and tensor parallelism via `shard_map`
- INT8 quantization

#### **DeepSeek R1 Implementation**
- Pure JAX implementation of DeepSeek V3 with Multi-head Latent Attention (MLA)
- Expert and tensor parallelism using `shard_map`
- INT8 quantization support
- Custom "ragged dot" kernel for efficient MoE processing

#### **Qwen 3 Implementation**
- Pure JAX implementation with MoE and dense model variants
- Expert and tensor parallelism support
- Designed for multi-host deployment (235B parameters)
- INT8 quantization

#### **Kimi K2 Implementation**
- Pure JAX implementation with Multi-head Latent Attention (MLA)
- 1T parameter MoE architecture
- Expert and tensor parallelism via `shard_map`
- INT8 quantization

#### **GPT OSS Implementation**
- Pure JAX implementation with Mixture of Experts
- Sliding window attention (configurable per layer)
- Custom gated activation with dynamic scaling
- Expert and tensor parallelism

#### **Nemotron 3 Implementation**
- Hybrid Mamba-Transformer architecture with SSM (State Space Model) layers
- Mixed layer types: Mamba, Attention, MoE, and MLP
- Mamba cache with SSM state and convolution states
- Expert and tensor parallelism

### Current Limitations

The upstream project has noted limitations:

- **Primary TPU focus**: Most testing and optimization on TPU hardware
- **GPU support in progress**: Full GPU optimization still being developed
- **Research-oriented**: Optimized for performance research rather than production deployment

## Oumi's JAX Integration

### Our Implementation

Oumi has integrated these JAX LLM Examples into our inference pipeline by:

1. **Vendoring implementations**: All model families synced from upstream jax-llm-examples
2. **Unified interface**: Integrated through `JAXInferenceEngine` following Oumi's `BaseInferenceEngine` contract
3. **Proper prefill/decode loops**: Each model uses its upstream-verified inference pattern
4. **Test coverage**: Unit and integration test suites
5. **Configuration system**: YAML-based configuration following Oumi patterns

### Supported Model Families

| Model Family | Source Implementation | Key Features | Status |
|--------------|----------------------|--------------|--------|
| **Llama3** | jax-llm-examples/llama3 | GQA, tensor parallelism, INT8 | Integrated |
| **Llama4** | jax-llm-examples/llama4 | MoE, NoPE layers, QK norm | Integrated |
| **DeepSeek R1** | jax-llm-examples/deepseek_r1_jax | MLA attention, MoE routing | Integrated |
| **Qwen3** | jax-llm-examples/qwen3 | MLA attention, splash attention | Integrated |
| **Kimi K2** | jax-llm-examples/kimi_k2 | 1T params, MLA attention | Integrated |
| **GPT OSS** | jax-llm-examples/gpt_oss | Sliding attention, MoE | Integrated |
| **Nemotron 3** | jax-llm-examples/nemotron3 | Hybrid Mamba-Transformer | Integrated |

## Installation and Usage

### Prerequisites

Install JAX support for Oumi:

```bash
# Install JAX dependencies (includes both CPU and CUDA 12 GPU support)
pip install "oumi[jax]"

# For TPU inference
pip install "oumi[jax]"  # Includes TPU utilities
# Then install JAX TPU backend:
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Note**: The JAX installation includes both CPU and CUDA 12 GPU support by default. JAX will automatically use the best available device (GPU if available, otherwise CPU). For other CUDA versions, see the [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html).

### Basic Usage

```python
from oumi.inference import JAXInferenceEngine
from oumi.core.configs import ModelParams, GenerationParams

# Initialize JAX engine
model_params = ModelParams(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype_str="bfloat16",
    trust_remote_code=True
)

engine = JAXInferenceEngine(model_params)

# Generate responses
conversations = engine.infer(
    conversations=[conversation],
    generation_params=GenerationParams(max_new_tokens=512)
)
```

### Configuration Examples

```yaml
# Basic JAX inference
model:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  torch_dtype_str: "bfloat16"
  load_pretrained_weights: true

generation:
  max_new_tokens: 512
  temperature: 0.7

engine: JAX
```

See `configs/examples/jax_inference/` for complete examples.

## Technical Details

### Key Optimizations from JAX LLM Examples

1. **Memory Bandwidth Focus**: Prioritizes HBM utilization over raw compute
2. **Fully-Replicated Activations**: Reduces communication for low-latency decoding
3. **Custom Kernels**: Specialized attention and MoE kernels
4. **Sharding Strategies**: Minimizes inter-device communication overhead

### Multi-Device Support

JAX LLM Examples implements distributed computation through:

- **Tensor Parallelism**: Model parameters sharded across devices
- **Expert Parallelism**: MoE experts distributed across devices
- **Multi-Host Clusters**: Scaling across multiple machines

### Quantization

Simple INT8 quantization is implemented across all models for:

- Reduced memory usage
- Maintained inference speed
- Minimal quality degradation

## Relationship to Upstream

The model code under each architecture directory (e.g., `llama3/llama3_jax/model.py`)
is **vendored directly from upstream** [jax-llm-examples](https://github.com/jax-ml/jax-llm-examples)
with formatting changes (Black style, import ordering). These files retain the
original `Copyright 2025 The JAX Authors` license header.

Oumi-authored integration code (inference engine, registry, manager, CLI, tests) uses
the `Copyright 2025 - Oumi` header and follows Oumi's code standards.

### What We Utilize from Upstream

- All 7 model implementations (identical to upstream)
- Checkpoint conversion utilities and download scripts
- INT8 quantization support
- Custom TPU kernels (ragged attention, ragged dot)
- Model-specific optimizations (MLA, MoE, GQA, etc.)

### What We Intentionally Defer

- **Serving infrastructure** (`serving/` directory): Continuous batching, KV cache
  management, prefix caching, HTTP server. Oumi has its own serving patterns.
- **TPU toolkit** (`misc/tpu_toolkit.sh`): Operational tooling for cluster setup.
- **Multi-host distributed serving**: Requires infrastructure-level changes.

## Scripts

### Verify All Models (No Download Needed)

```bash
# Verify all 7 model architectures work with random weights:
python scripts/examples/jax_verify_models.py

# Verify a specific model:
python scripts/examples/jax_verify_models.py --model llama3

# Verbose output:
python scripts/examples/jax_verify_models.py --verbose
```

### End-to-End Demo (Download + Convert + Infer)

```bash
# Quick demo with smallest public model (Qwen3 0.6B):
python scripts/examples/jax_models_demo.py --quick

# Specific model:
python scripts/examples/jax_models_demo.py --model qwen3-0.6b

# List available models:
python scripts/examples/jax_models_demo.py --list
```

### CLI Interface

```bash
# List models:
python -m oumi.models.experimental.jax_models list

# Recommend a model:
python -m oumi.models.experimental.jax_models recommend --max-size-gb 5

# Download, convert, and run:
python -m oumi.models.experimental.jax_models run qwen3-0.6b
```

## Development and Testing

### Test Suite

```bash
# Run JAX unit tests
pytest tests/unit/inference/test_jax_inference_engine.py -v

# Run integration tests (all models)
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
pytest tests/integration/jax_models/ -v

# Run model-level tests
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
pytest src/oumi/models/experimental/jax_models/*/tests/ -v

# Run only JAX tests across the codebase
pytest -m jax -v
```

## Known Limitations

- **Inference only**: No training support
- **No batching**: Conversations are processed one at a time (no batch parallelism)
- **Limited generation parameters**: fewer params supported vs other engines
- **No serving**: No continuous batching, prefix caching, or HTTP serving layer
- **No streaming**: Responses are generated fully before returning

## Learn More

### Resources

- **[JAX Documentation](https://jax.readthedocs.io/)** - Official JAX documentation
- **[JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples)** - Source repository for our implementations
- **[JAX Scaling Book](https://jax-ml.github.io/scaling-book/)** - Comprehensive guide to scaling with JAX

## Contributing

We welcome contributions to improve the JAX integration! Please see our [Contributing Guide](../../../CONTRIBUTING.md) for guidelines.

When contributing:

1. Test changes on available hardware
2. Follow existing patterns from other Oumi inference engines
3. Add appropriate test coverage
4. Update documentation for new features

---

*This integration is based on the open-source [JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples) project. All technical details and performance claims are derived from the upstream repository's documentation and should be verified independently for your specific use case.*
