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
- ✅ Pure JAX implementation with Grouped Query Attention (GQA)
- ✅ Tensor parallelism via JAX's `shard_map`
- ✅ Simple INT8 quantization
- ✅ **Verified performance**: 159 tokens/second on TPU v5e-16 (Llama 3.1 70B, batch size 8, 2048 context)
- ✅ Custom TPU ragged decode attention kernel
- 🚧 GPU support in progress

#### **DeepSeek R1 Implementation**
- ✅ Pure JAX implementation of DeepSeek V3 with Multi-head Latent Attention (MLA)
- ✅ Expert and tensor parallelism using `shard_map`
- ✅ INT8 quantization support
- ✅ **Verified performance**: 75.9 tokens/second on TPU v5e-64 (context length 512)
- ✅ Custom "ragged dot" kernel for efficient MoE processing
- 🚧 GPU support and further optimizations in progress

#### **Qwen 3 Implementation**
- ✅ Pure JAX implementation with MLA attention
- ✅ Expert and tensor parallelism support
- ✅ Designed for multi-host deployment (235B parameters)
- ✅ INT8 quantization
- 🚧 GPU support in progress

#### **Additional Models**
- ✅ Llama 4 implementation
- ✅ Kimi K2 implementation
- ✅ OpenAI GPT OSS implementation

### Current Limitations

The upstream project has noted limitations:

- **Primary TPU focus**: Most testing and optimization on TPU hardware
- **GPU support in progress**: Full GPU optimization still being developed
- **Research-oriented**: Optimized for performance research rather than production deployment

## Oumi's JAX Integration

### Our Implementation

Oumi has integrated these JAX LLM Examples into our inference pipeline by:

1. **Vendoring implementations**: All 7 model families synced from upstream jax-llm-examples
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

## Performance Expectations

Based on upstream JAX LLM Examples benchmarks:

### Verified Performance (TPU)
- **Llama 3.1 70B**: 159 tokens/second (TPU v5e-16, batch size 8, 2048 context)
- **DeepSeek R1**: 75.9 tokens/second (TPU v5e-64, context length 512)

### Current Status
- **TPU**: Optimized and tested
- **GPU**: Support in development upstream
- **CPU**: Available for testing and development

*Note: Performance will vary based on model size, hardware configuration, and sequence length.*

## Development and Testing

### Local Testing

Test the integration on your local machine:

```bash
# Test basic functionality (CPU)
python test_jax_local.py

# Interactive chat interface
python chat_with_jax.py

# Integration tests
python test_oumi_jax_engine.py
```

### Test Suite

```bash
# Run JAX-specific tests
pytest tests/integration/jax_models/ -v -m "cpu"

# GPU tests (requires CUDA)
pytest tests/integration/jax_models/ -v -m "single_gpu"
```

## Future Development

### Upstream Roadmap (JAX LLM Examples)

Based on the upstream repository, planned improvements include:

- **Enhanced GPU support**: Full optimization for NVIDIA hardware
- **Additional kernels**: Ragged decode MLA kernel for DeepSeek
- **Prefill optimizations**: Improved throughput for initial token processing
- **Model distillation**: Support for smaller, faster variants

### Oumi Integration Roadmap

- **Training support**: Extend beyond inference to training workflows
- **Advanced quantization**: 4-bit and mixed precision
- **Streaming APIs**: Real-time inference capabilities
- **Production hardening**: Enhanced error handling and monitoring

## Learn More

### Essential Resources

- **[JAX Documentation](https://jax.readthedocs.io/)** - Official JAX documentation
- **[JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples)** - Source repository for our implementations
- **[JAX Scaling Book](https://jax-ml.github.io/scaling-book/)** - Comprehensive guide to scaling with JAX

### Research Papers

- [JAX: Composable transformations of Python+NumPy programs](https://arxiv.org/abs/1912.03559)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [Llama 2: Open Foundation and Fine-tuned Chat Models](https://arxiv.org/abs/2307.09288)

## Contributing

We welcome contributions to improve the JAX integration! Please see our [Contributing Guide](../../../CONTRIBUTING.md) for guidelines.

When contributing:

1. Test changes on available hardware (CPU required, GPU/TPU optional)
2. Follow existing patterns from other Oumi inference engines
3. Add appropriate test coverage
4. Update documentation for new features

## Support

- **Issues**: [GitHub Issues](https://github.com/oumi-ai/oumi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/oumi-ai/oumi/discussions)
- **JAX Community**: [JAX GitHub](https://github.com/google/jax)

---

*This integration is based on the open-source [JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples) project. All performance claims and technical details are derived from the upstream repository's documentation and should be verified independently for your specific use case.*
