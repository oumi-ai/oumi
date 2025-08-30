# Gemma 3 Model Family

## Model Family Overview

Gemma 3 is Google's latest generation of lightweight, state-of-the-art open models. Built on the same research and technology used to create Gemini models, Gemma 3 provides excellent performance across various model sizes while maintaining efficiency.

## Available Models

### Gemma-3-270M-IT
- **Parameters**: 270M (268M)
- **Type**: Instruction-tuned conversational model
- **Context Length**: 8192 tokens
- **Memory Requirements**: ~1GB (NATIVE), ~300MB (GGUF Q4_K_M)
- **Best For**: Fast inference, low-resource environments, testing

### Gemma-3N-E4B-IT
- **Parameters**: ~4B
- **Type**: Instruction-tuned conversational model  
- **Context Length**: 16384 tokens
- **Memory Requirements**: ~8GB (NATIVE), ~2.5GB (GGUF Q4_K_M)
- **Best For**: Balanced performance and efficiency

## Available Configurations

### Gemma-3-270M-IT Configurations
- `270m_it_infer.yaml` - NATIVE engine (CPU/GPU compatible)
- `270m_it_vllm_infer.yaml` - vLLM engine (GPU optimized)
- `270m_it_gguf_infer.yaml` - vLLM + GGUF quantization (GPU)
- `270m_it_gguf_macos_infer.yaml` - LlamaCPP + GGUF (macOS optimized)

### Gemma-3N-E4B-IT Configurations
- `3n_e4b_it_infer.yaml` - NATIVE engine (CPU/GPU compatible)
- `3n_e4b_it_vllm_infer.yaml` - vLLM engine (GPU optimized)
- `3n_e4b_it_gguf_infer.yaml` - vLLM + GGUF quantization (GPU)
- `3n_e4b_it_gguf_macos_infer.yaml` - LlamaCPP + GGUF (macOS optimized)

## Quick Start

```bash
# Fast inference with 270M model (great for testing)
oumi infer -i -c configs/recipes/gemma3/inference/270m_it_infer.yaml

# Balanced performance with 4B model
oumi infer -i -c configs/recipes/gemma3/inference/3n_e4b_it_infer.yaml

# Memory-efficient GGUF on macOS (270M)
oumi infer -i -c configs/recipes/gemma3/inference/270m_it_gguf_macos_infer.yaml

# GPU-accelerated inference (4B model)
oumi infer -i -c configs/recipes/gemma3/inference/3n_e4b_it_vllm_infer.yaml
```

## Model Selection Guide

### Choose Gemma-3-270M-IT when:
- 🚀 **Fast prototyping** and testing
- 💾 **Limited memory** (< 2GB available)
- 📱 **Edge deployment** scenarios
- 🔧 **Development and debugging**

### Choose Gemma-3N-E4B-IT when:
- ⚖️ **Balanced** performance and efficiency needed
- 💬 **Production chat** applications
- 📚 **Content generation** tasks
- 🎯 **High-quality** responses required

## Hardware Requirements

### NATIVE Engine
| Model | RAM | GPU (Optional) |
|-------|-----|----------------|
| 270M | 2GB+ | Any CUDA GPU |
| 4B | 8GB+ | Any CUDA GPU |

### vLLM Engine
| Model | GPU Memory | GPU |
|-------|------------|-----|
| 270M | 1GB+ VRAM | CUDA GPU |
| 4B | 4GB+ VRAM | CUDA GPU |

### GGUF Variants (Q4_K_M)
| Model | Memory | Platform |
|-------|--------|----------|
| 270M | ~300MB | Any (CPU/GPU) |
| 4B | ~2.5GB | Any (CPU/GPU) |

## Model Features

- **Instruction Following**: Fine-tuned for chat and instruction-following
- **Efficient Architecture**: Optimized Gemma architecture
- **Multi-Platform**: Support for CPU, GPU, and mobile inference
- **GGUF Quantization**: Multiple precision levels available
- **Safety**: Built-in safety filters and responsible AI practices