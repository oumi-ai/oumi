# Mistral Magistral Model Family

## Model Family Overview

Mistral Magistral is an advanced series of large language models built by Mistral AI. These models provide state-of-the-art performance for conversational AI, content generation, and complex reasoning tasks while maintaining efficiency across different deployment scenarios.

## Available Models

### Magistral Small 2507
- **Parameters**: 23.6B
- **Architecture**: Llama-based
- **Context Length**: 32768 tokens  
- **Type**: General-purpose conversational model
- **Memory Requirements**: ~48GB (NATIVE), ~15GB (GGUF Q4_K_M)
- **Best For**: High-quality chat, content generation, reasoning tasks

## Available Configurations

### Magistral Small 2507 Configurations
- `small_2507_infer.yaml` - NATIVE engine (CPU/GPU compatible)
- `small_2507_vllm_infer.yaml` - vLLM engine (GPU optimized)
- `small_2507_gguf_infer.yaml` - vLLM + GGUF quantization (GPU)
- `small_2507_gguf_macos_infer.yaml` - LlamaCPP + GGUF (macOS optimized)

## Quick Start

```bash
# High-quality inference with NATIVE engine
oumi infer -i -c configs/recipes/mistral_magistral/inference/small_2507_infer.yaml

# GPU-accelerated inference with vLLM
oumi infer -i -c configs/recipes/mistral_magistral/inference/small_2507_vllm_infer.yaml

# Memory-efficient GGUF on macOS
oumi infer -i -c configs/recipes/mistral_magistral/inference/small_2507_gguf_macos_infer.yaml

# GPU GGUF inference (balanced performance/memory)
oumi infer -i -c configs/recipes/mistral_magistral/inference/small_2507_gguf_infer.yaml
```

## Model Selection Guide

### Choose Magistral Small 2507 when:
- 🎯 **High-quality** conversational AI needed
- 💬 **Complex reasoning** and analysis tasks
- 📚 **Long-form content** generation
- 🔧 **Professional applications** requiring accuracy
- ⚖️ **Balanced** performance vs. resource usage

## Hardware Requirements

### NATIVE Engine
- **RAM**: 48GB+ recommended
- **GPU**: Optional, any CUDA-compatible GPU
- **Storage**: ~47GB for full precision model

### vLLM Engine  
- **GPU Memory**: 24GB+ VRAM recommended (RTX 4090, A6000, etc.)
- **GPU**: CUDA-compatible GPU required
- **Storage**: ~47GB for full precision model

### GGUF Variants
| Quantization | Memory | Quality | Use Case |
|-------------|--------|---------|----------|
| Q4_K_M | ~15GB | Good | Default recommendation |
| Q5_K_M | ~18GB | High | Better quality |
| Q6_K | ~22GB | Very High | Maximum quality |
| IQ4_XS | ~13GB | Good | Smallest file size |

## Prompt Format

Magistral Small 2507 uses the Mistral instruction format:

```
<s>[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]{user_message}[/INST]
```

Example:
```
<s>[SYSTEM_PROMPT]You are a helpful AI assistant.[/SYSTEM_PROMPT][INST]What are the benefits of renewable energy?[/INST]
```

## Model Features

- **Advanced Reasoning**: Superior performance on complex reasoning tasks
- **Long Context**: 32K token context window for extended conversations
- **Multi-Modal Ready**: Architecture supports future multi-modal extensions
- **Efficient Architecture**: Optimized Llama-based architecture
- **GGUF Quantization**: Multiple precision levels for different hardware
- **Production Ready**: Suitable for commercial applications

## Performance Characteristics

- **Strengths**: Complex reasoning, long-form generation, instruction following
- **Context Window**: 32768 tokens (approximately 24,000-30,000 words)
- **Inference Speed**: Fast with vLLM, moderate with NATIVE, excellent with GGUF
- **Memory Efficiency**: GGUF variants offer significant memory savings

## Use Cases

- **Enterprise Chatbots**: Customer service and internal tools
- **Content Creation**: Technical writing, documentation, creative content
- **Code Analysis**: Code review, explanation, and generation assistance
- **Research Assistant**: Literature review, data analysis, report generation
- **Educational Tools**: Tutoring, explanation, and learning assistance