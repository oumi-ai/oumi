# NVIDIA Nemotron Nano

## Model Family Overview

NVIDIA Nemotron Nano is a series of conversational AI models optimized for efficient inference. The 9B parameter variant provides strong performance while maintaining reasonable computational requirements.

## Model Specifications

- **Model**: NVIDIA Nemotron Nano 9B v2
- **Parameters**: ~9B (8.89B)
- **Architecture**: Nemotron-H
- **Context Length**: 4096 tokens
- **Base Model**: nvidia/NVIDIA-Nemotron-Nano-9B-v2

## Available Configurations

### Inference Configurations

- `9b_infer.yaml` - NATIVE engine (CPU/GPU compatible)
- `9b_vllm_infer.yaml` - vLLM engine (GPU optimized) 
- `9b_gguf_infer.yaml` - vLLM + GGUF quantization (GPU)
- `9b_gguf_macos_infer.yaml` - LlamaCPP + GGUF (macOS optimized)

## Quick Start

```bash
# Interactive chat with NATIVE engine
oumi infer -i -c configs/recipes/nemotron_nano/inference/9b_infer.yaml

# GPU-accelerated inference with vLLM
oumi infer -i -c configs/recipes/nemotron_nano/inference/9b_vllm_infer.yaml

# Memory-efficient GGUF on macOS
oumi infer -i -c configs/recipes/nemotron_nano/inference/9b_gguf_macos_infer.yaml
```

## Model Features

- **Conversational AI**: Designed for chat and dialogue applications
- **Efficient Architecture**: Optimized Nemotron-H architecture
- **GGUF Support**: Multiple quantization levels available
- **Cross-Platform**: Compatible with various hardware configurations

## Hardware Requirements

### NATIVE Engine
- **RAM**: 16GB+ recommended
- **GPU**: Optional, CUDA-compatible GPUs supported

### vLLM Engine  
- **GPU Memory**: 8GB+ VRAM recommended
- **GPU**: CUDA-compatible GPU required

### GGUF Variants
- **Q4_K_M**: ~5.5GB memory requirement
- **macOS**: Apple Silicon optimized with Metal acceleration