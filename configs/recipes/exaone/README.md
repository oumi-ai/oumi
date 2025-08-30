# EXAONE Model Family

This directory contains configurations for LGAI EXAONE models.

## EXAONE 4.0 32B

LGAI EXAONE 4.0 is a state-of-the-art 32B parameter language model developed by LG AI Research. It represents the latest iteration in the EXAONE series, offering advanced capabilities for general-purpose language understanding and generation tasks.

### Model Details
- **Developer**: LG AI Research (LGAI)
- **Version**: EXAONE 4.0
- **Parameters**: 32B
- **Context Length**: 32,768 tokens
- **Architecture**: Transformer-based architecture optimized for multilingual capabilities
- **Specialization**: General-purpose language model with strong reasoning capabilities

### Available Configurations

#### Inference Configurations
- `4_0_32b_infer.yaml` - NATIVE engine (CPU/GPU compatible)
- `4_0_32b_vllm_infer.yaml` - vLLM engine (high-performance GPU)
- `4_0_32b_gguf_infer.yaml` - vLLM + GGUF quantization (GPU)
- `4_0_32b_gguf_macos_infer.yaml` - LlamaCPP + GGUF (macOS/CPU optimized)

### Usage Examples

```bash
# Native inference
oumi infer -i -c configs/recipes/exaone/inference/4_0_32b_infer.yaml

# High-performance GPU inference
oumi infer -i -c configs/recipes/exaone/inference/4_0_32b_vllm_infer.yaml

# Quantized GPU inference
oumi infer -i -c configs/recipes/exaone/inference/4_0_32b_gguf_infer.yaml

# macOS CPU/Metal inference
oumi infer -i -c configs/recipes/exaone/inference/4_0_32b_gguf_macos_infer.yaml
```

### Model Source
- **HuggingFace**: [LGAI-EXAONE/EXAONE-4.0-32B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B)
- **GGUF Quantizations**: [bartowski/LGAI-EXAONE_EXAONE-4.0-32B-GGUF](https://huggingface.co/bartowski/LGAI-EXAONE_EXAONE-4.0-32B-GGUF)

### Recommended Use Cases
- General-purpose text generation and completion
- Code generation and programming assistance
- Question answering and reasoning tasks
- Multilingual text processing
- Research and academic applications
- Long-form content creation with 32K context

### Performance Notes
- **32B parameters**: Requires substantial memory (16GB+ VRAM recommended for GPU inference)
- **Q3_K_XL quantization**: Provides good balance between model quality and memory usage
- **Long context**: 32K token context window enables handling of large documents
- **LG AI Research quality**: Built by a leading AI research institution with focus on practical applications

### Technical Specifications
- **Temperature**: 0.6 (balanced creativity and consistency)
- **Top-p**: 0.9 (nucleus sampling for quality generation)
- **Max New Tokens**: 8,192 (suitable for long-form responses)
- **GGUF Variant**: Q3_K_XL quantization for optimal quality/memory trade-off