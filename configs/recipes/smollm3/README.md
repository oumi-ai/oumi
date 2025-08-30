# SmolLM3 Model Family

SmolLM3 is a family of compact language models developed by HuggingFace TB, designed for efficient inference while maintaining strong performance. The SmolLM3-3B model provides an excellent balance between capability and resource requirements.

## Model Information

- **Model Name**: HuggingFaceTB/SmolLM3-3B
- **Parameters**: 3.08B
- **Context Length**: 8K tokens
- **Architecture**: Llama-based transformer
- **License**: Apache 2.0

## Available Configurations

### Inference Configurations

#### SmolLM3-3B
- `3b_infer.yaml` - NATIVE engine (CPU/GPU)
- `3b_vllm_infer.yaml` - vLLM engine (GPU optimized)
- `3b_gguf_infer.yaml` - vLLM + GGUF quantization (GPU)
- `3b_gguf_macos_infer.yaml` - LlamaCPP + GGUF (macOS Metal optimized)

## Quick Start

### NATIVE Engine (Default)
```bash
oumi infer -i -c configs/recipes/smollm3/inference/3b_infer.yaml
```

### vLLM Engine (GPU)
```bash
pip install oumi[gpu]
oumi infer -i -c configs/recipes/smollm3/inference/3b_vllm_infer.yaml
```

### GGUF Quantized (GPU)
```bash
pip install oumi[gpu]
oumi infer -i -c configs/recipes/smollm3/inference/3b_gguf_infer.yaml
```

### GGUF Quantized (macOS)
```bash
pip install oumi[llama_cpp]
oumi infer -i -c configs/recipes/smollm3/inference/3b_gguf_macos_infer.yaml
```

## Performance Notes

- **Memory Requirements**: ~6-8GB VRAM for full precision inference
- **GGUF Quantized**: ~2-4GB VRAM/RAM (depending on quantization level)
- **CPU Inference**: Supported but slower than GPU

## GGUF Quantization Options

The GGUF configs use Q4_K_M quantization by default (~1.9GB), but you can modify the `filename` parameter to use other quantizations:

- `HuggingFaceTB_SmolLM3-3B-Q2_K.gguf` - 1.13GB (lowest quality)
- `HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf` - 1.92GB (recommended)
- `HuggingFaceTB_SmolLM3-3B-Q5_K_M.gguf` - 2.28GB (higher quality)
- `HuggingFaceTB_SmolLM3-3B-Q6_K.gguf` - 2.62GB (near full precision)

## Model Capabilities

SmolLM3-3B is optimized for:
- General chat and conversation
- Code generation and understanding
- Text summarization and analysis
- Question answering
- Creative writing tasks

The model supports 8 languages and follows the ChatML prompt format for optimal performance.