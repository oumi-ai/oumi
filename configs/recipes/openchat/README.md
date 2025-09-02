# OpenChat Model Configurations

This directory contains configurations for OpenChat models, focusing on conversational AI capabilities.

## Models Included

### OpenChat-3.6-8B-20240522
- **Size**: 8B parameters
- **Type**: Conversational instruction-tuned model
- **Context Length**: 8192 tokens
- **Use Cases**: General conversation, chat applications, Q&A

## Available Configurations

Each model provides 4 inference configurations:

1. **`{model}_infer.yaml`** - NATIVE engine (CPU/GPU compatible)
2. **`{model}_vllm_infer.yaml`** - vLLM engine (GPU optimized)  
3. **`{model}_gguf_infer.yaml`** - vLLM + GGUF quantization (GPU)
4. **`{model}_gguf_macos_infer.yaml`** - LlamaCPP + GGUF (macOS/Metal)

## Usage

Run inference with any configuration:

```bash
# NATIVE engine (CPU/GPU)
oumi infer -i -c configs/recipes/openchat/inference/3_6_8b_infer.yaml

# vLLM engine (GPU)
oumi infer -i -c configs/recipes/openchat/inference/3_6_8b_vllm_infer.yaml

# GGUF quantized (GPU)
oumi infer -i -c configs/recipes/openchat/inference/3_6_8b_gguf_infer.yaml

# GGUF quantized (macOS Metal)
oumi infer -i -c configs/recipes/openchat/inference/3_6_8b_gguf_macos_infer.yaml
```

## Model Information

- **Developer**: OpenChat Team
- **License**: Apache 2.0
- **Paper**: [OpenChat](https://github.com/imoneoi/openchat)
- **HuggingFace**: [openchat/openchat-3.6-8b-20240522](https://huggingface.co/openchat/openchat-3.6-8b-20240522)