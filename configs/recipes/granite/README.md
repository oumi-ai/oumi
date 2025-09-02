# Granite Model Configurations

This directory contains configurations for IBM Granite models, focusing on enterprise-grade instruction-following capabilities.

## Models Included

### Granite-3.3-8B-Instruct
- **Size**: 8B parameters  
- **Type**: Instruction-tuned model
- **Context Length**: 8192 tokens
- **Use Cases**: Enterprise applications, instruction following, code assistance

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
oumi infer -i -c configs/recipes/granite/inference/3_3_8b_instruct_infer.yaml

# vLLM engine (GPU) 
oumi infer -i -c configs/recipes/granite/inference/3_3_8b_instruct_vllm_infer.yaml

# GGUF quantized (GPU)
oumi infer -i -c configs/recipes/granite/inference/3_3_8b_instruct_gguf_infer.yaml

# GGUF quantized (macOS Metal)
oumi infer -i -c configs/recipes/granite/inference/3_3_8b_instruct_gguf_macos_infer.yaml
```

## Model Information

- **Developer**: IBM
- **License**: Apache 2.0
- **Paper**: [Granite Models](https://github.com/ibm-granite/granite-models)
- **HuggingFace**: [ibm-granite/granite-3.3-8b-instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)