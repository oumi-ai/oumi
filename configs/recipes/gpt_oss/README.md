# gpt-oss

## Summary

Configs for OpenAI's gpt-oss model family. See the [blog post](https://huggingface.co/blog/welcome-openai-gpt-oss) for more information. The models in this family, both of which are MoE architectures, are:

- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b): 21B total/3.6B active parameters
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b): 117B total/5.1B active parameters

## Key Features

- **MXFP4 Quantization**: Native 4-bit mixed-precision floating-point format
- **Reasoning Capabilities**: Built-in chain-of-thought reasoning with effort levels
- **Tool Use**: Native support for built-in and custom tools
- **Efficient Deployment**: 120B model fits on single H100, 20B runs on 16GB GPU

## Installation

### Prerequisites

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for basic Oumi installation.

### GPT OSS Specific Requirements

To use GPT OSS models, you need to install additional dependencies:

```bash
# Install Oumi with GPT OSS support
pip install oumi[gpt_oss]

# Install special vLLM build for GPT OSS
pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Install Flash Attention 3 (required for optimal performance)
pip install flash-attn>=3.0.0 --no-build-isolation
```

### Hardware Requirements

- **gpt-oss-20b**: 16GB+ GPU memory (e.g., V100, A100, RTX 4090)
- **gpt-oss-120b**: 80GB GPU memory (e.g., H100, A100-80GB)
> **Note**: It is recommended to run these models on Hopper-family GPUs. We have tested on H100s with CUDA 12.8. The models can still run on other hardware, but you may need to deviate from our instructions below. See the blog above for more details.

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. As mentioned in the blog above, gpt-oss models require some of the latest versions of packages to run. Run the following command:

   ```bash
   uv pip install -U accelerate transformers kernels torchvision git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
   ```

3. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.

## Example Commands

### Inference

#### Remote Inference (Together AI)
To run interactive inference on gpt-oss-120b locally:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/120b_infer.yaml
```

To run interactive remote inference on gpt-oss-120b on Together AI:

```shell
# Interactive inference on gpt-oss-120b
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/120b_together_infer.yaml
```

#### Local Inference with vLLM

```shell
# Run gpt-oss-20b locally with MXFP4 quantization
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/20b_vllm_infer.yaml

# Run gpt-oss-120b on H100
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/120b_vllm_infer.yaml
```

### Fine-tuning

```shell
# LoRA fine-tuning for gpt-oss-20b
oumi train -c oumi://configs/recipes/gpt_oss/sft/20b_lora_train.yaml
```

### Quantization

To quantize a GPT OSS model to MXFP4 format:

```shell
oumi quantize \
    --model-id openai/gpt-oss-20b \
    --output-dir ./gpt-oss-20b-mxfp4 \
    --quantization-method mxfp4
```

## Configuration Options

### Reasoning Effort Levels

GPT OSS models support different reasoning effort levels:

- `low`: Fast responses with minimal reasoning
- `medium`: Balanced reasoning and response time (default)
- `high`: Deep reasoning with comprehensive analysis

Set in generation parameters:
```yaml
generation_parameters:
  reasoning_effort: "high"
```

### Tool Use

Enable tool use in inference configs:
```yaml
tool_use:
  enabled: true
  max_tools: 10
```

## Troubleshooting

### CUDA/Flash Attention Issues

If you encounter CUDA errors with Flash Attention 3:
```bash
# Reinstall with specific CUDA version
pip install flash-attn>=3.0.0 --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

### vLLM Installation Issues

If the special vLLM build fails to install:
1. Ensure you're using Python 3.9-3.12
2. Check CUDA compatibility (11.8+ required)
3. Try installing in a fresh virtual environment

### Memory Issues

For OOM errors with large models:
- Reduce `max_model_len` in vLLM configs
- Lower `gpu_memory_utilization` (e.g., 0.8)
- Enable CPU offloading if available
