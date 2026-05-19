# Quantization Examples

This directory contains example configurations for model quantization.
Oumi integrates with [LLM Compressor](https://github.com/vllm-project/llm-compressor)
and [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
to provide quantization capabilities.

> **NOTE**: Quantization requires a GPU to run.

## Configuration Files

- **`fp8_quantization_config.yaml`** - FP8 dynamic quantization (data-free, Hopper+)
- **`w4a16_quantization_config.yaml`** - 4-bit weight quantization via GPTQ (with calibration)
- **`bnb_quantization_config.yaml`** - BitsAndBytes 4-bit quantization

## Quick Start

```bash
# Data-free FP8 quantization (fastest, requires H100/H200)
oumi quantize --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --method fp8_dynamic --output tinyllama-fp8

# 4-bit weight quantization via GPTQ (requires calibration data)
oumi quantize -c configs/examples/quantization/w4a16_quantization_config.yaml

# Using configuration file
oumi quantize -c configs/examples/quantization/fp8_quantization_config.yaml
```

## Supported Methods

### LLM Compressor ([GitHub](https://github.com/vllm-project/llm-compressor))

We currently support the following subset of LLM Compressor methods:

- `fp8_dynamic` - FP8 dynamic quantization (data-free, Hopper+) **(default)**
- `fp8_block` - FP8 block-wise quantization (data-free, Hopper+)
- `w4a16` - 4-bit weight quantization via GPTQ (Turing+)
- `w4a16_asym` - 4-bit asymmetric weight quantization via AWQ (Turing+)
- `w8a16` - 8-bit weight quantization via GPTQ (Turing+)

### BitsAndBytes

- `bnb_4bit` - 4-bit quantization with NF4
- `bnb_8bit` - 8-bit linear quantization

## Output Formats

- **safetensors** - compressed-tensors format optimized for vLLM serving

## Requirements

```bash
pip install oumi[quantization]
```

For more details, see the [Quantization Guide](https://oumi.ai/docs/en/latest/user_guides/quantization.html).
