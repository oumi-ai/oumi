# Quantization Examples

> 🚧 **DEVELOPMENT STATUS**: The quantization feature is currently under active development.

This directory contains example configurations for model quantization using Oumi's AWQ and BitsAndBytes quantization methods.

> **NOTE**: Quantization requires a GPU to run.

## Configuration Files

- **`awq_quantization_config.yaml`** - AWQ 4-bit quantization with calibration
- **`bnb_quantization_config.yaml`** - BitsAndBytes 4-bit quantization

## Quick Start

```bash
# Simplest command-line usage
oumi quantize --method awq_q4_0 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output quantized_model

# Using configuration file
oumi quantize --config configs/examples/quantization/awq_quantization_config.yaml
```

## Supported Methods

### AWQ (Activation-aware Weight Quantization)

- `awq_q4_0` - 4-bit quantization (default)
- `awq_q4_1` - 4-bit with asymmetric quantization
- `awq_q8_0` - 8-bit quantization
- `awq_f16` - 16-bit float conversion

### BitsAndBytes

- `bnb_4bit` - 4-bit quantization with NF4
- `bnb_8bit` - 8-bit linear quantization

## Output Formats

- **safetensors** - HuggingFace safetensors format (`.safetensors` extension)

## Requirements

```bash
pip install oumi[quantization]

# Alternatively, for AWQ quantization only
pip install autoawq

# Alternatively, for BitsAndBytes quantization only
pip install bitsandbytes
```

For more details, see the [Quantization Guide](https://oumi.ai/docs/en/latest/user_guides/quantization.html).
