# Quantization Examples

> 🚧 **DEVELOPMENT STATUS**: The quantization feature is currently under active development.

This directory contains example configurations for model quantization using Oumi's AWQ and BitsAndBytes quantization methods.

## Configuration Files

- **`quantization_config.yaml`** - Basic AWQ 4-bit quantization with TinyLlama
- **`calibrated_quantization_config.yaml`** - Production-ready AWQ 4-bit with enhanced calibration (1024 samples)

## Quick Start

```bash
# Simplest command-line usage
oumi quantize --method awq_q4_0 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output model.pytorch

# Using configuration file (requires GPU)
oumi quantize --config examples/quantization/quantization_config.yaml

# Production configuration with more calibration samples
oumi quantize --config examples/quantization/calibrated_quantization_config.yaml
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

- **pytorch** - PyTorch state dict format (`.pytorch` extension)
- **safetensors** - HuggingFace safetensors format (`.safetensors` extension)

## Requirements

```bash
# For AWQ quantization
pip install autoawq

# For BitsAndBytes quantization
pip install bitsandbytes
```

For more details, see the [Quantization Guide](../../docs/quantization_guide.md).
