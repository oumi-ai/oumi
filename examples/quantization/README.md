# Quantization Examples

This directory contains example configurations for model quantization in Oumi.

## Configuration Files

- **`quantization_config.yaml`** - Basic quantization setup
- **`calibrated_quantization_config.yaml`** - Production quantization with calibration and optimized settings
- **`safetensors_quantize_config.yaml`** - Safetensors format output

## Quick Start

```bash
# Basic quantization (not calibrated). Note: this requires a machine with 1 GPU
oumi quantize --config examples/quantization/quantization_config.yaml

# Production configuration with calibration
oumi quantize --config examples/quantization/calibrated_quantization_config.yaml
```

## Available Methodss

- **AWQ Methods**: `awq_q4_0`, `awq_q4_1`, `awq_q8_0`, `awq_f16`
- **Direct Methods**: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `f16`, `f32`
- **Output Formats**: GGUF, PyTorch, Safetensors

For more details, see the [Quantization Guide](../../docs/quantization_guide.md).
