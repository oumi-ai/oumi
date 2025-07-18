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

For more details, see the [Quantization Guide](../../docs/quantization_guide.md).
