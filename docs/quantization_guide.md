# Model Quantization Guide

> 🚧 **DEVELOPMENT STATUS**: The quantization feature is currently under active development. Some features may be experimental or subject to change.

This guide covers the `oumi quantize` command for reducing model size while maintaining performance.

## Quick Start

```bash
# Simplest example - quantize TinyLlama to 4-bit
oumi quantize --method awq_q4_0 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output model.pytorch
```

**Expected Output:**
```
✅ Model quantized successfully!
📁 Output saved to: model.pytorch
🔧 Method: awq_q4_0
📋 Format: pytorch
📊 Quantized size: 0.65 GB
```

## Quantization Methods

### AWQ Quantization (Recommended)

AWQ (Activation-aware Weight Quantization) provides the best quality-to-size ratio by considering activation patterns during quantization.

```bash
# 4-bit AWQ quantization
oumi quantize --method awq_q4_0 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output tinyllama_awq4bit.pytorch
oumi quantize --method awq_q4_0 --model "oumi-ai/HallOumi-8B" --output halloumi_awq4bit.pytorch

# Higher precision 8-bit AWQ
oumi quantize --method awq_q8_0 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output tinyllama_awq8bit.pytorch
```

**Supported AWQ Methods:**
- `awq_q4_0`: 4-bit quantization (default, ~4x compression)
- `awq_q4_1`: 4-bit with asymmetric quantization
- `awq_q8_0`: 8-bit quantization (~2x compression, higher quality)
- `awq_f16`: 16-bit float conversion

### BitsAndBytes Quantization

For broader model compatibility when AWQ isn't supported:

```bash
# 4-bit quantization with NF4
oumi quantize --method bnb_4bit --model "gpt2" --output gpt2_4bit.pytorch

# 8-bit quantization
oumi quantize --method bnb_8bit --model "microsoft/DialoGPT-medium" --output dialogpt_8bit.pytorch
```

**Supported BitsAndBytes Methods:**
- `bnb_4bit`: 4-bit quantization with NF4 (~4x compression)
- `bnb_8bit`: 8-bit linear quantization (~2x compression)

## Configuration Files

For reproducible workflows, use configuration files:

```yaml
# quantization_config.yaml
model:
  model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  trust_remote_code: false
method: "awq_q4_0"
output_path: "tinyllama_quantized.pytorch"
output_format: "pytorch"      # Options: pytorch, safetensors
awq_group_size: 128          # AWQ-specific: weight grouping size
calibration_samples: 512     # AWQ-specific: calibration dataset size
```

Run with configuration:
```bash
oumi quantize --config quantization_config.yaml
```

## Output Formats

Currently supported output formats:
- **pytorch**: PyTorch state dict format (`.pytorch` extension)
- **safetensors**: HuggingFace safetensors format (`.safetensors` extension)

> **Note**: GGUF format mentioned in examples is not yet implemented in the current version.

## Installation

```bash
# For AWQ quantization
pip install autoawq

# For BitsAndBytes quantization
pip install bitsandbytes
```

## Advanced Configuration

### AWQ-specific Settings

```yaml
# Advanced AWQ configuration
awq_group_size: 128          # Weight grouping (64/128/256)
awq_zero_point: true         # Enable zero-point quantization
awq_version: "GEMM"          # Kernel version (GEMM/GEMV)
calibration_samples: 512     # Number of calibration samples
cleanup_temp: true           # Remove temporary files
```

### Performance Tuning

```yaml
batch_size: 8               # Batch size for calibration (auto if null)
verbose: true               # Enable detailed logging
```

## Examples Directory

See [examples/quantization/](../examples/quantization/) for ready-to-use configurations:
- `quantization_config.yaml` - Basic quantization setup
- `calibrated_quantization_config.yaml` - Production setup with enhanced calibration

## CLI Reference

```bash
# Basic usage
oumi quantize --method METHOD --model MODEL --output OUTPUT

# With configuration file
oumi quantize --config CONFIG_FILE

# Override config options
oumi quantize --config CONFIG_FILE --method awq_q8_0
```

### Parameters

- `--method`: Quantization method (default: awq_q4_0)
- `--model`: Model identifier (HuggingFace ID or local path)
- `--output`: Output file path (default: quantized_model.gguf)
- `--config`: Configuration file path

## Troubleshooting

1. **GPU Memory**: Large models may require significant GPU memory during quantization
2. **Supported Models**: AWQ works best with Llama-family models; use BitsAndBytes for others
3. **Output Path**: Ensure the output extension matches the format (`.pytorch` or `.safetensors`)
