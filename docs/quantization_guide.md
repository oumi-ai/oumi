# Model Quantization Guide

This guide covers the `oumi quantize` command for reducing model size while maintaining performance.

## Quick Start

### AWQ Quantization (Recommended)

### AWQ Quantization

```bash
oumi quantize --method awq_q4_0 --model "oumi-ai/HallOumi-8B" --output halloumi_awq4bit.pytorch
```

**Expected Result:**
✅ Model quantized successfully!
📁 Output saved to: halloumi_awq4bit.pytorch
📊 Original size: 15.0 GB
📉 Output size: 5.4 GB
🗜️  Compression ratio: 2.80x

**Other Example Commonds**

```bash
oumi quantize --method awq_q4_0 --model "meta-llama/Llama-2-7b-hf" --output model.pytorch
oumi quantize --method awq_q4_0 --model "Qwen/Qwen3-14B" --output Qwen3-14B_awq4bit.pytorch
```

### Configuration Files

```yaml
model:
  model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
method: "awq_q4_0"
output_path: "tinyllama_quantized.pytorch"
output_format: "pytorch"
awq_group_size: 128
calibration_samples: 512
```

### PyTorch Format Output

```bash
oumi quantize --method awq_q4_0 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output model.pytorch
```

## Implementation Details

### Core Quantization Methods

- **AWQ Quantization**: `awq_q4_0`, `awq_q4_1`, `awq_q8_0`, `awq_f16`
- **BitsAndBytes**: `bnb_4bit`, `bnb_8bit` with fallback support
- **Direct GGUF**: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `f16`, `f32`

## Installation

```bash
# AWQ quantization (recommended)
pip install autoawq

# BitsAndBytes quantization (broad compatibility)
pip install bitsandbytes

# GGUF output support (optional)
pip install llama-cpp-python
```

## Quantization Methods

### AWQ (Activation-aware Weight Quantization) - Recommended

**Supported Models:** Llama, Mistral, TinyLlama, CodeLlama, QWen

### BitsAndBytes Quantization

| Method | Description | Compression | Compatibility |
|--------|-------------|-------------|---------------|
| `bnb_4bit` | 4-bit with NF4 | 4x | Universal |
| `bnb_8bit` | 8-bit linear | 2x | Universal |

**Supported Models:** GPT-2, DialoGPT, all PyTorch models

### Direct GGUF Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `q4_0` | 4-bit block quantization | Quick conversion |
| `q4_1` | 4-bit with bias terms | Better accuracy |
| `q8_0` | 8-bit quantization | High quality |
| `f16` | 16-bit float | Format conversion |

## Configuration Files

Create reusable quantization configurations:

```yaml
# quantization_config.yaml
method: awq_q4_0
model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_path: "quantized_model.gguf"
overwrite: true
```

```bash
oumi quantize --config quantization_config.yaml
```

## Examples

See the [examples/quantization/](../examples/quantization/) directory for sample configurations:

- `quantization_config.yaml` - Simple quantization setup

## CLI Reference

```bash
# Basic usage
oumi quantize --method METHOD --model MODEL --output OUTPUT

# With configuration
oumi quantize --config CONFIG_FILE

# Override config options
oumi quantize --config CONFIG_FILE --method awq_q8_0 --overwrite
```

### Parameters

- `--method`: Quantization method (awq_q4_0, bnb_4bit, etc.)
- `--model`: Model identifier (HuggingFace ID or local path)
- `--output`: Output file path
- `--config`: Configuration file path
- `--overwrite`: Overwrite existing output file
