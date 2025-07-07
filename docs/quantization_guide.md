# Model Quantization Guide

This guide covers the `oumi quantize` command for reducing model size while maintaining performance.

## Quick Start

### Basic AWQ Quantization (Recommended)

```bash
# AWQ 4-bit quantization
oumi quantize --method awq_q4_0 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output model.gguf
```

Expected output:
```
✅ AWQ quantization completed successfully!
📁 Output saved to: model.gguf
📊 Original size: 2.2 GB
📉 Output size: 661.5 MB
🗜️ Compression ratio: 3.32x
```

### Alternative Methods

```bash
# BitsAndBytes for unsupported models
oumi quantize --method bnb_4bit --model "microsoft/DialoGPT-small" --output model.pytorch

# Using configuration file  
oumi quantize --config examples/quantization/basic_quantize_config.yaml
```

## Installation

```bash
# Core dependencies
pip install torch transformers

# AWQ quantization (recommended)
pip install autoawq

# BitsAndBytes quantization (broad compatibility)
pip install bitsandbytes

# GGUF output support (optional)
pip install llama-cpp-python
```

## Quantization Methods

### AWQ (Activation-aware Weight Quantization) - Recommended

| Method | Description | Compression | Quality Loss |
|--------|-------------|-------------|--------------|
| `awq_q4_0` | AWQ 4-bit → GGUF q4_0 | 3.3x | 3-4% |
| `awq_q4_1` | AWQ 4-bit → GGUF q4_1 | 3.3x | 2-3% |
| `awq_q8_0` | AWQ 8-bit → GGUF q8_0 | 1.9x | 0.5% |
| `awq_f16` | AWQ → GGUF f16 | 1.8x | ~0% |

**Supported Models:** Llama, Mistral, TinyLlama, CodeLlama

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

## Output Formats

### GGUF (Recommended for Inference)
```bash
oumi quantize --method awq_q4_0 --model "model_name" --output model.gguf
```
**Use with:** llama.cpp, Ollama, CPU inference

### PyTorch  
```bash
oumi quantize --method awq_q4_0 --model "model_name" --output model.pytorch
```
**Use with:** PyTorch inference, custom applications

### Safetensors
```bash
oumi quantize --method bnb_4bit --model "model_name" --output model.safetensors
```
**Use with:** HuggingFace ecosystem

## Configuration Files

Create reusable quantization configurations:

```yaml
# basic_quantize_config.yaml
method: awq_q4_0
model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_path: "quantized_model.gguf"
overwrite: true
```

```bash
oumi quantize --config basic_quantize_config.yaml
```

## Examples

See the [examples/quantization/](../examples/quantization/) directory for sample configurations:

- `basic_quantize_config.yaml` - Simple quantization setup
- `advanced_quantize_config.yaml` - Advanced options
- `safetensors_quantize_config.yaml` - Safetensors output

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