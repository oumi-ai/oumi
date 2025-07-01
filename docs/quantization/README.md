# Model Quantization with Oumi

Oumi provides a comprehensive model quantization system that reduces model size and memory requirements while maintaining inference performance. The system supports multiple quantization methods and output formats for different deployment scenarios.

## Quick Start

### Basic Usage

```bash
# AWQ 4-bit quantization (recommended)
oumi quantize --method awq_q4_0 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output model.gguf

# BitsAndBytes 4-bit quantization
oumi quantize --method bnb_4bit --model "microsoft/DialoGPT-small" --output model.pytorch

# Using configuration file
oumi quantize --config quantization_config.yaml
```

### Installation

```bash
# Basic quantization support
pip install torch transformers

# AWQ quantization (recommended)
pip install autoawq

# BitsAndBytes quantization
pip install bitsandbytes

# GGUF output format support
pip install llama-cpp-python
```

## Quantization Methods

### AWQ (Activation-aware Weight Quantization) - Recommended

AWQ provides the best quality quantization by using calibration data to identify important weights that should be preserved during quantization.

| Method | Description | Compression | Quality |
|--------|-------------|-------------|---------|
| `awq_q4_0` | AWQ 4-bit → GGUF q4_0 | 4x | High |
| `awq_q4_1` | AWQ 4-bit → GGUF q4_1 | 4x | Higher |
| `awq_q8_0` | AWQ 8-bit → GGUF q8_0 | 2x | Highest |
| `awq_f16` | AWQ → GGUF f16 | 2x | Reference |

**Supported Models:**
- ✅ Llama/Llama-2/Llama-3 models
- ✅ Mistral models
- ✅ TinyLlama models
- ✅ CodeLlama models
- ❌ GPT-2 based models (use BitsAndBytes instead)

### BitsAndBytes Quantization

Efficient GPU quantization that works with most model architectures.

| Method | Description | Compression | Compatibility |
|--------|-------------|-------------|---------------|
| `bnb_4bit` | 4-bit with NF4 | 4x | High |
| `bnb_8bit` | 8-bit linear | 2x | Highest |

### Direct GGUF Methods

Direct conversion to GGUF format without AWQ preprocessing.

| Method | Description | Use Case |
|--------|-------------|----------|
| `q4_0` | 4-bit block quantization | Fast conversion |
| `q4_1` | 4-bit with bias terms | Better accuracy |
| `q5_0` | 5-bit quantization | Balanced |
| `q8_0` | 8-bit quantization | High quality |
| `f16` | 16-bit float | Format conversion |

## Output Formats

### GGUF (Recommended for Inference)

- **Use case:** llama.cpp, CPU inference, edge deployment
- **Format:** Single file with metadata
- **Compatibility:** Universal

```bash
oumi quantize --method awq_q4_0 --model "model_name" --output model.gguf
```

### PyTorch

- **Use case:** PyTorch inference, research
- **Format:** Directory with safetensors
- **Compatibility:** PyTorch ecosystem

```bash
oumi quantize --method awq_q4_0 --model "model_name" --output model.pytorch
```

### Safetensors

- **Use case:** HuggingFace transformers, GPU inference
- **Format:** Safe tensor serialization
- **Compatibility:** HuggingFace ecosystem

```bash
oumi quantize --method bnb_4bit --model "model_name" --output model.safetensors
```

## Configuration

### Using Configuration Files

Create a YAML configuration file:

```yaml
# quantization_config.yaml
model:
  model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  tokenizer_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

method: "awq_q4_0"
output_path: "tinyllama-q4.gguf"
output_format: "gguf"

# AWQ-specific settings
awq_group_size: 128
calibration_samples: 512
cleanup_temp: true

verbose: true
```

Then run:

```bash
oumi quantize --config quantization_config.yaml
```

### Python API

```python
from oumi.core.configs import QuantizationConfig, ModelParams
from oumi import quantize

config = QuantizationConfig(
    model=ModelParams(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    method="awq_q4_0",
    output_path="model.gguf",
    output_format="gguf",
    calibration_samples=512,
    verbose=True
)

result = quantize(config)
print(f"Compression ratio: {result['compression_ratio']}")
```

## Advanced Configuration

### AWQ Parameters

```yaml
# High quality settings
awq_group_size: 64          # Smaller = higher accuracy, slower
calibration_samples: 1024   # More samples = better quality
awq_zero_point: true        # Better accuracy for most models
awq_version: "GEMM"         # Faster kernels

# Performance settings
batch_size: 16              # Adjust based on GPU memory
cleanup_temp: true          # Remove intermediate files
```

### Model-Specific Configurations

For different model sizes:

```yaml
# Small models (< 1B parameters)
calibration_samples: 128
batch_size: 32
awq_group_size: 128

# Medium models (1B-7B parameters)
calibration_samples: 512
batch_size: 16
awq_group_size: 128

# Large models (> 7B parameters)
calibration_samples: 256
batch_size: 8
awq_group_size: 64
```

## Fallback Modes

The quantization system includes intelligent fallback mechanisms:

### 1. AWQ Simulation Mode
When AutoAWQ is not installed:
- Creates realistic mock outputs for testing
- Validates configurations and interfaces
- Provides size estimates

### 2. BitsAndBytes Fallback
When AutoAWQ is unavailable (e.g., macOS):
- Uses BitsAndBytes for real quantization
- Maintains model quality
- Works with all PyTorch-supported models

### 3. GGUF Conversion Fallback
When llama-cpp-python is not installed:
- Saves as PyTorch format instead
- Provides clear installation instructions
- Preserves quantized model

## Troubleshooting

### Common Issues

**1. "gpt2 isn't supported yet"**
```bash
# Use BitsAndBytes for GPT-2 models
oumi quantize --method bnb_4bit --model "microsoft/DialoGPT-small" --output model.pytorch
```

**2. "llama-cpp-python not available"**
```bash
# Install for GGUF support
pip install llama-cpp-python
```

**3. "You are trying to access a gated repo"**
```bash
# Authenticate with HuggingFace
huggingface-cli login
```

**4. CUDA out of memory**
```yaml
# Reduce batch size and calibration samples
batch_size: 4
calibration_samples: 128
```

### Performance Tips

1. **Start small:** Test with TinyLlama before larger models
2. **Use AWQ:** Best quality-to-compression ratio
3. **Adjust parameters:** Lower `awq_group_size` for better quality
4. **Monitor memory:** Reduce `batch_size` if needed
5. **Cache models:** Download once, quantize multiple times

## Examples

See the [examples directory](../examples/quantization/) for:
- Production configurations
- Model-specific examples
- Batch processing scripts
- Integration examples

## API Reference

- [QuantizationConfig](api/quantization_config.md) - Configuration options
- [Quantization Methods](api/methods.md) - Available methods
- [Output Formats](api/formats.md) - Format specifications
- [Error Handling](api/errors.md) - Error codes and solutions