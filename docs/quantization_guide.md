# Model Quantization Guide

This guide covers the `oumi quantize` command for reducing model size and memory requirements while maintaining inference performance.

## 🚧 Current Status

The quantization feature is currently **in development**. The CLI interface and configuration system are fully implemented, but the core quantization logic is being developed incrementally.

**What's Available Now:**
- ✅ Complete CLI interface (`oumi quantize`)
- ✅ Configuration system and validation
- ✅ Input validation and error handling
- ✅ Documentation and examples
- 🚧 Core quantization implementation (in development)

**Current Behavior:**
The command currently runs in **simulation mode**, validating all inputs and configuration without performing actual quantization. This allows testing the interface and preparing configurations for when the implementation is complete.

## Overview

Model quantization converts neural network weights from higher precision (e.g., float32) to lower precision (e.g., 4-bit, 8-bit) representations. This reduces:

- **Model file size** (2x to 8x compression)
- **Memory usage** during inference
- **Inference latency** (especially on CPU)

The trade-off is a small reduction in model accuracy, which is often acceptable for deployment scenarios.

## Quick Start

### Testing the CLI (Current Status)

Test the quantization interface with a HuggingFace model:

```bash
oumi quantize --method q4_0 --model meta-llama/Llama-2-7b-hf --output llama2-7b-q4.gguf
```

Expected output (simulation mode):
```
Starting quantization of model: meta-llama/Llama-2-7b-hf
Quantization method: q4_0
Output path: llama2-7b-q4.gguf
WARNING: Quantization feature is currently in development.
INFO: Current implementation requires additional dependencies and setup.
INFO: Simulating quantization process for validation...
✅ Model quantized successfully!
📁 Output saved to: llama2-7b-q4.gguf
📊 Status: simulated
```

Test with a local model:

```bash
oumi quantize --method q8_0 --model ./my_model --output ./quantized/model.gguf
```

### Using Configuration Files

Create a configuration file `quantize_config.yaml`:

```yaml
# Model to quantize
model:
  model_name: "meta-llama/Llama-2-7b-hf"
  tokenizer_name: "meta-llama/Llama-2-7b-hf"

# Quantization settings
method: "q4_0"
output_path: "models/llama2-7b-q4.gguf"
output_format: "gguf"
verbose: true
```

Test with configuration file:

```bash
oumi quantize --config quantize_config.yaml
```

Override config settings:

```bash
oumi quantize --config quantize_config.yaml --method q8_0 --output different_output.gguf
```

**Note:** All commands currently run in simulation mode and validate inputs without performing actual quantization.

## Quantization Methods

### GGUF-Compatible Methods (for llama.cpp)

| Method | Bits | Compression | Quality | Use Case |
|--------|------|-------------|---------|----------|
| `q4_0` | 4-bit | ~4x | Good | General purpose, balanced size/quality |
| `q4_1` | 4-bit | ~4x | Better | Improved 4-bit with bias terms |
| `q5_0` | 5-bit | ~3x | Very Good | Better quality than 4-bit |
| `q5_1` | 5-bit | ~3x | Excellent | Best 5-bit quality |
| `q8_0` | 8-bit | ~2x | Excellent | Minimal quality loss |

### Precision Methods

| Method | Description | Compression | Use Case |
|--------|-------------|-------------|----------|
| `f16` | 16-bit float | ~2x | GPU inference, good quality |
| `f32` | 32-bit float | 1x | Format conversion only |

### Recommendations

- **General deployment**: Use `q4_0` for good balance of size and quality
- **Quality-sensitive applications**: Use `q8_0` for minimal quality loss
- **Size-constrained environments**: Use `q4_0` or `q4_1`
- **GPU inference**: Use `f16` with safetensors format

## Output Formats

### GGUF Format

Best for CPU inference and edge deployment:

```yaml
output_format: "gguf"
output_path: "model.gguf"
```

**Advantages:**
- Single file with metadata
- Optimized for llama.cpp
- Fast CPU inference
- Memory mapping support

**Use with:** llama.cpp, llama-cpp-python, Ollama

### Safetensors Format

Best for GPU inference with HuggingFace:

```yaml
output_format: "safetensors"
output_path: "quantized_model/"
```

**Advantages:**
- Safe serialization
- HuggingFace compatible
- Good for GPU inference
- Supports BitsAndBytes quantization

**Use with:** HuggingFace transformers, vLLM

### PyTorch Format

Native PyTorch serialization:

```yaml
output_format: "pytorch"
output_path: "quantized_model/"
```

**Advantages:**
- Native PyTorch support
- Research-friendly
- Easy integration

**Use with:** PyTorch, custom inference code

## Configuration Examples

### High-Quality GGUF for Production

```yaml
model:
  model_name: "meta-llama/Llama-2-13b-chat-hf"
  tokenizer_name: "meta-llama/Llama-2-13b-chat-hf"

method: "q8_0"  # Minimal quality loss
output_path: "production/llama2-13b-chat-q8.gguf"
output_format: "gguf"
verbose: true
```

### Compact Model for Edge Deployment

```yaml
model:
  model_name: "microsoft/DialoGPT-small"

method: "q4_0"  # Maximum compression
output_path: "edge/dialogpt-small-q4.gguf"
output_format: "gguf"
batch_size: 16
verbose: false
```

### GPU Inference with Safetensors

```yaml
model:
  model_name: "mistralai/Mistral-7B-Instruct-v0.1"

method: "q8_0"
output_path: "gpu_models/mistral-7b-instruct/"
output_format: "safetensors"
verbose: true
```

### Local Model Quantization

```yaml
model:
  model_name: "./my_fine_tuned_model"
  tokenizer_name: "meta-llama/Llama-2-7b-hf"  # Base tokenizer

method: "q4_1"
output_path: "quantized/my_model_q4.gguf"
output_format: "gguf"
```

## Advanced Usage

### Batch Processing Multiple Models

Create a script to quantize multiple models:

```bash
#!/bin/bash

# Quantize different sizes of the same model family
models=(
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
)

for model in "${models[@]}"; do
  echo "Quantizing $model..."
  oumi quantize \
    --method q4_0 \
    --model "$model" \
    --output "quantized/$(basename $model)-q4.gguf"
done
```

### Custom Quantization Pipeline

```yaml
# Advanced configuration with custom settings
model:
  model_name: "codellama/CodeLlama-7b-Python-hf"
  model_kwargs:
    torch_dtype: "float16"
    trust_remote_code: true

method: "q5_0"  # Good balance for code models
output_path: "code_models/codellama-7b-python-q5.gguf"
output_format: "gguf"
batch_size: 8  # Reduce for lower memory usage
verbose: true
```

### Integration with Inference

After quantization, use the quantized model for inference:

```yaml
# inference_config.yaml
model:
  model_name: "quantized/llama2-7b-q4.gguf"
  tokenizer_name: "meta-llama/Llama-2-7b-hf"
  model_kwargs:
    filename: "llama2-7b-q4.gguf"  # For GGUF files

engine: "VLLM"  # or "LLAMACPP" for GGUF files

generation:
  max_new_tokens: 256
  temperature: 0.7
```

## Performance Considerations

### Memory Requirements

Quantization requires loading the full model in memory:

- **7B model**: ~14GB RAM (float16) + output space
- **13B model**: ~26GB RAM (float16) + output space
- **70B model**: ~140GB RAM (float16) + output space

### Processing Time

Quantization time depends on model size and method:

- **7B model, q4_0**: ~10-30 minutes
- **13B model, q8_0**: ~30-60 minutes
- **70B model**: Several hours

### Disk Space

Ensure sufficient disk space:

- Original model size + quantized model size
- Temporary files during processing
- GGUF conversion may need 2x model size temporarily

## Troubleshooting

### Common Issues

**Out of memory errors:**
```bash
# Reduce batch size or use smaller model
oumi quantize --config config.yaml --batch-size 4
```

**Missing dependencies:**
```bash
# Install required packages
pip install transformers torch
pip install llama-cpp-python  # For GGUF
pip install bitsandbytes     # For safetensors quantization
```

**Model not found:**
```bash
# Verify model path or identifier
oumi quantize --model meta-llama/Llama-2-7b-hf --method q4_0 --output test.gguf
```

### Performance Optimization

**For large models:**
- Use smaller batch sizes
- Enable verbose logging to monitor progress
- Use SSD storage for better I/O performance

**For faster quantization:**
- Use CUDA-enabled quantization when available
- Prefer f16/f32 methods for format conversion
- Use local models to avoid download time

## Integration Examples

### Use with llama.cpp

```bash
# After quantization
./llama.cpp/main -m quantized/model.gguf -p "Hello, world!"
```

### Use with Python

```python
from llama_cpp import Llama

# Load quantized GGUF model
llm = Llama(model_path="quantized/model.gguf")
response = llm("Hello, world!", max_tokens=50)
print(response['choices'][0]['text'])
```

### Use with Oumi Inference

```bash
# Create inference config for quantized model
oumi infer --config inference_quantized.yaml
```

## Development Roadmap

The quantization feature is being developed in phases:

### Phase 1: Interface and Configuration ✅ Complete
- [x] CLI command structure (`oumi quantize`)
- [x] Configuration system (`QuantizationConfig`)
- [x] Input validation and error handling
- [x] Documentation and examples
- [x] Integration with Oumi CLI framework

### Phase 2: Core Implementation 🚧 In Progress
- [ ] GGUF format support via llama.cpp integration
- [ ] Safetensors format with BitsAndBytes quantization
- [ ] PyTorch format with torch quantization
- [ ] Memory-efficient model loading and processing
- [ ] Progress tracking and error recovery

### Phase 3: Advanced Features 📋 Planned
- [ ] Batch quantization of multiple models
- [ ] Custom quantization calibration datasets
- [ ] Quality assessment and benchmarking
- [ ] Integration with Oumi model registry
- [ ] Distributed quantization for large models

### Phase 4: Production Features 📋 Future
- [ ] Quantization pipeline automation
- [ ] Model format conversion utilities
- [ ] Performance optimization and caching
- [ ] Cloud-based quantization services

## Contributing to Development

The quantization implementation welcomes contributions! Key areas where help is needed:

1. **llama.cpp Integration**: Implementing GGUF conversion using llama.cpp tools
2. **Memory Management**: Efficient handling of large models during quantization
3. **Format Support**: Adding support for additional quantization formats
4. **Testing**: Comprehensive testing with various model architectures
5. **Documentation**: Expanding usage examples and troubleshooting guides

## Current Limitations

While in development, the quantization feature has these limitations:

- **Simulation Mode**: Validates inputs but doesn't perform actual quantization
- **Dependencies**: Requires additional setup for llama.cpp, transformers, etc.
- **Large Models**: Memory management not yet optimized for very large models
- **Format Coverage**: Not all quantization methods fully implemented

## Getting Involved

To contribute or stay updated on quantization development:

1. **Test the Interface**: Use simulation mode to validate your configurations
2. **Prepare Configurations**: Create and test config files for your models
3. **Report Issues**: Submit feedback on CLI usability and configuration options
4. **Contribute Code**: Help implement core quantization functionality

This guide provides comprehensive coverage of the quantization functionality. For more advanced use cases, development updates, or troubleshooting, refer to the API documentation or reach out to the Oumi community.
