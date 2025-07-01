# AWQ Quantization Testing Examples

This directory contains comprehensive testing examples for the AWQ (Activation-aware Weight Quantization) implementation in Oumi.

## Overview

The AWQ quantization system provides high-quality model compression using activation-aware algorithms that preserve model accuracy better than traditional quantization methods.

## Test Files

### Configuration Files

- **`basic_awq_test.yaml`** - Simple AWQ test with small model for quick validation
- **`production_awq_llama2.yaml`** - Production-ready configuration for Llama 2 7B
- **`high_quality_awq_q8.yaml`** - High-quality 8-bit quantization for minimal accuracy loss  
- **`fast_awq_f16.yaml`** - Fast F16 conversion with AWQ optimizations
- **`edge_deployment_awq.yaml`** - Maximum compression for resource-constrained environments

### Legacy Configuration Files
- **`basic_quantize_config.yaml`** - Basic direct GGUF quantization
- **`advanced_quantize_config.yaml`** - Advanced direct quantization options
- **`safetensors_quantize_config.yaml`** - Safetensors format quantization

### Test Scripts

- **`test_awq_examples.sh`** - Bash script running multiple CLI-based tests
- **`python_test_examples.py`** - Python script for API-based testing

## AWQ Dependencies and Modes

The AWQ quantization system supports three operating modes with automatic fallback:

### Mode 1: Full AWQ Quantization (Recommended)
**Requirements:** `autoawq`, `torch`, `transformers`, `accelerate`
**Platforms:** Linux + CUDA, Windows + CUDA
**Quality:** Highest (activation-aware calibration)

```bash
# Install for full AWQ support
pip install autoawq torch transformers accelerate

# Optional: GGUF output support
pip install llama-cpp-python
```

### Mode 2: BitsAndBytes Fallback (Production Alternative)
**Requirements:** `bitsandbytes`, `torch`, `transformers`
**Platforms:** macOS, Linux (CPU), Windows (CPU), ARM64
**Quality:** Comparable to AWQ (4-bit NF4 quantization)

```bash
# Install for BitsAndBytes fallback
pip install bitsandbytes torch transformers
```

### Mode 3: Simulation Mode (Development/Testing)
**Requirements:** Only Oumi base dependencies
**Platforms:** Any platform
**Purpose:** Interface testing, configuration validation, CI/CD

The system automatically detects available dependencies and selects the best available mode (`src/oumi/quantize.py:432-478`).

## Quick Start

### CLI Testing

Run all CLI-based tests (works in both simulation and real modes):

```bash
cd /Users/yuzhangshang/oumi
./examples/quantization/test_awq_examples.sh
```

Run individual configuration tests:

```bash
# Basic test with small model
oumi quantize --config examples/quantization/basic_awq_test.yaml

# Production Llama 2 test
oumi quantize --config examples/quantization/production_awq_llama2.yaml

# High-quality Q8 test  
oumi quantize --config examples/quantization/high_quality_awq_q8.yaml
```

### Python API Testing

Run Python-based tests:

```bash
cd /Users/yuzhangshang/oumi
python examples/quantization/python_test_examples.py
```

## AWQ Methods Available

| Method | Description | Use Case |
|--------|-------------|----------|
| `awq_q4_0` | AWQ 4-bit → GGUF q4_0 | Best balance of quality and compression (default) |
| `awq_q4_1` | AWQ 4-bit → GGUF q4_1 | Improved 4-bit accuracy with bias terms |
| `awq_q8_0` | AWQ 8-bit → GGUF q8_0 | Minimal quality loss, good compression |
| `awq_f16` | AWQ → GGUF f16 | Format optimization with AWQ preprocessing |

## Legacy Methods (Direct GGUF)

- **Methods**: q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32
- **Formats**: gguf, safetensors, pytorch
- **Models**: HuggingFace IDs, local paths, Oumi registry

## Simulation Mode

When AutoAWQ dependencies are not installed, AWQ methods automatically run in simulation mode:

### What Simulation Mode Does
- ✅ Validates all CLI parameters and configuration
- ✅ Tests model identifier resolution
- ✅ Creates realistic mock output files with proper GGUF headers
- ✅ Estimates output file sizes based on model and method
- ✅ Provides complete user experience testing

### Simulation Mode Output
```bash
$ oumi quantize --method awq_q4_0 --model microsoft/DialoGPT-small --output test.gguf

🔧 AWQ quantization completed (SIMULATION MODE)
⚠️  AWQ dependencies not installed - created mock output for testing
💡 Install autoawq for real quantization: pip install autoawq
📁 Output saved to: test.gguf
🎭 Mode: Simulation
📦 Method: SIMULATED: AWQ → GGUF (awq_q4_0)
📉 Output size: 30.0 MB
```

### Mock File Characteristics
- **GGUF Headers**: Proper GGUF magic numbers and version
- **Realistic Sizes**: Based on actual model compression ratios
- **File Structure**: Valid file format for testing pipelines

## Development Progress

Track the implementation progress in the [main documentation](../../docs/quantization_guide.md#development-roadmap).

## Contributing

Help complete the quantization implementation:

1. **Test Configurations**: Try different model types and configurations
2. **Report Issues**: Submit feedback on CLI usability
3. **Contribute Code**: Help implement core quantization features
4. **Documentation**: Improve examples and guides

For more details, see the [Quantization Guide](../../docs/quantization_guide.md).
