# Llama 2 7B AWQ Quantization Example

This example demonstrates AWQ quantization on Llama 2 7B model, showing both simulation mode and real quantization workflows.

## Files Created

### 1. Configuration File: `llama2_7b_awq_example.yaml`
Production-ready configuration for Llama 2 7B AWQ quantization:

```yaml
# Model configuration
model:
  model_name: "meta-llama/Llama-2-7b-hf"
  tokenizer_name: "meta-llama/Llama-2-7b-hf"
  model_kwargs:
    torch_dtype: "auto"
    device_map: "auto"
    trust_remote_code: false

# AWQ quantization settings
method: "awq_q4_0"              # 4-bit AWQ quantization
output_path: "models/llama2-7b-awq-q4.gguf"
output_format: "gguf"

# AWQ-specific parameters
awq_group_size: 128             # Standard group size
awq_zero_point: true            # Better accuracy
awq_version: "GEMM"             # Faster kernels
calibration_samples: 512        # Quality/speed balance
cleanup_temp: true              # Save disk space

# Performance settings
batch_size: 8
verbose: true
```

### 2. Variants Configuration: `llama2_awq_variants.yaml`
Shows different AWQ methods for different use cases:

- **Variant 1: Maximum Quality (Q8)** - ~7GB output, minimal quality loss
- **Variant 2: Maximum Compression (Q4_0)** - ~4GB output, best balance (default)
- **Variant 3: Enhanced Quality (Q4_1)** - ~4GB output, better than Q4_0
- **Variant 4: Fast Conversion (F16)** - ~13GB output, fastest conversion
- **Variant 5: Memory-Constrained** - Optimized for limited RAM/VRAM

### 3. Automation Script: `run_llama2_awq_example.sh`
Complete automation with dependency checking and result analysis.

## Running the Example

### Option 1: Simulation Mode (No Dependencies Required)

```bash
# Run simulation - tests interface without real quantization
oumi quantize --config examples/quantization/llama2_7b_awq_example.yaml
```

**Expected Output (Simulation Mode):**
```
🔧 SIMULATION MODE: AWQ quantization simulation
   Model: meta-llama/Llama-2-7b-hf
   Method: awq_q4_0
   Output: models/llama2-7b-awq-q4.gguf
   AWQ Group Size: 128
   Calibration Samples: 512

✅ SIMULATION: AWQ quantization completed successfully!
📁 SIMULATION: Mock output created at: models/llama2-7b-awq-q4.gguf
📊 SIMULATION: Mock file size: 4.0 GB

🔧 AWQ quantization completed (SIMULATION MODE)
⚠️  AWQ dependencies not installed - created mock output for testing
💡 Install autoawq for real quantization: pip install autoawq
📁 Output saved to: models/llama2-7b-awq-q4.gguf
🎭 Mode: Simulation
📦 Method: SIMULATED: AWQ → GGUF (awq_q4_0)
📉 Output size: 4.0 GB
```

### Option 2: Real Quantization (Dependencies Required)

```bash
# Install AWQ dependencies
pip install autoawq torch transformers accelerate

# Run real quantization
oumi quantize --config examples/quantization/llama2_7b_awq_example.yaml
```

**Expected Output (Real Mode):**
```
Starting quantization of model: meta-llama/Llama-2-7b-hf
Quantization method: awq_q4_0
Output path: models/llama2-7b-awq-q4.gguf

AWQ library found: autoawq 0.2.0
CUDA available: NVIDIA GeForce RTX 4090

Loading model for AWQ quantization: meta-llama/Llama-2-7b-hf
Configuring AWQ quantization parameters...
AWQ config: {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'}
Performing AWQ quantization...
[Progress bars and detailed logging...]

Converting AWQ model to GGUF...
✅ Model quantized successfully!
📁 Output saved to: models/llama2-7b-awq-q4.gguf
📊 Original size: 13.5 GB
📉 Quantized size: 3.9 GB
🗜️  Compression ratio: 3.46x
```

## What Simulation Mode Shows You

### 1. Interface Validation
- ✅ All CLI parameters work correctly
- ✅ Configuration file is valid
- ✅ Model identifier resolves properly
- ✅ Output paths are accessible

### 2. Realistic Output
- Creates 4GB mock file for Llama 2 7B Q4 quantization
- Proper GGUF headers (magic number: "GGUF", version: 3)
- File can be inspected with GGUF tools
- Size estimates match real quantization results

### 3. Configuration Testing
- Tests all AWQ-specific parameters
- Validates calibration settings
- Checks batch size and memory configuration
- Verifies cleanup and output options

## Use Cases for Each Mode

### Simulation Mode Use Cases:
1. **Development**: Test configurations before investing time/resources
2. **CI/CD**: Validate interface in automated testing
3. **Learning**: Understand the quantization pipeline
4. **Configuration Testing**: Try different settings quickly

### Real Quantization Use Cases:
1. **Production**: Create actual quantized models for deployment
2. **Benchmarking**: Compare quality/performance of different methods
3. **Research**: Experiment with AWQ parameters
4. **Deployment**: Generate models for inference

## Memory and Time Requirements

### Simulation Mode:
- **Memory**: <1GB RAM
- **Time**: 10-30 seconds
- **Dependencies**: Only Oumi
- **Output**: Mock 4GB GGUF file

### Real Quantization Mode:
- **Memory**: 16-24GB RAM, 8-12GB VRAM (with CUDA)
- **Time**: 20-45 minutes (GPU), 2-4 hours (CPU)
- **Dependencies**: autoawq, torch, transformers, accelerate
- **Output**: Real 3.9GB quantized Llama 2 7B model

## File Structure After Running

```
models/
├── llama2-7b-awq-q4.gguf           # Main output (4GB)
├── llama2-7b-awq-q4_0.gguf         # Q4_0 variant 
├── llama2-7b-awq-q4_1.gguf         # Q4_1 variant
├── llama2-7b-awq-q8_0.gguf         # Q8_0 variant (7GB)
└── llama2-7b-awq-f16.gguf          # F16 variant (13GB)

test_outputs/
└── [temporary test files]
```

## Quality Expectations

Based on typical AWQ results for Llama 2 7B:

| Method | Size | Perplexity | Use Case |
|--------|------|------------|----------|
| Original | 13.5GB | Baseline | Reference |
| awq_q8_0 | ~7GB | +0.5% | High quality |
| awq_q4_1 | ~4GB | +2-3% | Enhanced 4-bit |
| awq_q4_0 | ~4GB | +3-4% | Balanced (recommended) |
| awq_f16 | ~7GB | ~0% | Format conversion |

## Next Steps

1. **Try simulation mode** to familiarize yourself with the interface
2. **Install dependencies** when ready for real quantization
3. **Experiment with variants** to find the best quality/size trade-off
4. **Integrate with inference** using llama.cpp or Oumi inference

This example provides a complete workflow from configuration to deployment, with both testing and production scenarios covered.