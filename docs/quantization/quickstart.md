# Quantization Quick Start Guide

This guide will get you started with model quantization in just a few minutes.

## Prerequisites

```bash
# Basic dependencies (usually already installed)
pip install torch transformers

# Choose one quantization backend:
pip install autoawq        # For AWQ (recommended)
# OR
pip install bitsandbytes   # For BitsAndBytes

# Optional: GGUF output support
pip install llama-cpp-python
```

## 1. Your First Quantization

Start with a small model to test the setup:

```bash
oumi quantize \
  --method awq_q4_0 \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --output my_first_quantized_model.gguf
```

**Expected output:**
```
✅ AWQ quantization completed successfully!
⚠️  GGUF conversion failed - saved as PyTorch format instead
� For GGUF output, install: pip install llama-cpp-python
� Output saved to: my_first_quantized_model.pytorch
� Original size: Unknown
� Output size: 734.2 MB
�️  Compression ratio: Unknown
```

## 2. Understanding the Command

```bash
oumi quantize \
  --method awq_q4_0 \              # Quantization method
  --model "TinyLlama/..." \        # Model to quantize
  --output model.gguf              # Output file
```

### Key Parameters:
- **`--method`**: Quantization algorithm (`awq_q4_0`, `bnb_4bit`, etc.)
- **`--model`**: HuggingFace model ID or local path
- **`--output`**: Output file/directory path

## 3. Common Quantization Scenarios

### Scenario 1: High-Quality Quantization
For production use where quality is critical:

```bash
oumi quantize \
  --method awq_q8_0 \
  --model "meta-llama/Llama-2-7b-hf" \
  --output llama2-high-quality.gguf
```
*Results in 2x compression with minimal quality loss*

### Scenario 2: Maximum Compression
For resource-constrained environments:

```bash
oumi quantize \
  --method awq_q4_0 \
  --model "meta-llama/Llama-2-7b-hf" \
  --output llama2-compact.gguf
```
*Results in 4x compression with good quality*

### Scenario 3: GPU-Optimized Quantization
For GPU inference with PyTorch:

```bash
oumi quantize \
  --method bnb_4bit \
  --model "microsoft/DialoGPT-medium" \
  --output dialogpt-gpu.pytorch
```

### Scenario 4: Unsupported Model Fallback
For models not supported by AWQ:

```bash
oumi quantize \
  --method bnb_4bit \
  --model "microsoft/DialoGPT-small" \
  --output gpt2-model.pytorch
```

## 4. Using Configuration Files

For complex setups, use YAML configuration:

Create `config.yaml`:
```yaml
model:
  model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

method: "awq_q4_0"
output_path: "configured_model.gguf"
output_format: "gguf"

calibration_samples: 512
verbose: true
```

Run:
```bash
oumi quantize --config config.yaml
```

## 5. Verify Your Results

Check the quantized model:

```bash
# Check file size
ls -lh my_first_quantized_model.gguf

# For PyTorch format, check directory
ls -la model_name.pytorch/
```

## 6. Common Issues & Solutions

### Issue: "gpt2 isn't supported yet"
**Solution:** Use BitsAndBytes instead:
```bash
oumi quantize --method bnb_4bit --model "microsoft/DialoGPT-small" --output model.pytorch
```

### Issue: "llama-cpp-python not available"
**Solution:** Either install it or use PyTorch format:
```bash
# Option 1: Install dependency
pip install llama-cpp-python

# Option 2: Use PyTorch format
oumi quantize --method awq_q4_0 --model "model_name" --output model.pytorch
```

### Issue: "CUDA out of memory"
**Solution:** Use a smaller model or adjust parameters:
```bash
# Try with TinyLlama first
oumi quantize --method awq_q4_0 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output test.gguf
```

## 7. Next Steps

Once you've successfully quantized your first model:

1. **Try different methods:** Experiment with `awq_q8_0`, `bnb_4bit`, etc.
2. **Scale to larger models:** Move from TinyLlama to Llama-2 7B
3. **Optimize parameters:** Adjust `calibration_samples` and `awq_group_size`
4. **Integrate with your application:** Use the quantized models in your inference pipeline

## 8. Performance Expectations

| Model Size | Method | Original | Quantized | Ratio | Time |
|------------|--------|----------|-----------|-------|------|
| TinyLlama 1.1B | awq_q4_0 | 2.2 GB | 661 MB | 3.3x | ~5 min |
| Llama-2 7B | awq_q4_0 | 13.5 GB | 3.9 GB | 3.5x | ~15 min |
| Llama-2 7B | awq_q8_0 | 13.5 GB | 7.2 GB | 1.9x | ~10 min |

*Times are approximate and depend on hardware*

## Ready to Quantize!

You now have everything needed to start quantizing models. Begin with the TinyLlama example above and gradually work your way up to larger models as you become more comfortable with the process.

For more advanced usage, see the [full documentation](README.md).