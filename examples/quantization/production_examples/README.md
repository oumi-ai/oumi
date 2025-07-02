# Production Quantization Examples

This directory contains production-ready quantization configurations and scripts for common deployment scenarios.

## Quick Reference

| Use Case | Configuration | Method | Compression | Quality |
|----------|---------------|--------|-------------|---------|
| High-quality inference | `high_quality.yaml` | awq_q8_0 | 2x | Excellent |
| Balanced deployment | `balanced.yaml` | awq_q4_0 | 4x | Good |
| Edge/mobile deployment | `edge.yaml` | awq_q4_0 | 4x | Acceptable |
| GPU inference | `gpu_optimized.yaml` | bnb_4bit | 4x | Good |

## Configuration Files

### High-Quality Production (`high_quality.yaml`)

Best quality with moderate compression for production systems with sufficient resources.

```yaml
model:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
  tokenizer_name: "meta-llama/Llama-2-7b-chat-hf"
  model_kwargs:
    torch_dtype: "auto"
    device_map: "auto"

method: "awq_q8_0"
output_path: "models/production/llama2-7b-chat-hq.gguf"
output_format: "gguf"

# High-quality AWQ settings
awq_group_size: 64           # Smaller groups for better accuracy
calibration_samples: 1024    # More samples for better calibration
awq_zero_point: true
awq_version: "GEMM"
cleanup_temp: true

# Performance settings
batch_size: 8                # Conservative for stability
verbose: true
```

**Usage:**
```bash
oumi quantize --config high_quality.yaml
```

**Expected Results:**
- Original: ~13.5 GB
- Quantized: ~7.2 GB
- Compression: 1.9x
- Quality: Minimal degradation

### Balanced Production (`balanced.yaml`)

Good balance of quality and compression for most production use cases.

```yaml
model:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
  tokenizer_name: "meta-llama/Llama-2-7b-chat-hf"

method: "awq_q4_0"
output_path: "models/production/llama2-7b-chat-balanced.gguf"
output_format: "gguf"

# Balanced AWQ settings
awq_group_size: 128          # Standard group size
calibration_samples: 512     # Good calibration quality
awq_zero_point: true
awq_version: "GEMM"
cleanup_temp: true

# Balanced performance
batch_size: 16
verbose: true
```

**Expected Results:**
- Original: ~13.5 GB
- Quantized: ~3.9 GB
- Compression: 3.5x
- Quality: Good with minor degradation

### Edge Deployment (`edge.yaml`)

Maximum compression for resource-constrained environments.

```yaml
model:
  model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  tokenizer_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

method: "awq_q4_0"
output_path: "models/edge/tinyllama-edge.gguf"
output_format: "gguf"

# Edge-optimized settings
awq_group_size: 128
calibration_samples: 256     # Fewer samples for speed
awq_zero_point: true
cleanup_temp: true

# Memory-efficient settings
batch_size: 4
verbose: false
```

**Expected Results:**
- Original: ~2.2 GB
- Quantized: ~661 MB
- Compression: 3.3x
- Quality: Good for edge deployment

### GPU-Optimized (`gpu_optimized.yaml`)

Optimized for GPU inference with PyTorch/HuggingFace ecosystem.

```yaml
model:
  model_name: "microsoft/DialoGPT-medium"
  tokenizer_name: "microsoft/DialoGPT-medium"
  model_kwargs:
    torch_dtype: "float16"
    device_map: "auto"

method: "bnb_4bit"
output_path: "models/gpu/dialogpt-medium-4bit.pytorch"
output_format: "pytorch"

# GPU-optimized settings
batch_size: 32
verbose: true
```

**Expected Results:**
- Works with most GPU setups
- Compatible with HuggingFace transformers
- Direct GPU memory optimization

## Batch Processing Scripts

### `batch_quantize.py`

Process multiple models with different configurations:

```python
#!/usr/bin/env python3
"""
Batch quantization script for production deployment.
"""

from oumi.core.configs import QuantizationConfig, ModelParams
from oumi import quantize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Production model configurations
MODELS = [
    {
        "name": "TinyLlama",
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "methods": ["awq_q4_0", "awq_q8_0"],
        "output_dir": "models/tinyllama/"
    },
    {
        "name": "Llama-2-7B",
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "methods": ["awq_q4_0", "awq_q8_0"],
        "output_dir": "models/llama2-7b/"
    }
]

def quantize_model(model_config, method):
    """Quantize a single model with specified method."""
    config = QuantizationConfig(
        model=ModelParams(model_name=model_config["model_name"]),
        method=method,
        output_path=f"{model_config['output_dir']}{model_config['name']}-{method}.gguf",
        output_format="gguf",
        calibration_samples=512,
        verbose=True
    )
    
    try:
        result = quantize(config)
        print(f"✅ {model_config['name']} {method}: {result['compression_ratio']}")
        return True
    except Exception as e:
        print(f"❌ {model_config['name']} {method}: {e}")
        return False

def main():
    """Run batch quantization."""
    results = []
    
    for model_config in MODELS:
        for method in model_config["methods"]:
            success = quantize_model(model_config, method)
            results.append((model_config["name"], method, success))
    
    # Summary
    print("\n📊 Batch Quantization Results:")
    for name, method, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name} {method}")

if __name__ == "__main__":
    main()
```

### `validate_quantized.py`

Validate quantized models for production deployment:

```python
#!/usr/bin/env python3
"""
Validation script for quantized models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def validate_pytorch_model(model_path):
    """Validate a PyTorch quantized model."""
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Test inference
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Model validation successful")
        print(f"   Sample output: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def validate_gguf_model(model_path):
    """Validate a GGUF model file."""
    try:
        # Check file exists and has GGUF header
        with open(model_path, 'rb') as f:
            header = f.read(4)
            if header != b'GGUF':
                print(f"❌ Invalid GGUF header: {header}")
                return False
        
        # Check file size is reasonable
        size = Path(model_path).stat().st_size
        if size < 1024 * 1024:  # Less than 1MB is suspicious
            print(f"❌ File too small: {size} bytes")
            return False
        
        print(f"✅ GGUF validation successful ({size // (1024*1024)} MB)")
        return True
        
    except Exception as e:
        print(f"❌ GGUF validation failed: {e}")
        return False

def main():
    """Validate all quantized models."""
    model_dir = Path("models")
    
    # Find all quantized models
    pytorch_models = list(model_dir.glob("**/*.pytorch"))
    gguf_models = list(model_dir.glob("**/*.gguf"))
    
    print(f"📁 Found {len(pytorch_models)} PyTorch models, {len(gguf_models)} GGUF models")
    
    # Validate each model
    for model_path in pytorch_models:
        print(f"\n🔍 Validating {model_path}")
        validate_pytorch_model(model_path)
    
    for model_path in gguf_models:
        print(f"\n🔍 Validating {model_path}")
        validate_gguf_model(model_path)

if __name__ == "__main__":
    main()
```

## Deployment Guidelines

### 1. Model Selection

- **TinyLlama:** Edge devices, mobile, testing
- **Llama-2 7B:** General production, servers
- **Larger models:** High-performance requirements

### 2. Method Selection

- **awq_q8_0:** Production systems requiring highest quality
- **awq_q4_0:** Balanced production deployment (recommended)
- **bnb_4bit:** GPU inference, unsupported model architectures

### 3. Infrastructure Requirements

| Model Size | Method | RAM Required | GPU Memory | Storage |
|------------|--------|--------------|------------|---------|
| TinyLlama 1.1B | awq_q4_0 | 2 GB | 1 GB | 661 MB |
| Llama-2 7B | awq_q4_0 | 8 GB | 4 GB | 3.9 GB |
| Llama-2 7B | awq_q8_0 | 12 GB | 8 GB | 7.2 GB |

### 4. Quality Validation

Always validate quantized models:

1. **Functional testing:** Ensure model loads and generates text
2. **Quality testing:** Compare outputs with original model
3. **Performance testing:** Measure inference speed and memory usage
4. **Integration testing:** Test in production environment

### 5. Monitoring

Monitor quantized models in production:

- **Response quality:** Track user satisfaction
- **Performance metrics:** Latency, throughput
- **Resource usage:** Memory, CPU, GPU utilization
- **Error rates:** Model failures, OOM errors

## Troubleshooting

### Common Production Issues

1. **Model fails to load**
   - Check file integrity
   - Verify sufficient memory
   - Validate file format

2. **Poor quality outputs**
   - Try higher precision method (q8_0 vs q4_0)
   - Increase calibration samples
   - Validate input preprocessing

3. **Performance issues**
   - Check hardware compatibility
   - Optimize batch sizes
   - Consider different output format

### Emergency Rollback

Keep original models available for quick rollback:

```bash
# Quick rollback script
cp models/backup/original_model/* models/production/
systemctl restart inference_service
```

## Best Practices

1. **Version control:** Track quantization configurations
2. **Testing pipeline:** Automated validation before deployment
3. **Gradual rollout:** A/B test quantized models
4. **Monitoring:** Continuous quality and performance monitoring
5. **Documentation:** Document model versions and configurations