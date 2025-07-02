# QuantizationConfig API Reference

The `QuantizationConfig` class defines all parameters for model quantization operations.

## Class Definition

```python
from oumi.core.configs import QuantizationConfig, ModelParams

config = QuantizationConfig(
    model=ModelParams(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    method="awq_q4_0",
    output_path="model.gguf",
    output_format="gguf"
)
```

## Parameters

### Core Parameters

#### `model: ModelParams`
- **Type:** `ModelParams`
- **Required:** Yes
- **Description:** Configuration for the model to be quantized

```python
model = ModelParams(
    model_name="meta-llama/Llama-2-7b-hf",  # Required
    tokenizer_name="meta-llama/Llama-2-7b-hf",  # Optional
    model_kwargs={"torch_dtype": "auto"}  # Optional
)
```

#### `method: str`
- **Type:** `str`
- **Default:** `"awq_q4_0"`
- **Description:** Quantization method to use

**Available methods:**
```python
# AWQ methods (recommended)
"awq_q4_0"  # 4-bit AWQ → GGUF q4_0 (default)
"awq_q4_1"  # 4-bit AWQ → GGUF q4_1
"awq_q8_0"  # 8-bit AWQ → GGUF q8_0
"awq_f16"   # AWQ → GGUF f16

# BitsAndBytes methods
"bnb_4bit"  # 4-bit BitsAndBytes
"bnb_8bit"  # 8-bit BitsAndBytes

# Direct GGUF methods
"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"
```

#### `output_path: str`
- **Type:** `str`
- **Default:** `"quantized_model.gguf"`
- **Description:** Path where the quantized model will be saved

```python
output_path = "models/llama2-7b-q4.gguf"        # GGUF file
output_path = "models/llama2-7b-quantized/"     # Directory
output_path = "model.pytorch"                   # PyTorch format
```

#### `output_format: str`
- **Type:** `str`
- **Default:** `"gguf"`
- **Description:** Output format for the quantized model

**Available formats:**
```python
"gguf"        # GGUF format (recommended for inference)
"pytorch"     # PyTorch format
"safetensors" # Safetensors format
```

### AWQ-Specific Parameters

#### `awq_group_size: int`
- **Type:** `int`
- **Default:** `128`
- **Description:** Group size for AWQ quantization
- **Range:** `32-256`

```python
awq_group_size = 64   # Higher accuracy, slower
awq_group_size = 128  # Balanced (default)
awq_group_size = 256  # Faster, potentially lower accuracy
```

#### `awq_zero_point: bool`
- **Type:** `bool`
- **Default:** `True`
- **Description:** Enable zero-point quantization for AWQ

#### `awq_version: str`
- **Type:** `str`
- **Default:** `"GEMM"`
- **Description:** AWQ kernel version
- **Options:** `"GEMM"`, `"GEMV"`

#### `calibration_samples: int`
- **Type:** `int`
- **Default:** `512`
- **Description:** Number of calibration samples for AWQ
- **Range:** `128-2048`

```python
calibration_samples = 128   # Fast, lower quality
calibration_samples = 512   # Balanced (default)
calibration_samples = 1024  # Slower, higher quality
```

#### `cleanup_temp: bool`
- **Type:** `bool`
- **Default:** `True`
- **Description:** Remove temporary AWQ files after conversion

### General Parameters

#### `batch_size: Optional[int]`
- **Type:** `Optional[int]`
- **Default:** `None` (automatic)
- **Description:** Batch size for quantization process

```python
batch_size = None  # Automatic sizing
batch_size = 8     # Conservative
batch_size = 16    # Balanced
batch_size = 32    # Aggressive (requires more memory)
```

#### `verbose: bool`
- **Type:** `bool`
- **Default:** `False`
- **Description:** Enable detailed logging

## Configuration Examples

### Basic Configuration

```python
config = QuantizationConfig(
    model=ModelParams(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    method="awq_q4_0",
    output_path="tinyllama-q4.gguf"
)
```

### High-Quality Configuration

```python
config = QuantizationConfig(
    model=ModelParams(model_name="meta-llama/Llama-2-7b-hf"),
    method="awq_q8_0",
    output_path="llama2-hq.gguf",
    
    # High-quality AWQ settings
    awq_group_size=64,
    calibration_samples=1024,
    
    verbose=True
)
```

### Performance-Optimized Configuration

```python
config = QuantizationConfig(
    model=ModelParams(
        model_name="meta-llama/Llama-2-7b-hf",
        model_kwargs={"torch_dtype": "auto", "device_map": "auto"}
    ),
    method="awq_q4_0",
    output_path="llama2-fast.gguf",
    
    # Performance settings
    batch_size=16,
    calibration_samples=256,
    awq_group_size=128,
    cleanup_temp=True
)
```

### BitsAndBytes Configuration

```python
config = QuantizationConfig(
    model=ModelParams(model_name="microsoft/DialoGPT-medium"),
    method="bnb_4bit",
    output_path="dialogpt-bnb.pytorch",
    output_format="pytorch"
)
```

## YAML Configuration

You can also define configurations in YAML format:

```yaml
# config.yaml
model:
  model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  model_kwargs:
    torch_dtype: "auto"

method: "awq_q4_0"
output_path: "model.gguf"
output_format: "gguf"

# AWQ settings
awq_group_size: 128
calibration_samples: 512
cleanup_temp: true

# General settings
batch_size: 16
verbose: true
```

Load with:
```python
config = QuantizationConfig.from_yaml("config.yaml")
```

## Validation

The configuration is automatically validated:

```python
config = QuantizationConfig(
    model=ModelParams(model_name="invalid-model"),
    method="invalid_method"
)
config.finalize_and_validate()  # Raises ValueError for invalid settings
```

## Return Values

When used with the `quantize()` function, returns a dictionary:

```python
result = {
    "quantization_method": "AWQ → GGUF",
    "original_size": "2.2 GB",
    "quantized_size": "661.5 MB", 
    "quantized_size_bytes": 693723136,
    "compression_ratio": "3.32x",
    "output_path": "model.gguf"
}
```

## Best Practices

1. **Start with defaults:** Use default parameters for initial testing
2. **Adjust for quality:** Decrease `awq_group_size` and increase `calibration_samples` for better quality
3. **Optimize for speed:** Increase `awq_group_size` and decrease `calibration_samples` for faster quantization
4. **Monitor memory:** Adjust `batch_size` based on available GPU memory
5. **Use appropriate methods:** AWQ for Llama models, BitsAndBytes for unsupported architectures