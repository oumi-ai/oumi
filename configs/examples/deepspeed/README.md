# DeepSpeed Examples

This directory contains example configurations for using DeepSpeed distributed training with Oumi. DeepSpeed provides memory-efficient training through ZeRO (Zero Redundancy Optimizer) stages and optional CPU/NVMe offloading.

## ZeRO Stages Overview

- **ZeRO-0**: Disabled ZeRO optimization (standard data parallelism)
- **ZeRO-1**: Optimizer state sharding only
- **ZeRO-2**: Optimizer state + gradient sharding
- **ZeRO-3**: Full sharding (optimizer + gradients + model parameters)

## Example Configurations

### ZeRO-2 Example
- **File**: `llama3_1_8b_deepspeed_z2.yaml`
- **Use case**: Good balance between memory savings and performance
- **Memory optimization**: Shards optimizer states and gradients
- **Recommended for**: Models that mostly fit in GPU memory but need optimizer savings

### ZeRO-3 Example
- **File**: `llama3_1_8b_deepspeed_z3.yaml`
- **Use case**: Maximum memory efficiency without CPU offloading
- **Memory optimization**: Shards everything (parameters, gradients, optimizer states)
- **Recommended for**: Large models that don't fit in GPU memory

### ZeRO-3 with CPU Offloading
- **File**: `llama3_1_8b_deepspeed_z3_offload.yaml`
- **Use case**: Training very large models on limited GPU memory
- **Memory optimization**: ZeRO-3 + offloads optimizer states and parameters to CPU
- **Recommended for**: When GPU memory is severely constrained
- **Trade-off**: Slower training due to CPU-GPU data transfers

## Configuration Parameters

### Basic DeepSpeed Configuration

```yaml
deepspeed:
  enable_deepspeed: True
  zero_stage: "3"              # ZeRO stage (0, 1, 2, or 3)
  precision: "bf16"            # Mixed precision ("fp16" or "bf16")
```

### Memory Optimization

```yaml
deepspeed:
  # Communication optimization
  overlap_comm: True
  contiguous_gradients: True
  
  # ZeRO-3 specific parameters
  stage3_max_live_parameters: 1000000000
  stage3_gather_16bit_weights_on_model_save: True
```

### CPU Offloading

```yaml
deepspeed:
  # Offload optimizer states to CPU
  offload_optimizer:
    device: "cpu"
    pin_memory: True
    buffer_count: 4
  
  # Offload parameters to CPU (ZeRO-3 only)
  offload_param:
    device: "cpu"
    pin_memory: True
    buffer_count: 4
```

### Using External DeepSpeed Config Files

You can also use external DeepSpeed JSON configuration files:

```yaml
deepspeed:
  enable_deepspeed: True
  deepspeed_config_path: "path/to/deepspeed_config.json"
```

## Performance Tips

1. **ZeRO Stage Selection**:
   - Use ZeRO-2 for models that mostly fit in GPU memory
   - Use ZeRO-3 for large models that don't fit in GPU memory
   - Only use CPU offloading when absolutely necessary

2. **Batch Size Tuning**:
   - ZeRO-3 allows larger batch sizes due to memory savings
   - Reduce batch size when using CPU offloading
   - Adjust `gradient_accumulation_steps` to maintain effective batch size

3. **Mixed Precision**:
   - Use `bf16` on newer GPUs (Ampere+) for better numerical stability
   - Use `fp16` on older GPUs for maximum performance

4. **Communication Optimization**:
   - Keep `overlap_comm: True` for better performance
   - Use `contiguous_gradients: True` for efficient communication

## Compatibility Notes

- DeepSpeed and FSDP cannot be enabled simultaneously
- Some features may be incompatible with DeepSpeed ZeRO-3:
  - Certain quantization methods
  - Some PEFT techniques may have limitations
- DeepSpeed version is pinned to `>=0.10.0,<=0.16.9` for stability

## Running Examples

```bash
# ZeRO-2 training
oumi train -c configs/examples/deepspeed/llama3_1_8b_deepspeed_z2.yaml

# ZeRO-3 training
oumi train -c configs/examples/deepspeed/llama3_1_8b_deepspeed_z3.yaml

# ZeRO-3 with CPU offloading
oumi train -c configs/examples/deepspeed/llama3_1_8b_deepspeed_z3_offload.yaml
```

## Troubleshooting

- **OOM errors**: Try enabling CPU offloading or using smaller batch sizes
- **Slow training**: Reduce CPU offloading or use higher ZeRO stage without offloading
- **Version conflicts**: Ensure DeepSpeed version is within supported range

For more information, refer to the [DeepSpeed documentation](https://www.deepspeed.ai/).