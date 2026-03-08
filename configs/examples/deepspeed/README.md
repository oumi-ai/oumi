# DeepSpeed Configuration Examples

This directory contains example configurations for training with DeepSpeed ZeRO optimization.

## Available Configurations

- `llama3_1_8b_deepspeed_z2_train.yaml` - ZeRO-2 configuration with optimizer state and gradient sharding
- `llama3_1_8b_deepspeed_z3_train.yaml` - ZeRO-3 configuration with full parameter, gradient, and optimizer state sharding
- `llama3_1_8b_deepspeed_z3_offload_train.yaml` - ZeRO-3 with CPU offloading for maximum memory efficiency

## Checkpoint Reconstruction

After training with DeepSpeed ZeRO-2 or ZeRO-3, model parameters are sharded across multiple files. To reconstruct a complete model checkpoint from these fragments, use DeepSpeed's `zero_to_fp32` utility:

```bash
# Reconstruct checkpoint from ZeRO fragments
wget https://raw.githubusercontent.com/microsoft/DeepSpeed/master/deepspeed/utils/zero_to_fp32.py
python zero_to_fp32.py /path/to/checkpoint/directory /path/to/output/pytorch_model.bin
```

The reconstructed `pytorch_model.bin` file can then be used for inference or further fine-tuning without DeepSpeed.

**Reference**: [DeepSpeed zero_to_fp32 Documentation](https://deepspeed.readthedocs.io/en/latest/_modules/deepspeed/utils/zero_to_fp32.html)
