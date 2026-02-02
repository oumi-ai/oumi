# DeepSeek

## Summary

Configs for DeepSeek AI's native MoE models. For distilled models, see [deepseek_r1](../deepseek_r1/README.md).

Models in this directory include:

- [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) - 671B total, 37B active (1 shared + 8 of 256 routed experts)

## Model Details

### DeepSeek-R1-0528

- **Total Parameters**: 671B
- **Active Parameters**: 37B per token
- **Architecture**: MoE with 1 shared expert + 8 of 256 routed experts per layer
- **Context Length**: 128K tokens
- **Use Case**: Advanced reasoning tasks

## Quickstart

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. (Optional) If you wish to kick off jobs on a remote cluster, follow our [job launcher setup guide](https://oumi.ai/docs/en/latest/user_guides/launch/launch.html#setup).
3. Run your desired oumi command (examples below)!

## Example Commands

### Training

To launch DeepSeek-R1-0528 LoRA training with multiple GPUs:

```shell
oumi distributed torchrun -m oumi train -c oumi://configs/recipes/deepseek/sft/r1_0528_lora/train.yaml
```

**Note**: DeepSeek-R1-0528 requires significant GPU memory. Recommended setup: 8+ H100 80GB GPUs.

## Hardware Requirements

| Model | Minimum GPUs | Recommended GPUs |
|-------|--------------|------------------|
| DeepSeek-R1-0528 | 8x H100 80GB | 16x H100 80GB |
