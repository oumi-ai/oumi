# Ulysses Sequence Parallelism Test Configurations

This directory contains test configurations for the newly integrated Ulysses Sequence Parallelism (SP) feature in Oumi.

## Overview

Ulysses Sequence Parallelism enables training on extremely long sequences (500K+ tokens) by sharding sequences across multiple GPUs. This is particularly useful for:

- Long document processing
- Extended context training
- Memory-constrained scenarios with long sequences

## Key Features

- **Sequence Sharding**: Divides sequences across multiple GPUs instead of model parameters
- **TRL Integration**: Seamlessly works with TRL's SFTTrainer
- **DeepSpeed Compatible**: Integrates with existing DeepSpeed configurations
- **Automatic Attention Patching**: Handles sequence parallelism at the attention level

## Configuration Options

### Training Parameters

```yaml
training:
  trainer_type: TRL_SFT_ULYSSES  # Use Ulysses-enabled SFT trainer
  enable_ulysses_sequence_parallel: True
  ulysses_sequence_parallel_size: 2  # Number of GPUs for sequence sharding
```

### DeepSpeed Integration

```yaml
deepspeed:
  enable_deepspeed: True
  zero_stage: "2"  # Works with ZeRO stages 1-3
  ulysses_sequence_parallel_size: 2  # Must match training parameter
```

## Requirements

1. **Multi-GPU Setup**: Requires at least 2 GPUs for sequence parallelism
2. **Distributed Training**: Must be run with distributed training enabled
3. **DeepSpeed**: Recommended for optimal performance

## Usage Example

```bash
# Run with torchrun for distributed training
torchrun --nproc_per_node=2 -m oumi.train -c configs/recipes/ulysses_sp_test/sft_ulysses_test.yaml
```

## Implementation Details

The Ulysses SP integration includes:

- **UlyssesSPAttentionHF**: Patches attention layers for sequence parallelism
- **UlyssesSPDataLoaderAdapter**: Handles sequence sharding in data loading
- **UlyssesSFTTrainer**: Extended SFTTrainer with sequence parallelism support

## Performance Considerations

- **Memory Usage**: Enables training longer sequences with the same memory footprint
- **Communication Overhead**: Adds all-gather/scatter operations during attention
- **Optimal Shard Size**: Sequence length should be divisible by `ulysses_sequence_parallel_size`

## Troubleshooting

1. **World Size Mismatch**: Ensure `ulysses_sequence_parallel_size <= world_size`
2. **Sequence Length**: Verify sequences can be evenly divided across GPUs
3. **Distributed Init**: Ensure distributed training is properly initialized

## References

- [DeepSpeed Ulysses Tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)
- [Snowflake Arctic Training PR](https://github.com/snowflakedb/ArcticTraining/pull/45)
- [Oumi Training Documentation](https://oumi.ai/docs/en/latest/user_guides/train/train.html)