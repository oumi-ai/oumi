# Megatron-LM Integration for Oumi

This document provides a comprehensive guide to the SkyRL+Megatron integration in Oumi, enabling training of very large language models (70B+) with advanced model parallelism strategies.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Migration Guide](#migration-guide)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [References](#references)

## Overview

The Megatron integration brings enterprise-grade model parallelism to Oumi through three key components:

1. **Megatron-Bridge**: Seamless HuggingFace ↔ Megatron checkpoint conversion
2. **Megatron-Core**: High-performance distributed training backend
3. **SkyRL Components**: Modular RL training infrastructure adapted for Oumi

### Why Megatron?

For models 70B+ parameters, standard data parallelism (FSDP/DDP) becomes insufficient:

- **Memory constraints**: Single model doesn't fit on one GPU
- **Communication overhead**: All-reduce across all GPUs is slow
- **Limited scalability**: Cannot efficiently use 100+ GPUs

Megatron solves this with **Model Parallelism**:

| Parallelism Type | What It Does | When to Use |
|------------------|--------------|-------------|
| **Tensor Parallel (TP)** | Splits layers horizontally | Always for 70B+ models |
| **Pipeline Parallel (PP)** | Splits layers vertically | 100B+ models or limited GPU memory |
| **Context Parallel (CP)** | Splits sequence length | Long sequences (8K+ tokens) |
| **Expert Parallel (EP)** | Splits MoE experts | MoE models (Mixtral, DeepSeek-MoE) |

### Key Features

✅ **Full Parallelism Support**: TP, PP, CP, EP
✅ **HF Compatibility**: Seamless conversion to/from HuggingFace format
✅ **GRPO Algorithm**: Group Relative Policy Optimization for RL
✅ **Sequence Packing**: Efficient batch processing for variable-length sequences
✅ **vLLM Integration**: Fast inference for rollout generation
✅ **Coexistence**: Works alongside existing veRL trainer

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Oumi Training Config                     │
│  - TrainingConfig                                           │
│  - MegatronParams (NEW)                                     │
│  - GrpoParams                                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              OumiMegatronGrpoTrainer (NEW)                  │
│  - Implements BaseTrainer interface                         │
│  - Handles GRPO training loop                               │
│  - Manages checkpointing & export                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Megatron-Bridge                            │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ HF → Megatron    │  │ Megatron → HF    │                │
│  │ Conversion       │  │ Export           │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Megatron-Core                             │
│  - Tensor Parallelism (split layers)                        │
│  - Pipeline Parallelism (split stages)                      │
│  - Context Parallelism (split sequences)                    │
│  - Expert Parallelism (split MoE)                           │
│  - Distributed Optimizer                                    │
│  - Activation Checkpointing                                 │
└─────────────────────────────────────────────────────────────┘
```

### Integration Flow

1. **One-Time Conversion**: HF checkpoint → Megatron format (uses Megatron-Bridge)
2. **Training**: GRPO with Megatron parallelism (uses Megatron-Core)
3. **Export**: Megatron → HF format for inference (uses vLLM)

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+
- PyTorch 2.5+
- Multi-GPU setup (16+ GPUs recommended for 70B models)

### Install Megatron Dependencies

```bash
# Option 1: Install from PyPI (recommended)
pip install oumi[megatron]

# Option 2: Install from source (if packages unavailable)
pip install git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
pip install megatron-core>=0.9.0
pip install megatron-lm>=0.9.0
pip install vllm>=0.10.0
pip install flash-attn>=2.0.0 --no-build-isolation
```

### Verify Installation

```python
from oumi.core.trainers.megatron.bridge_utils import check_megatron_bridge_available

check_megatron_bridge_available()  # Should not raise an error
print("Megatron-Bridge is available!")
```

## Quick Start

### 1. Prepare Your Configuration

Create a config file `train_70b_grpo.yaml`:

```yaml
model:
  model_name: "meta-llama/Llama-3.1-70B-Instruct"
  torch_dtype_str: "bfloat16"
  load_pretrained_weights: True

data:
  train:
    datasets:
      - dataset_name: "gsm8k"
        split: "train"

training:
  trainer_type: "MEGATRON_GRPO"
  max_steps: 1000
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-7
  output_dir: "output/llama70b_megatron"

megatron:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 2
  enable_sequence_packing: True
  inference_backend: "vllm"

grpo:
  num_generations: 4
  temperature: 0.9
  epsilon: 0.2
```

### 2. Launch Training

```bash
# Single node (16 GPUs)
oumi distributed torchrun -c train_70b_grpo.yaml

# Multi-node (2 nodes x 8 GPUs = 16 GPUs)
# On node 0:
oumi distributed torchrun --nnodes=2 --node_rank=0 --master_addr=<node0_ip> \
  -c train_70b_grpo.yaml
# On node 1:
oumi distributed torchrun --nnodes=2 --node_rank=1 --master_addr=<node0_ip> \
  -c train_70b_grpo.yaml
```

### 3. Export to HuggingFace Format

After training completes, the model is automatically exported to HF format:

```
output/llama70b_megatron/
├── checkpoints/              # Megatron checkpoints (for resuming)
├── hf_model/                 # HuggingFace format (for inference)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
└── megatron_checkpoint/      # Initial converted checkpoint
```

Use the exported model with vLLM:

```python
from vllm import LLM

llm = LLM(model="output/llama70b_megatron/hf_model")
outputs = llm.generate(["What is 2+2?"])
```

## Configuration

### MegatronParams

Full configuration reference for `megatron` section:

```yaml
megatron:
  # Core parallelism dimensions
  tensor_model_parallel_size: 8      # Split layers across 8 GPUs
  pipeline_model_parallel_size: 2    # Split stages across 2 GPUs
  context_parallel_size: 1           # No sequence splitting (seq < 8K)
  expert_model_parallel_size: 1      # No MoE (Llama doesn't use MoE)

  # Batch configuration
  micro_batch_size: 1                # Per-GPU batch size
  global_batch_size: 8               # Total effective batch size

  # Features
  enable_sequence_packing: True      # Pack sequences for efficiency

  # DDP configuration
  ddp_config:
    grad_reduce_in_fp32: True        # Better numerical stability
    overlap_grad_reduce: False       # Enable for performance (disable for debugging)
    overlap_param_gather: False
    average_in_collective: True

  # Optimizer configuration
  optimizer_config:
    optimizer_cpu_offload: False     # Offload optimizer to CPU (saves memory)
    optimizer_offload_fraction: 0.0  # Fraction to offload (0-1)

  # Transformer configuration (activation checkpointing)
  transformer_config:
    recompute_granularity: "selective"  # "full", "selective", or null
    recompute_method: "uniform"
    recompute_modules: ["core_attn"]    # Which modules to checkpoint

  # Checkpointing
  checkpoint_config:
    use_async_checkpoint: False      # Async for faster saves
    use_fully_parallel_strategy: True

  # Weight synchronization
  enable_weight_sync: True
  weight_sync_method: "nccl"         # "nccl", "gloo", or "checkpoint"

  # Inference backend
  inference_backend: "vllm"          # "vllm", "sglang", or "megatron"
```

### Parallelism Strategy Guide

#### Llama 70B (16 GPUs)

```yaml
megatron:
  tensor_model_parallel_size: 8    # 80 heads / 8 = 10 heads per GPU
  pipeline_model_parallel_size: 2  # 80 layers / 2 = 40 layers per stage
  context_parallel_size: 1
```

Total model parallel: 8 × 2 = 16 GPUs
Data parallel: 16 / 16 = 1 (no data parallelism)

#### Llama 405B (256 GPUs)

```yaml
megatron:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 16
  context_parallel_size: 2
```

Total model parallel: 8 × 16 × 2 = 256 GPUs
Data parallel: 256 / 256 = 1

**Rule of thumb**: `TP × PP × CP × EP × DP = Total GPUs`

## Migration Guide

### From veRL GRPO to Megatron GRPO

#### What Changes

| Aspect | veRL GRPO | Megatron GRPO |
|--------|-----------|---------------|
| **Trainer Type** | `VERL_GRPO` | `MEGATRON_GRPO` |
| **Parallelism** | FSDP (data parallel) | TP+PP+CP (model parallel) |
| **Backend** | Ray + FSDP | Megatron-Core |
| **Scale** | Up to ~70B | 70B to 405B+ |
| **Config** | No megatron section | Add `megatron:` section |

#### Migration Steps

1. **Change trainer type**:
   ```yaml
   training:
     trainer_type: "MEGATRON_GRPO"  # Was: VERL_GRPO
   ```

2. **Add Megatron configuration**:
   ```yaml
   megatron:
     tensor_model_parallel_size: 8
     pipeline_model_parallel_size: 2
     enable_sequence_packing: True
   ```

3. **Disable FSDP/DeepSpeed** (incompatible with Megatron):
   ```yaml
   fsdp:
     enable_fsdp: False  # Was: True

   deepspeed:
     enable_deepspeed: False
   ```

4. **Adjust batch sizes** (Megatron uses micro-batches):
   ```yaml
   training:
     per_device_train_batch_size: 1  # Micro-batch size
     gradient_accumulation_steps: 4   # Larger accumulation

   megatron:
     micro_batch_size: 1              # Matches per_device
     global_batch_size: 4             # micro * accumulation
   ```

5. **Update dependencies**:
   ```bash
   pip install oumi[megatron]
   ```

#### Example Comparison

**Before (veRL)**:
```yaml
training:
  trainer_type: "VERL_GRPO"
  per_device_train_batch_size: 4

fsdp:
  enable_fsdp: True
  cpu_offload: True
```

**After (Megatron)**:
```yaml
training:
  trainer_type: "MEGATRON_GRPO"
  per_device_train_batch_size: 1

megatron:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 2

fsdp:
  enable_fsdp: False
```

## Advanced Usage

### Custom Parallelism for Different Hardware

**64 GPUs**:
```yaml
megatron:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 4
  context_parallel_size: 2
# Total: 8 × 4 × 2 = 64
```

**128 GPUs**:
```yaml
megatron:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 8
  context_parallel_size: 2
# Total: 8 × 8 × 2 = 128
```

### Virtual Pipeline Parallelism

For better pipeline efficiency (reduces bubble overhead):

```yaml
megatron:
  pipeline_model_parallel_size: 16
  virtual_pipeline_model_parallel_size: 32  # 2x interleaving
```

### CPU Optimizer Offloading

For very large models where optimizer states don't fit in GPU memory:

```yaml
megatron:
  optimizer_config:
    optimizer_cpu_offload: True
    optimizer_offload_fraction: 1.0  # Offload 100%
    overlap_cpu_optimizer_d2h_h2d: True  # Overlap transfers
```

### Aggressive Activation Checkpointing

To save memory at the cost of compute:

```yaml
megatron:
  transformer_config:
    recompute_granularity: "full"        # Checkpoint everything
    recompute_method: "uniform"
    recompute_num_layers: 1              # Checkpoint every layer
    recompute_modules: ["core_attn", "mlp"]  # Both attention and MLP
```

## Troubleshooting

### Common Issues

#### 1. "Megatron-Bridge not available"

**Error**: `ImportError: megatron-bridge is not available`

**Solution**:
```bash
pip install megatron-bridge megatron-core megatron-lm
# Or install from source:
pip install git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
```

#### 2. "world_size must be divisible by model parallel size"

**Error**: `world_size (16) must be divisible by the product of all model parallel sizes (32)`

**Solution**: Adjust parallelism to match GPU count:
```yaml
# For 16 GPUs:
megatron:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 2  # 8 × 2 = 16 ✓
```

#### 3. Out of Memory (OOM)

**Solutions**:
1. Enable activation checkpointing:
   ```yaml
   training:
     enable_gradient_checkpointing: True
   ```

2. Reduce micro-batch size:
   ```yaml
   megatron:
     micro_batch_size: 1  # Reduce from 2
   ```

3. Enable optimizer CPU offloading:
   ```yaml
   megatron:
     optimizer_config:
       optimizer_cpu_offload: True
   ```

4. Increase pipeline parallelism:
   ```yaml
   megatron:
     pipeline_model_parallel_size: 4  # Increase from 2
   ```

#### 4. Slow Training

**Check**:
- Enable overlap:
  ```yaml
  megatron:
    ddp_config:
      overlap_grad_reduce: True
      overlap_param_gather: True
  ```

- Use async checkpointing:
  ```yaml
  megatron:
    checkpoint_config:
      use_async_checkpoint: True
  ```

- Profile with:
  ```yaml
  training:
    include_performance_metrics: True
  ```

## Performance Tuning

### Memory vs. Speed Trade-offs

| Technique | Memory Saved | Speed Impact | When to Use |
|-----------|--------------|--------------|-------------|
| Activation checkpointing | 50-70% | -20% | Always for 70B+ |
| CPU optimizer offload | 30-40% | -30% | When OOM |
| Increase PP size | Variable | -10% | When OOM |
| Sequence packing | N/A | +30% | RL with variable lengths |

### Recommended Settings

**70B Model (16 GPUs, A100-80GB)**:
```yaml
megatron:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 2
  enable_sequence_packing: True
  transformer_config:
    recompute_granularity: "selective"
  optimizer_config:
    optimizer_cpu_offload: False
```

**405B Model (256 GPUs, H100-80GB)**:
```yaml
megatron:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 16
  context_parallel_size: 2
  virtual_pipeline_model_parallel_size: 32
  enable_sequence_packing: True
  transformer_config:
    recompute_granularity: "full"
  optimizer_config:
    optimizer_cpu_offload: True
    optimizer_offload_fraction: 1.0
```

## References

### Papers

- **GRPO**: [DeepSeekMath Paper](https://arxiv.org/pdf/2402.03300) - Group Relative Policy Optimization
- **Megatron**: [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Model Parallel Training

### Documentation

- [Megatron-Bridge GitHub](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [NeMo-RL (Reference Implementation)](https://github.com/NVIDIA-NeMo/RL)
- [SkyRL GitHub](https://github.com/berkeley-sky-lab/SkyRL)

### Related Oumi Docs

- [Training Guide](https://oumi.ai/docs/en/latest/user_guides/train/train.html)
- [Configuration Reference](https://github.com/oumi-ai/oumi/blob/main/src/oumi/core/configs/training_config.py)

## Contributing

Found a bug or want to improve the Megatron integration? Please file an issue or submit a PR:

- [Oumi GitHub Issues](https://github.com/oumi-ai/oumi/issues)
- [Oumi Contributing Guide](https://github.com/oumi-ai/oumi/blob/main/CONTRIBUTING.md)

---

**Last Updated**: 2025-11-13
**Oumi Version**: 0.1.0+
**Megatron-Bridge Version**: 0.1.0+
