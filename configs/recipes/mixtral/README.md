# Mixtral MoE Training Configs

Training configurations for Mistral's Mixtral Mixture-of-Experts models.

## Supported Models

| Model | Total Params | Active Params | Experts | Architecture |
|-------|-------------|---------------|---------|--------------|
| Mixtral-8x7B | 47B | ~12B | 8 experts, top-2 | Sparse MoE |
| Mixtral-8x22B | 141B | ~39B | 8 experts, top-2 | Sparse MoE |

## Available Configs

### Mixtral-8x7B

| Config | Training Type | Backend | Est. VRAM/GPU |
|--------|---------------|---------|---------------|
| `sft/8x7b_lora/train.yaml` | LoRA | FSDP | ~30GB |
| `sft/8x7b_qlora/train.yaml` | QLoRA (4-bit) | FSDP | ~20GB |
| `sft/8x7b_lora_deepspeed/train.yaml` | LoRA | DeepSpeed ZeRO-3 | ~25GB |

### Mixtral-8x22B

| Config | Training Type | Backend | Est. VRAM/GPU |
|--------|---------------|---------|---------------|
| `sft/8x22b_lora/train.yaml` | LoRA | FSDP + CPU offload | ~55GB |
| `sft/8x22b_qlora/train.yaml` | QLoRA (4-bit) | FSDP | ~35GB |
| `sft/8x22b_lora_deepspeed/train.yaml` | LoRA | DeepSpeed ZeRO-3 + offload | ~40GB |

## Usage

```bash
# Mixtral-8x7B with LoRA (FSDP)
oumi distributed torchrun -m oumi train -c configs/recipes/mixtral/sft/8x7b_lora/train.yaml

# Mixtral-8x7B with QLoRA
oumi distributed torchrun -m oumi train -c configs/recipes/mixtral/sft/8x7b_qlora/train.yaml

# Mixtral-8x7B with DeepSpeed
oumi train -c configs/recipes/mixtral/sft/8x7b_lora_deepspeed/train.yaml

# Mixtral-8x22B with QLoRA
oumi distributed torchrun -m oumi train -c configs/recipes/mixtral/sft/8x22b_qlora/train.yaml
```

## MoE Best Practices

These configs follow MoE fine-tuning best practices:

1. **Attention-only LoRA**: Only `q_proj`, `k_proj`, `v_proj`, `o_proj` are targeted
   - DO NOT target MLP/expert layers (`w1`, `w2`, `w3`, `gate`)
   - Sparse MoE layers don't work well with PEFT

2. **4-bit Quantization**: Use QLoRA with NF4 quantization
   - Avoid 8-bit quantization (known issues with MoE)

3. **FSDP Wrapping**: Uses `MixtralDecoderLayer` for transformer layer wrapping

## Hardware Requirements

- **Mixtral-8x7B**: 8x A100-40GB or better, or single A100-80GB for QLoRA
- **Mixtral-8x22B**: 8x H100-80GB recommended

## References

- [HuggingFace Mixtral Guide](https://huggingface.co/blog/mixtral)
- [Mixtral Model Card](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
