# DBRX MoE Training Configs

Training configurations for Databricks' DBRX Mixture-of-Experts model.

## Supported Models

| Model | Total Params | Active Params | Experts | Architecture |
|-------|-------------|---------------|---------|--------------|
| DBRX-Instruct | 132B | ~36B | 16 experts, top-4 | Fine-grained MoE |

## Available Configs

| Config | Training Type | Backend | Est. VRAM/GPU |
|--------|---------------|---------|---------------|
| `sft/instruct_lora/train.yaml` | LoRA | FSDP + CPU offload | ~55GB |
| `sft/instruct_qlora/train.yaml` | QLoRA (4-bit) | FSDP | ~35GB |
| `sft/instruct_lora_deepspeed/train.yaml` | LoRA | DeepSpeed ZeRO-3 + offload | ~40GB |

## Usage

```bash
# DBRX with LoRA (FSDP)
oumi distributed torchrun -m oumi train -c configs/recipes/dbrx/sft/instruct_lora/train.yaml

# DBRX with QLoRA
oumi distributed torchrun -m oumi train -c configs/recipes/dbrx/sft/instruct_qlora/train.yaml

# DBRX with DeepSpeed
oumi train -c configs/recipes/dbrx/sft/instruct_lora_deepspeed/train.yaml
```

## Known Issues

DBRX has some known limitations with LoRA/QLoRA:

1. **Fused Expert Parameters**: DBRX uses fused expert parameters that are incompatible with standard PEFT
   - Only attention layers (`Wqkv`, `out_proj`) can be reliably targeted
   - See: https://huggingface.co/databricks/dbrx-instruct/discussions/10

2. **4-bit Quantization**: May have issues due to fused experts
   - Consider using `SinclairSchneider/dbrx-instruct-quantization-fixed` if issues arise

## MoE Best Practices

These configs follow MoE fine-tuning best practices:

1. **Attention-only LoRA**: Only `Wqkv` and `out_proj` are targeted
   - DBRX uses fused QKV attention
   - Expert layers are NOT targeted due to fused implementation

2. **FSDP Wrapping**: Uses `DbrxBlock` for transformer layer wrapping

3. **Trust Remote Code**: Required for DBRX architecture

## Hardware Requirements

- **Minimum**: 8x H100-80GB (640GB total VRAM)
- **Recommended**: 8x H100-80GB with CPU offload enabled
- **System RAM**: 256GB+ recommended for CPU offload

## Architecture Details

DBRX uses a fine-grained MoE architecture:
- 16 experts per layer (vs 8 for Mixtral)
- Top-4 expert selection (vs top-2 for Mixtral)
- Rotary Position Encodings (RoPE)
- Grouped Query Attention (GQA)
- Gated Linear Units (GLU)

## References

- [DBRX Model Card](https://huggingface.co/databricks/dbrx-instruct)
- [DBRX Technical Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
