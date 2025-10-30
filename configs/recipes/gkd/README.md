# Generalized Knowledge Distillation (GKD) Training

This directory contains configuration recipes for training models using Generalized Knowledge Distillation (GKD), an on-policy distillation method implemented in TRL.

## Overview

**GKD** implements on-policy distillation where:
1. The **student model** generates outputs for given prompts
2. The **teacher model** provides corrections/guidance
3. The student learns from its own mistakes in real-time

This differs from traditional offline distillation where:
- Teacher outputs are pre-generated
- Student trains on a static dataset of teacher completions

## Key Features

- **On-Policy Learning**: Student generates outputs during training, learning from self-generated mistakes
- **Real-Time Feedback**: Teacher provides immediate corrections
- **Flexible Loss**: Uses Generalized Jensen-Shannon Divergence (JSD) with tunable interpolation
- **Data Mixing**: Combines on-policy (student-generated) and off-policy (dataset) examples

## When to Use GKD

**Use GKD when:**
- You want the student to learn from its own distribution
- You have sufficient GPU memory for both models
- You want more dynamic distillation than static teacher outputs
- You're training on tasks where on-policy learning is important

**Use offline distillation when:**
- GPU memory is limited (can't fit both models)
- Teacher is extremely large or slow
- Dataset is small and static generation is acceptable
- You want simpler, more predictable training

## Requirements

### Memory Requirements
- Both teacher and student models must fit in GPU memory simultaneously
- For large models, consider:
  - Using quantization for the teacher (`torch_dtype: "int8"`)
  - Smaller teacher models
  - Gradient checkpointing
  - PEFT/LoRA for the student

### Dependencies
- TRL >= 0.24.0 (already included in Oumi)
- Flash Attention 2 (optional, recommended for efficiency)

## Configuration Parameters

### Core GKD Parameters (in `training.gkd`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `teacher_model_name_or_path` | str | **Required** | HuggingFace model ID or path for teacher |
| `teacher_model_init_kwargs` | dict | `{}` | Kwargs for teacher model loading (e.g., `torch_dtype`, `attn_implementation`) |
| `temperature` | float | 0.9 | Sampling temperature for generation (0.0, 1.0] |
| `max_new_tokens` | int | 128 | Maximum tokens to generate per prompt |
| `lmbda` | float | 0.5 | Student data fraction [0.0, 1.0]: 0.5 = 50% on-policy, 50% off-policy |
| `beta` | float | 0.5 | JSD interpolation [0.0, 1.0]: 0.0=KL, 0.5=symmetric, 1.0=reverse KL |
| `disable_dropout` | bool | True | Disable dropout in student (recommended for stability) |
| `seq_kd` | bool | False | Use sequence-level KD (token-level recommended) |

### Hyperparameter Tuning Guide

**Temperature** (`temperature`):
- Lower (0.1-0.5): More focused generations, student learns conservative behavior
- Higher (0.7-1.0): More diverse generations, student explores more

**Lambda** (`lmbda`):
- 0.0: Pure off-policy (only dataset examples)
- 0.5: Balanced (50% on-policy, 50% off-policy) - **recommended default**
- 1.0: Pure on-policy (only student-generated)

**Beta** (`beta`):
- 0.0: KL(teacher || student) - teacher-focused
- 0.5: Symmetric JSD - **recommended default**
- 1.0: KL(student || teacher) - student-focused

## Example Configurations

### Example 1: Basic GKD (Qwen 0.5B ← 1.5B)
```yaml
model:
  model_name: "Qwen/Qwen2.5-0.5B-Instruct"
  torch_dtype_str: "bfloat16"

training:
  trainer_type: "TRL_GKD"
  use_peft: True

  gkd:
    teacher_model_name_or_path: "Qwen/Qwen2.5-1.5B-Instruct"
    teacher_model_init_kwargs:
      torch_dtype: "bfloat16"
    temperature: 0.9
    lmbda: 0.5
    beta: 0.5
    max_new_tokens: 256

peft:
  lora_r: 16
  lora_alpha: 32
```

### Example 2: Memory-Efficient (Quantized Teacher)
```yaml
training:
  gkd:
    teacher_model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
    teacher_model_init_kwargs:
      torch_dtype: "int8"  # Quantize teacher to save memory
      device_map: "auto"
```

### Example 3: Conservative Distillation
```yaml
training:
  gkd:
    teacher_model_name_or_path: "teacher-model"
    temperature: 0.3  # Lower temperature for focused learning
    lmbda: 0.3        # More off-policy (70% dataset)
    beta: 0.0         # KL divergence (teacher-focused)
```

## Comparison: GKD vs Offline Distillation

| Aspect | GKD (On-Policy) | Offline Distillation |
|--------|-----------------|----------------------|
| **Memory** | High (both models) | Low (one model at a time) |
| **Speed** | Slower (generation + training) | Faster (training only) |
| **Distribution** | Student's distribution | Teacher's distribution |
| **Flexibility** | Dynamic, adaptive | Static, pre-computed |
| **Best For** | Interactive tasks, safety | Resource-constrained, batch processing |

## Common Issues and Solutions

### 1. Out of Memory (OOM)
**Solutions:**
- Reduce batch size (`per_device_train_batch_size`)
- Use gradient accumulation
- Quantize teacher model (`torch_dtype: "int8"` or `"int4"`)
- Use smaller teacher model
- Enable gradient checkpointing
- Use LoRA for student

### 2. Slow Training
**Solutions:**
- Reduce `max_new_tokens`
- Increase batch size (if memory allows)
- Use Flash Attention 2
- Optimize teacher inference (compiled model, vLLM)

### 3. Poor Distillation Quality
**Solutions:**
- Adjust `lmbda` (try more on-policy: 0.6-0.8)
- Tune `temperature` (try 0.7-0.9)
- Increase training epochs
- Check teacher-student model compatibility

### 4. Unstable Training
**Solutions:**
- Lower learning rate
- Increase warmup ratio
- Ensure `disable_dropout: True`
- Try different `beta` values

## Best Practices

1. **Start Small**: Test with small models first (e.g., Qwen 0.5B ← 1.5B)
2. **Monitor Memory**: Track GPU memory usage, especially with large teachers
3. **Use PEFT**: Nearly always use LoRA/PEFT for the student
4. **Teacher Selection**: Teacher should be 2-4x larger than student for best results
5. **Dataset**: Use prompt-only datasets (similar to SFT format)
6. **Validation**: Monitor both loss and generation quality

## References

- **Paper**: [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)
- **TRL Docs**: [GKD Trainer Documentation](https://huggingface.co/docs/trl/main/en/gkd_trainer)
- **Demo Space**: [HuggingFace GKD Demo](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)

## Warnings

⚠️ **Experimental Feature**: GKDTrainer is marked as experimental in TRL and may be subject to changes or removal in future versions. Monitor the [TRL GitHub](https://github.com/huggingface/trl) for updates.

## Support

For issues or questions:
- Oumi Issues: https://github.com/oumi-ai/oumi/issues
- TRL Issues: https://github.com/huggingface/trl/issues
- Oumi Docs: https://oumi.ai/docs
