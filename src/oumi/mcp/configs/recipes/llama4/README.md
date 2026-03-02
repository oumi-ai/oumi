# Llama 4

Configs for Meta's Llama 4 model family. This includes the Scout and Maverick models with 17B parameters each.

## Model Variants

| Model | Variant | Parameters | Experts | Context Length | Training Tokens | Knowledge Cutoff |
|--|--|--|--|--|--|--|
| Llama 4 Scout | Base | 17B (Activated) 109B (Total) | 16 | 10M | ~40T | August 2024 |
| Llama 4 Scout | Instruct | 17B (Activated) 109B (Total) | 16 | 10M | ~40T | August 2024 |
| Llama 4 Maverick | Base | 17B (Activated) 400B (Total) | 128 | 1M | ~22T | August 2024 |
| Llama 4 Maverick | Instruct | 17B (Activated) 400B (Total) | 128 | 1M | ~22T | August 2024 |

## Model Architecture

| Attribute | Value |
|--|--|
| Vocab size | 128,256 |
| Hidden size | 8192 |
| MLP intermediate size | 28,672 |
| Num layers | 80 |
| Num attention heads | 64 |
| Num KV heads | 8 |
| Weight tying | False |

## Launch Commands

### Scout Base Model
```shell
# Full fine-tuning
oumi launch up -c configs/recipes/llama4/sft/scout_base_full/gcp_job.yaml --cluster llama4

# LoRA fine-tuning
oumi launch up -c configs/recipes/llama4/sft/scout_base_lora/gcp_job.yaml --cluster llama4

# Quantized LoRA fine-tuning
oumi launch up -c configs/recipes/llama4/sft/scout_base_qlora/gcp_job.yaml --cluster llama4
```

### Scout Instruct Model
```shell
# Full fine-tuning
oumi launch up -c configs/recipes/llama4/sft/scout_instruct_full/gcp_job.yaml --cluster llama4

# LoRA fine-tuning
oumi launch up -c configs/recipes/llama4/sft/scout_instruct_lora/gcp_job.yaml --cluster llama4

# Quantized LoRA fine-tuning
oumi launch up -c configs/recipes/llama4/sft/scout_instruct_qlora/gcp_job.yaml --cluster llama4
```

### Maverick Base Model
```shell
# Full fine-tuning
oumi launch up -c configs/recipes/llama4/sft/maverick_base_full/gcp_job.yaml --cluster llama4

# LoRA fine-tuning
oumi launch up -c configs/recipes/llama4/sft/maverick_base_lora/gcp_job.yaml --cluster llama4

# Quantized LoRA fine-tuning
oumi launch up -c configs/recipes/llama4/sft/maverick_base_qlora/gcp_job.yaml --cluster llama4
```

### Maverick Instruct Model
```shell
# Full fine-tuning
oumi launch up -c configs/recipes/llama4/sft/maverick_instruct_full/gcp_job.yaml --cluster llama4

# LoRA fine-tuning
oumi launch up -c configs/recipes/llama4/sft/maverick_instruct_lora/gcp_job.yaml --cluster llama4

# Quantized LoRA fine-tuning
oumi launch up -c configs/recipes/llama4/sft/maverick_instruct_qlora/gcp_job.yaml --cluster llama4
```

## Key Features

- Native multimodal support for text and image understanding
- Mixture-of-experts (MoE) architecture
- Support for 12 languages: Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese
- Early fusion for native multimodality
- Optimized for visual recognition, image reasoning, captioning, and answering general questions about images
- Supports up to 5 input images
- Available in BF16 weights with on-the-fly int4 quantization support
- Maverick model also available in FP8 quantized weights

## System Requirements

- Transformers v4.51.0 or later
- PyTorch with CUDA support
- H100 GPU recommended for optimal performance
