# Llama 3.3

Configs for Meta's Llama 3.3 model family. This includes the 70B text-to-text models.

## 70B

### Model Info

| Attribute | Value |
|--|--|
| Vocab size | 128,256 |
| Hidden size | 8192 |
| MLP intermediate size | 28,672 |
| Num layers | 80 |
| Num attention heads | 64 |
| Num KV heads | 8 |
| Weight tying | False |
| Model max length | 131,072 (initially trained with 8192) |

### Launch Command

Example command for 70B full fine-tuning on GCP:
```shell
oumi launch up -c configs/recipes/llama3_3/sft/70b_full/gcp_job.yaml --cluster llama3-3
```

Example command for 70B LoRA fine-tuning on GCP:
```shell
oumi launch up -c configs/recipes/llama3_3/sft/70b_lora/gcp_job.yaml --cluster llama3-3
```

Example command for 70B quantized LoRA fine-tuning on GCP:
```shell
oumi launch up -c configs/recipes/llama3_3/sft/70b_qlora/gcp_job.yaml --cluster llama3-3
```
