# Llama 3.1

Configs for Meta's Llama 3.1 model family. This includes the 8B, 70B, and 405B model sizes.

## 8B

### Model Info

| Attribute | Value |
|--|--|
| Vocab size | 128,256 |
| Hidden size | 4096 |
| MLP intermediate size | 14,336 |
| Num layers | 32 |
| Num attention heads | 32 |
| Num KV heads | 8 |
| Weight tying | False |
| Model max length | 131,072 (initially trained with 8192) |

### Launch Command

Example command for 8B full fine-tuning on GCP:
```shell
oumi launch up -c configs/recipes/llama3_1/sft/8b_full/gcp_job.yaml --cluster llama3-1
```

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
oumi launch up -c configs/recipes/llama3_1/sft/70b_full/gcp_job.yaml --cluster llama3-1
```

## 405B

This model is **very** memory-intensive, so be prepared to increase the number of nodes or use memory-reducing parameters to avoid OOMs (see [our documentation page](https://oumi.ai/docs/en/latest/faq/oom.html) for some examples).
### Model Info

| Attribute | Value |
|--|--|
| Vocab size | 128,256 |
| Hidden size | 16,384 |
| MLP intermediate size | 53,248 |
| Num layers | 126 |
| Num attention heads | 128 |
| Num KV heads | 8 |
| Weight tying | False |
| Model max length | 131,072 (initially trained with 8192) |

### Launch Command

Example command for 405B QLora on Polaris:
```shell
oumi launch up -c configs/recipes/llama3_1/sft/405b_qlora/polaris_job.yaml --cluster preemptable.$ALCF_USER --user $ALCF_USER
```
