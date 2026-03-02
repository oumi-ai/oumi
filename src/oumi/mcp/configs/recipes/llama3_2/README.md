# Llama 3.2

Configs for Meta's Llama 3.2 model family. This includes the 1B and 3B text-to-text models.

## 1B

### Model Info

| Attribute | Value |
|--|--|
| Vocab size | 128,256 |
| Hidden size | 2048 |
| MLP intermediate size | 8192 |
| Num layers | 16 |
| Num attention heads | 32 |
| Num KV heads | 8 |
| Weight tying | True |
| Model max length | 131,072 (initially trained with 8192) |

### Launch Command

Example command for 1B full fine-tuning on your local machine:
```shell
oumi train -c configs/recipes/llama3_2/sft/1b_full/train.yaml
```

## 3B

### Model Info

| Attribute | Value |
|--|--|
| Vocab size | 128,256 |
| Hidden size | 3072 |
| MLP intermediate size | 8192 |
| Num layers | 28 |
| Num attention heads | 24 |
| Num KV heads | 8 |
| Weight tying | True |
| Model max length | 131,072 (initially trained with 8192) |

### Launch Command

Example command for 3B full fine-tuning on GCP:
```shell
oumi launch up -c configs/recipes/llama3_2/sft/3b_full/gcp_job.yaml --cluster llama3-2
```
