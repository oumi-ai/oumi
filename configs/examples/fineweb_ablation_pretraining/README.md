# HuggingFace FineWeb Ablation Model

Configs for pre-training the FineWeb ablation model, a Llama 2B model. The primary focus
is improving MFU. See https://huggingface.co/HuggingFaceFW/ablation-model-fineweb-v1.

## Model Info

| Attribute | Value |
|--|--|
| Vocab size | 50,272 |
| Hidden size | 2048 |
| MLP intermediate size | 8192 |
| Num layers | 24 |
| Num attention heads | 32 |
| Num KV heads | 32 |
| Weight tying | True |

## Launch Command

Currently, the best training method is DDP:
```shell
oumi launch up -c configs/examples/fineweb_ablation_pretraining/ddp/gcp_job.yaml --cluster fineweb
```

For FSDP training:
```shell
oumi launch up -c configs/examples/fineweb_ablation_pretraining/fsdp/gcp_job.yaml --cluster fineweb-fsdp
```
