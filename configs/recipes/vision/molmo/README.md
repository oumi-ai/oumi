# Molmo

Configs for Allen Institute for AI's Molmo 7B models. Molmo is a family of open vision-language models trained on PixMo, a dataset of 1 million highly-curated image-text pairs.

## Supported Models

### Molmo 7B-D
- **Hugging Face**: https://huggingface.co/allenai/Molmo-7B-D-0924
- **Base Model**: Qwen2-7B
- **Model Size**: ~8.02B parameters

### Molmo 7B-O
- **Hugging Face**: https://huggingface.co/allenai/Molmo-7B-O-0924
- **Base Model**: OLMo-7B-1024 (preview of next generation OLMo models)
- **Model Size**: ~7.67B parameters

## Model Info

| Attribute | Molmo 7B-D | Molmo 7B-O |
|--|--|--|
| Base Model | Qwen2-7B | OLMo-7B-1024 |
| Vision Backbone | OpenAI CLIP | OpenAI CLIP |
| Model Size | ~8.02B parameters | ~7.67B parameters |
| Model Type | Image-Text-to-Text | Image-Text-to-Text |
| Context Length | 2048 (configurable) | 2048 (configurable) |
| License | Apache 2.0 | Apache 2.0 |
| Trust Remote Code | Required | Required |



## Launch Command

Example command for full fine-tuning:
```shell
oumi train -c configs/recipes/vision/molmo/sft/molmo_o_full/train.yaml

oumi train -c configs/recipes/vision/molmo/sft/molmo_d_full/train.yaml
```

## Notes

- Both models require `trust_remote_code=True` due to custom modeling code
- Available variants: `oumi-ai/Molmo-7B-D-0924` and `oumi-ai/Molmo-7B-O-0924` for compatibility with latest transformers
- Gradient checkpointing is not supported by these models
