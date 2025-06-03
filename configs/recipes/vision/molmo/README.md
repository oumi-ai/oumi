# Molmo 7B-D

Configs for Allen Institute for AI's Molmo 7B-D model. Molmo is a family of open vision-language models trained on PixMo, a dataset of 1 million highly-curated image-text pairs. See https://huggingface.co/allenai/Molmo-7B-D-0924

## Model Info

| Attribute | Value |
|--|--|
| Base Model | Qwen2-7B |
| Vision Backbone | OpenAI CLIP |
| Model Size | ~8.02B parameters |
| Model Type | Image-Text-to-Text |
| Context Length | 2048 (configurable) |
| License | Apache 2.0 |
| Trust Remote Code | Required |

## Performance

Molmo 7B-D achieves state-of-the-art performance among multimodal models with similar size:
- **Average Score on 11 Academic Benchmarks**: 77.3
- **Human Preference Elo Rating**: 1056
- Performs between GPT-4V and GPT-4o on both academic benchmarks and human evaluation

## Features

- Fully open-source multimodal model
- Supports image understanding and text generation
- Based on Qwen2-7B architecture with OpenAI CLIP vision backbone
- Trained on high-quality PixMo dataset
- Compatible with standard transformers library (with trust_remote_code=True)

## Launch Command

Example command for full fine-tuning:
```shell
oumi train -c configs/recipes/vision/molmo/sft/full/train.yaml
```

## Notes

- The original model requires `trust_remote_code=True` due to custom modeling code
- Currently using `oumi-ai/Molmo-7B-D-0924` variant for compatibility with latest transformers
- Gradient checkpointing is not supported by this model
