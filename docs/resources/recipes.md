# Recipes

To help you get started with Oumi, we've prepared a set of recipes for common use cases. These recipes are designed to be easy to understand and modify, and should be a good starting point for your own projects. Each recipe is a YAML file that can be used to train, evaluate, or deploy a model.

## Overview

The recipes are organized by model family and task type. Each recipe includes:

- Configuration files for different tasks (training, evaluation, inference)
- Platform-specific job configurations (Cloud (e.g. GCP), Polaris, or local)
- Multiple training methods (FFT, LoRA, QLoRA, FSDP/DDP)

To use a recipe, simply download the desired configuration file, modify any parameters as needed, and run the configuration using the Oumi CLI. For example:

```bash
oumi train --config path/to/config.yaml
oumi evaluate --config path/to/config.yaml
oumi infer --config path/to/config.yaml
```

You can also checkout the README.md in each recipe's directory for more details and examples.


## Common Models

### 🦙 Llama Family

| Model | Configuration | Links |
|-------|--------------|-------|
| Llama 3.1 8B | `recipes/llama3_1/sft/8b_lora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_lora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_lora/train.yaml` |
| | `recipes/llama3_1/sft/8b_full/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_full/train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_full/train.yaml` |
| | `recipes/llama3_1/sft/8b_qlora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_qlora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_qlora/train.yaml` |
| | `recipes/llama3_1/pretraining/8b/train.yaml` | {download}`Download </../configs/recipes/llama3_1/pretraining/8b/train.yaml>` {gh}`configs/recipes/llama3_1/pretraining/8b/train.yaml` |
| | `recipes/llama3_1/inference/8b_infer.yaml` | {download}`Download </../configs/recipes/llama3_1/inference/8b_infer.yaml>` {gh}`configs/recipes/llama3_1/inference/8b_infer.yaml` |
| | `recipes/llama3_1/evaluation/8b_eval.yaml` | {download}`Download </../configs/recipes/llama3_1/evaluation/8b_eval.yaml>` {gh}`configs/recipes/llama3_1/evaluation/8b_eval.yaml` |
| Llama 3.1 70B | `recipes/llama3_1/sft/70b_lora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/70b_lora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/70b_lora/train.yaml` |
| | `recipes/llama3_1/sft/70b_full/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/70b_full/train.yaml>` {gh}`configs/recipes/llama3_1/sft/70b_full/train.yaml` |
| | `recipes/llama3_1/sft/70b_qlora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/70b_qlora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/70b_qlora/train.yaml` |
| | `recipes/llama3_1/inference/70b_infer.yaml` | {download}`Download </../configs/recipes/llama3_1/inference/70b_infer.yaml>` {gh}`configs/recipes/llama3_1/inference/70b_infer.yaml` |
| | `recipes/llama3_1/evaluation/70b_eval.yaml` | {download}`Download </../configs/recipes/llama3_1/evaluation/70b_eval.yaml>` {gh}`configs/recipes/llama3_1/evaluation/70b_eval.yaml` |
| Llama 3.1 405B | `recipes/llama3_1/sft/405b_lora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/405b_lora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/405b_lora/train.yaml` |
| | `recipes/llama3_1/sft/405b_qlora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/405b_qlora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/405b_qlora/train.yaml` |
| | `recipes/llama3_1/sft/405b_full/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/405b_full/train.yaml>` {gh}`configs/recipes/llama3_1/sft/405b_full/train.yaml` |
| Llama 3.2 3B | `recipes/llama3_2/sft/3b_full/train.yaml` | {download}`Download </../configs/recipes/llama3_2/sft/3b_full/train.yaml>` {gh}`configs/recipes/llama3_2/sft/3b_full/train.yaml` |
| | `recipes/llama3_2/sft/3b_lora/train.yaml` | {download}`Download </../configs/recipes/llama3_2/sft/3b_lora/train.yaml>` {gh}`configs/recipes/llama3_2/sft/3b_lora/train.yaml` |
| | `recipes/llama3_2/sft/3b_qlora/train.yaml` | {download}`Download </../configs/recipes/llama3_2/sft/3b_qlora/train.yaml>` {gh}`configs/recipes/llama3_2/sft/3b_qlora/train.yaml` |
| | `recipes/llama3_2/evaluation/3b_eval.yaml` | {download}`Download </../configs/recipes/llama3_2/evaluation/3b_eval.yaml>` {gh}`configs/recipes/llama3_2/evaluation/3b_eval.yaml` |
| | `recipes/llama3_2/inference/3b_infer.yaml` | {download}`Download </../configs/recipes/llama3_2/inference/3b_infer.yaml>` {gh}`configs/recipes/llama3_2/inference/3b_infer.yaml` |

### 🎨 Vision Models

| Model | Configuration | Links |
|-------|--------------|-------|
| Llama 3.2 Vision 11B | `recipes/vision/llama3_2_vision/sft/11b_train.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/sft/11b_train.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/sft/11b_train.yaml` |
| | `recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml` |
| | `recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml` |
| | `recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml` |
| LLaVA 7B | `recipes/vision/llava_7b/sft/train.yaml` | {download}`Download </../configs/recipes/vision/llava_7b/sft/train.yaml>` {gh}`configs/recipes/vision/llava_7b/sft/train.yaml` |
| | `recipes/vision/llava_7b/inference/infer.yaml` | {download}`Download </../configs/recipes/vision/llava_7b/inference/infer.yaml>` {gh}`configs/recipes/vision/llava_7b/inference/infer.yaml` |
| Phi3 Vision | `recipes/vision/phi3/sft/train.yaml` | {download}`Download </../configs/recipes/vision/phi3/sft/train.yaml>` {gh}`configs/recipes/vision/phi3/sft/train.yaml` |
| Qwen2-VL 2B | `recipes/vision/qwen2_vl_2b/sft/train.yaml` | {download}`Download </../configs/recipes/vision/qwen2_vl_2b/sft/train.yaml>` {gh}`configs/recipes/vision/qwen2_vl_2b/sft/train.yaml` |

### 🎯 Training Techniques

| Model | Configuration | Links |
|-------|--------------|-------|
| Llama 3.1 8B | `recipes/llama3_1/sft/8b_lora/fsdp_train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml` |
| | `recipes/llama3_1/sft/8b_full/longctx_train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml` |
| Phi-3 | `recipes/phi3/dpo/train.yaml` | {download}`Download </../configs/recipes/phi3/dpo/train.yaml>` {gh}`configs/recipes/phi3/dpo/train.yaml` |
| | `recipes/phi3/dpo/fsdp_nvidia_24g_train.yaml` | {download}`Download </../configs/recipes/phi3/dpo/fsdp_nvidia_24g_train.yaml>` {gh}`configs/recipes/phi3/dpo/fsdp_nvidia_24g_train.yaml` |
| FineWeb Pretraining | `examples/fineweb_ablation_pretraining/ddp/train.yaml` | {download}`Download </../configs/examples/fineweb_ablation_pretraining/ddp/train.yaml>` {gh}`configs/examples/fineweb_ablation_pretraining/ddp/train.yaml` |
| | `examples/fineweb_ablation_pretraining/fsdp/train.yaml` | {download}`Download </../configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml>` {gh}`configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml` |

### 🚀 Inference

| Model | Configuration | Links |
|-------|--------------|-------|
| Llama 3.1 8B | `recipes/llama3_1/inference/8b_infer.yaml` | {download}`Download </../configs/recipes/llama3_1/inference/8b_infer.yaml>` {gh}`configs/recipes/llama3_1/inference/8b_infer.yaml` |
| Llama 3.1 70B | `recipes/llama3_1/inference/70b_infer.yaml` | {download}`Download </../configs/recipes/llama3_1/inference/70b_infer.yaml>` {gh}`configs/recipes/llama3_1/inference/70b_infer.yaml` |
| Llama 3.2 Vision 11B | `recipes/vision/llama3_2_vision/inference/11b_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml` |
| | `recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml` |
| | `recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml` |
| GPT-2 | `recipes/gpt2/inference/infer.yaml` | {download}`Download </../configs/recipes/gpt2/inference/infer.yaml>` {gh}`configs/recipes/gpt2/inference/infer.yaml` |
| Mistral | `examples/bulk_inference/mistral_small_infer.yaml` | {download}`Download </../configs/examples/bulk_inference/mistral_small_infer.yaml>` {gh}`configs/examples/bulk_inference/mistral_small_infer.yaml` |


## Additional Resources

- [Training Guide](/user_guides/train/train.md)
- [Inference Guide](/user_guides/infer/infer.md)
- [Example Notebooks](https://github.com/oumi-ai/oumi/tree/main/notebooks)
