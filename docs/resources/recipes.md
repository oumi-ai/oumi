# Recipes

To help you get started with Oumi, we've prepared a set of recipes for common use cases. These recipes are designed to be easy to understand and modify, and should be a good starting point for your own projects. Each recipe is a YAML file that can be used to train, evaluate, or deploy a model. We also have corresponding job configs for most recipes that let you run the job remotely; they're usually files ending in `_job.yaml` in the same directory as the recipe config.

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

You can also check out the `README.md` in each recipe's directory for more details and examples. You can easily adapt these recipes to use with other supported {doc}`models </resources/models/models>`, {doc}`datasets </resources/datasets/datasets>`, and {doc}`cloud providers </user_guides/launch/launch>`.

## Common Models

### 🐋 DeepSeek R1 Family

| Model | Configuration | Links |
|-------|--------------|-------|
| DeepSeek R1 671B | `recipes/deepseek_r1/inference/671b_together_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/671b_together_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/671b_together_infer.yaml` |
| Distilled Llama 8B | `recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml` | {download}`Download </../configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml>` {gh}`configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml` |
| | `recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml` | {download}`Download </../configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml>` {gh}`configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml` |
| | `recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml` | {download}`Download </../configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml>` {gh}`configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml` |
| | `recipes/deepseek_r1/evaluation/distill_llama_8b/eval.yaml` | {download}`Download </../configs/recipes/deepseek_r1/evaluation/distill_llama_8b/eval.yaml>` {gh}`configs/recipes/deepseek_r1/evaluation/distill_llama_8b/eval.yaml` |
| | `recipes/deepseek_r1/inference/distill_llama_8b_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/distill_llama_8b_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/distill_llama_8b_infer.yaml` |
| Distilled Llama 70B | `recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml` | {download}`Download </../configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml>` {gh}`configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml` |
| | `recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml` | {download}`Download </../configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml>` {gh}`configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml` |
| | `recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml` | {download}`Download </../configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml>` {gh}`configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml` |
| | `recipes/deepseek_r1/evaluation/distill_llama_70b/eval.yaml` | {download}`Download </../configs/recipes/deepseek_r1/evaluation/distill_llama_70b/eval.yaml>` {gh}`configs/recipes/deepseek_r1/evaluation/distill_llama_70b/eval.yaml` |
| | `recipes/deepseek_r1/inference/distill_llama_70b_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/distill_llama_70b_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/distill_llama_70b_infer.yaml` |
| Distilled Qwen 1.5B | `recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml` | {download}`Download </../configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml>` {gh}`configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml` |
| | `recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml` | {download}`Download </../configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml>` {gh}`configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml` |
| | `recipes/deepseek_r1/evaluation/distill_qwen_1_5b/eval.yaml` | {download}`Download </../configs/recipes/deepseek_r1/evaluation/distill_qwen_1_5b/eval.yaml>` {gh}`configs/recipes/deepseek_r1/evaluation/distill_qwen_1_5b/eval.yaml` |
| | `recipes/deepseek_r1/inference/distill_qwen_1_5b_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/distill_qwen_1_5b_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/distill_qwen_1_5b_infer.yaml` |
| Distilled Qwen 32B | `recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml` | {download}`Download </../configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml>` {gh}`configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml` |
| | `recipes/deepseek_r1/evaluation/distill_qwen_32b/eval.yaml` | {download}`Download </../configs/recipes/deepseek_r1/evaluation/distill_qwen_32b/eval.yaml>` {gh}`configs/recipes/deepseek_r1/evaluation/distill_qwen_32b/eval.yaml` |
| | `recipes/deepseek_r1/inference/distill_qwen_32b_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/distill_qwen_32b_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/distill_qwen_32b_infer.yaml` |

### 🦙 Llama Family

| Model | Configuration | Links |
|-------|--------------|-------|
| Llama 3.1 8B | `recipes/llama3_1/sft/8b_full/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_full/train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_full/train.yaml` |
| | `recipes/llama3_1/sft/8b_lora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_lora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_lora/train.yaml` |
| | `recipes/llama3_1/sft/8b_qlora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_qlora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_qlora/train.yaml` |
| | `recipes/llama3_1/pretraining/8b/train.yaml` | {download}`Download </../configs/recipes/llama3_1/pretraining/8b/train.yaml>` {gh}`configs/recipes/llama3_1/pretraining/8b/train.yaml` |
| | `recipes/llama3_1/evaluation/8b_eval.yaml` | {download}`Download </../configs/recipes/llama3_1/evaluation/8b_eval.yaml>` {gh}`configs/recipes/llama3_1/evaluation/8b_eval.yaml` |
| | `recipes/llama3_1/inference/8b_infer.yaml` | {download}`Download </../configs/recipes/llama3_1/inference/8b_infer.yaml>` {gh}`configs/recipes/llama3_1/inference/8b_infer.yaml` |
| Llama 3.3 70B | `recipes/llama3_3/sft/70b_full/train.yaml` | {download}`Download </../configs/recipes/llama3_3/sft/70b_full/train.yaml>` {gh}`configs/recipes/llama3_3/sft/70b_full/train.yaml` |
| | `recipes/llama3_3/sft/70b_lora/train.yaml` | {download}`Download </../configs/recipes/llama3_3/sft/70b_lora/train.yaml>` {gh}`configs/recipes/llama3_3/sft/70b_lora/train.yaml` |
| | `recipes/llama3_3/sft/70b_qlora/train.yaml` | {download}`Download </../configs/recipes/llama3_3/sft/70b_qlora/train.yaml>` {gh}`configs/recipes/llama3_3/sft/70b_qlora/train.yaml` |
| | `recipes/llama3_3/evaluation/70b_eval.yaml` | {download}`Download </../configs/recipes/llama3_3/evaluation/70b_eval.yaml>` {gh}`configs/recipes/llama3_3/evaluation/70b_eval.yaml` |
| | `recipes/llama3_3/inference/70b_infer.yaml` | {download}`Download </../configs/recipes/llama3_3/inference/70b_infer.yaml>` {gh}`configs/recipes/llama3_3/inference/70b_infer.yaml` |
| Llama 3.1 405B | `recipes/llama3_1/sft/405b_full/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/405b_full/train.yaml>` {gh}`configs/recipes/llama3_1/sft/405b_full/train.yaml` |
| | `recipes/llama3_1/sft/405b_lora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/405b_lora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/405b_lora/train.yaml` |
| | `recipes/llama3_1/sft/405b_qlora/train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/405b_qlora/train.yaml>` {gh}`configs/recipes/llama3_1/sft/405b_qlora/train.yaml` |
| Llama 3.2 1B | `recipes/llama3_2/sft/1b_full/train.yaml` | {download}`Download </../configs/recipes/llama3_2/sft/1b_full/train.yaml>` {gh}`configs/recipes/llama3_2/sft/1b_full/train.yaml` |
| | `recipes/llama3_2/evaluation/1b_eval.yaml` | {download}`Download </../configs/recipes/llama3_2/evaluation/1b_eval.yaml>` {gh}`configs/recipes/llama3_2/evaluation/1b_eval.yaml` |
| | `recipes/llama3_2/inference/1b_infer.yaml` | {download}`Download </../configs/recipes/llama3_2/inference/1b_infer.yaml>` {gh}`configs/recipes/llama3_2/inference/1b_infer.yaml` |
| Llama 3.2 3B | `recipes/llama3_2/sft/3b_full/train.yaml` | {download}`Download </../configs/recipes/llama3_2/sft/3b_full/train.yaml>` {gh}`configs/recipes/llama3_2/sft/3b_full/train.yaml` |
| | `recipes/llama3_2/sft/3b_lora/train.yaml` | {download}`Download </../configs/recipes/llama3_2/sft/3b_lora/train.yaml>` {gh}`configs/recipes/llama3_2/sft/3b_lora/train.yaml` |
| | `recipes/llama3_2/sft/3b_qlora/train.yaml` | {download}`Download </../configs/recipes/llama3_2/sft/3b_qlora/train.yaml>` {gh}`configs/recipes/llama3_2/sft/3b_qlora/train.yaml` |
| | `recipes/llama3_2/evaluation/3b_eval.yaml` | {download}`Download </../configs/recipes/llama3_2/evaluation/3b_eval.yaml>` {gh}`configs/recipes/llama3_2/evaluation/3b_eval.yaml` |
| | `recipes/llama3_2/inference/3b_infer.yaml` | {download}`Download </../configs/recipes/llama3_2/inference/3b_infer.yaml>` {gh}`configs/recipes/llama3_2/inference/3b_infer.yaml` |

### 🎨 Vision Models

| Model | Configuration | Links |
|-------|---------------|-------|
| Llama 3.2 Vision 11B | `recipes/vision/llama3_2_vision/sft/11b_full/train.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml` |
| | `recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml` |
| | `recipes/vision/llama3_2_vision/inference/11b_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml` |
| | `recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml` |
| | `recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml` |
| LLaVA 7B | `recipes/vision/llava_7b/sft/train.yaml` | {download}`Download </../configs/recipes/vision/llava_7b/sft/train.yaml>` {gh}`configs/recipes/vision/llava_7b/sft/train.yaml` |
| | `recipes/vision/llava_7b/inference/infer.yaml` | {download}`Download </../configs/recipes/vision/llava_7b/inference/infer.yaml>` {gh}`configs/recipes/vision/llava_7b/inference/infer.yaml` |
| | `recipes/vision/llava_7b/inference/vllm_infer.yaml` | {download}`Download </../configs/recipes/vision/llava_7b/inference/vllm_infer.yaml>` {gh}`configs/recipes/vision/llava_7b/inference/vllm_infer.yaml` |
| Phi3 Vision | `recipes/vision/phi3/sft/full/train.yaml` | {download}`Download </../configs/recipes/vision/phi3/sft/full/train.yaml>` {gh}`configs/recipes/vision/phi3/sft/full/train.yaml` |
| | `recipes/vision/phi3/inference/vllm_infer.yaml` | {download}`Download </../configs/recipes/vision/phi3/inference/vllm_infer.yaml>` {gh}`configs/recipes/vision/phi3/inference/vllm_infer.yaml` |
| Phi4 Multimodal | `recipes/vision/phi4/sft/full/train.yaml` | {download}`Download </../configs/recipes/vision/phi4/sft/full/train.yaml>` {gh}`configs/recipes/vision/phi4/sft/full/train.yaml` |
| | `recipes/vision/phi4/inference/vllm_infer.yaml` | {download}`Download </../configs/recipes/vision/phi4/inference/vllm_infer.yaml>` {gh}`configs/recipes/vision/phi4/inference/vllm_infer.yaml` |
| Qwen2-VL 2B | `recipes/vision/qwen2_vl_2b/sft/full/train.yaml` | {download}`Download </../configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml>` {gh}`configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml` |
| | `recipes/vision/qwen2_vl_2b/inference/infer.yaml` | {download}`Download </../configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml>` {gh}`configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml` |
| | `recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml` | {download}`Download </../configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml>` {gh}`configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml` |
| | `recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml` | {download}`Download </../configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml>` {gh}`configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml` |
| Qwen2.5-VL 3B | `recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml` | {download}`Download </../configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml>` {gh}`configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml` |
| | `recipes/vision/qwen2_5_vl_3b/inference/infer.yaml` | {download}`Download </../configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml>` {gh}`configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml` |
| | `recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml` | {download}`Download </../configs/recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml>` {gh}`configs/recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml` |
| SmolVLM | `recipes/vision/smolvlm/sft/full/train.yaml` | {download}`Download </../configs/recipes/vision/smolvlm/sft/full/train.yaml>` {gh}`configs/recipes/vision/smolvlm/sft/full/train.yaml` |

### 🎯 Training Techniques

This section lists an example config for various training techniques supported by Oumi.

| Technique | Configuration | Links |
|-------|--------------|-------|
| FSDP | `recipes/llama3_1/sft/8b_lora/fsdp_train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml` |
| Long-context training | `recipes/llama3_1/sft/8b_full/longctx_train.yaml` | {download}`Download </../configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml>` {gh}`configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml` |
| DPO | `recipes/phi3/dpo/train.yaml` | {download}`Download </../configs/recipes/phi3/dpo/train.yaml>` {gh}`configs/recipes/phi3/dpo/train.yaml` |
| GRPO | `examples/grpo_tldr/train.yaml` | {download}`Download </../configs/examples/grpo_tldr/train.yaml>` {gh}`configs/examples/grpo_tldr/train.yaml` |
| DDP Pretraining | `examples/fineweb_ablation_pretraining/ddp/train.yaml` | {download}`Download </../configs/examples/fineweb_ablation_pretraining/ddp/train.yaml>` {gh}`configs/examples/fineweb_ablation_pretraining/ddp/train.yaml` |
| FSDP Pretraining | `examples/fineweb_ablation_pretraining/fsdp/train.yaml` | {download}`Download </../configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml>` {gh}`configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml` |

### 🚀 Inference

| Model | Configuration | Links |
|-------|--------------|-------|
| DeepSeek R1 671B | `recipes/deepseek_r1/inference/671b_together_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/671b_together_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/671b_together_infer.yaml` |
| DeepSeek R1 Distill Llama 8B | `recipes/deepseek_r1/inference/distill_llama_8b_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/distill_llama_8b_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/distill_llama_8b_infer.yaml` |
| DeepSeek R1 Distill Llama 70B | `recipes/deepseek_r1/inference/distill_llama_70b_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/distill_llama_70b_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/distill_llama_70b_infer.yaml` |
| DeepSeek R1 Distill Qwen 1.5B | `recipes/deepseek_r1/inference/distill_qwen_1_5b_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/distill_qwen_1_5b_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/distill_qwen_1_5b_infer.yaml` |
| DeepSeek R1 Distill Qwen 32B | `recipes/deepseek_r1/inference/distill_qwen_32b_infer.yaml` | {download}`Download </../configs/recipes/deepseek_r1/inference/distill_qwen_32b_infer.yaml>` {gh}`configs/recipes/deepseek_r1/inference/distill_qwen_32b_infer.yaml` |
| Llama 3.1 8B | `recipes/llama3_1/inference/8b_infer.yaml` | {download}`Download </../configs/recipes/llama3_1/inference/8b_infer.yaml>` {gh}`configs/recipes/llama3_1/inference/8b_infer.yaml` |
| | `recipes/llama3_1/inference/8b_sglang_infer.yaml` | {download}`Download </../configs/recipes/llama3_1/inference/8b_sglang_infer.yaml>` {gh}`configs/recipes/llama3_1/inference/8b_sglang_infer.yaml` |
| | `recipes/llama3_1/inference/8b_rvllm_infer.yaml` | {download}`Download </../configs/recipes/llama3_1/inference/8b_rvllm_infer.yaml>` {gh}`configs/recipes/llama3_1/inference/8b_rvllm_infer.yaml` |
| Llama 3.1 70B | `recipes/llama3_1/inference/70b_infer.yaml` | {download}`Download </../configs/recipes/llama3_1/inference/70b_infer.yaml>` {gh}`configs/recipes/llama3_1/inference/70b_infer.yaml` |
| Llama 3.2 1B | `recipes/llama3_2/inference/1b_infer.yaml` | {download}`Download </../configs/recipes/llama3_2/inference/1b_infer.yaml>` {gh}`configs/recipes/llama3_2/inference/1b_infer.yaml` |
| | `recipes/llama3_2/inference/1b_sglang_infer.yaml` | {download}`Download </../configs/recipes/llama3_2/inference/1b_sglang_infer.yaml>` {gh}`configs/recipes/llama3_2/inference/1b_sglang_infer.yaml` |
| | `recipes/llama3_2/inference/1b_vllm_infer.yaml` | {download}`Download </../configs/recipes/llama3_2/inference/1b_vllm_infer.yaml>` {gh}`configs/recipes/llama3_2/inference/1b_vllm_infer.yaml` |
| Llama 3.2 3B | `recipes/llama3_2/inference/3b_infer.yaml` | {download}`Download </../configs/recipes/llama3_2/inference/3b_infer.yaml>` {gh}`configs/recipes/llama3_2/inference/3b_infer.yaml` |
| | `recipes/llama3_2/inference/3b_sglang_infer.yaml` | {download}`Download </../configs/recipes/llama3_2/inference/3b_sglang_infer.yaml>` {gh}`configs/recipes/llama3_2/inference/3b_sglang_infer.yaml` |
| | `recipes/llama3_2/inference/3b_vllm_infer.yaml` | {download}`Download </../configs/recipes/llama3_2/inference/3b_vllm_infer.yaml>` {gh}`configs/recipes/llama3_2/inference/3b_vllm_infer.yaml` |
| Llama 3.2 Vision 11B | `recipes/vision/llama3_2_vision/inference/11b_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml` |
| | `recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml` |
| | `recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml` | {download}`Download </../configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml>` {gh}`configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml` |
| GPT-2 | `recipes/gpt2/inference/infer.yaml` | {download}`Download </../configs/recipes/gpt2/inference/infer.yaml>` {gh}`configs/recipes/gpt2/inference/infer.yaml` |

## Additional Resources

- [Training Guide](/user_guides/train/train.md)
- [Inference Guide](/user_guides/infer/infer.md)
- [Example Notebooks](https://github.com/oumi-ai/oumi/tree/main/notebooks)
