# Config Health Summary

Scanned **495** configs in 314.9s.

## Results

| Status | Count |
|--------|-------|
| Healthy | 336 |
| Warnings | 5 |
| Failing | 154 |
| **Total** | **495** |

## By Config Type

| Type | Count |
|------|-------|
| training | 150 |
| job | 140 |
| inference | 126 |
| evaluation | 52 |
| judge | 15 |
| synthesis | 7 |
| quantization | 2 |
| analyze | 1 |
| async_evaluation | 1 |
| tuning | 1 |

## Failures (156)

- **configs/recipes/olmo3/evaluation/32b/eval.yaml**: Model not found on HF Hub: allenai/Olmo-3-32B-Instruct
- **configs/recipes/olmo3/inference/32b_infer.yaml**: Model not found on HF Hub: allenai/Olmo-3-32B-Instruct
- **configs/recipes/olmo3/sft/32b_lora/train.yaml**: Model not found on HF Hub: allenai/Olmo-3-32B-Instruct
- **configs/recipes/qwen3/inference/30b_a3b_instruct_vllm_infer.yaml**: Model not found on HF Hub: Qwen/Qwen3-30B-A3B-Instruct
- **configs/recipes/qwen3_next/evaluation/80b_a3b_eval.yaml**: Model not found on HF Hub: Qwen/Qwen3-Next-80B-A3B
- **configs/recipes/qwen3_next/inference/80b_a3b_infer.yaml**: Model not found on HF Hub: Qwen/Qwen3-Next-80B-A3B
- **configs/recipes/qwen3_next/sft/80b_a3b_lora/train.yaml**: Model not found on HF Hub: Qwen/Qwen3-Next-80B-A3B
- **configs/examples/deepspeed/llama3_1_8b_deepspeed_z2_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/examples/deepspeed/llama3_1_8b_deepspeed_z3_offload_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/examples/deepspeed/llama3_1_8b_deepspeed_z3_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/examples/fineweb_ablation_pretraining/ddp/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: HuggingFaceFW/ablation-model-fineweb-v1
- **configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: HuggingFaceFW/ablation-model-fineweb-v1
- **configs/examples/gkd/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: HuggingFaceTB/SmolLM2-135M-Instruct
- **configs/examples/gold/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: HuggingFaceTB/SmolLM2-135M-Instruct
- **configs/examples/gold/train_gptoss120b_qwen06b.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-0.6B
- **configs/examples/gold/train_gptoss120b_qwen8b.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-8B
- **configs/examples/grpo_tldr/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2-0.5B-Instruct
- **configs/examples/grpo_verl_countdown/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: d1shs0ap/cognitive-behaviors-Llama-3.2-3B
- **configs/examples/grpo_verl_geometry3k/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-VL-3B-Instruct
- **configs/examples/grpo_verl_gsm8k/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-0.5B-Instruct
- **configs/examples/letter_counting/grpo/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-3B-Instruct
- **configs/examples/misc/tulu3_sft_mini.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: EleutherAI/pythia-14m
- **configs/projects/aya/sft/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/projects/chatqa/chatqa_stage1_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-mini-4k-instruct
- **configs/projects/chatqa/chatqa_stage2_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-mini-4k-instruct
- **configs/projects/coalm/405b_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-405B-Instruct
- **configs/projects/coalm/70b_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.3-70B-Instruct
- **configs/projects/coalm/8b_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: oumi-ai/Molmo-7B-D-0924
- **configs/projects/dcvlr/starter_kit/molmo-o-train-openr1.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: oumi-ai/Molmo-7B-O-0924
- **configs/projects/dcvlr/starter_kit/qwenvl-openr1.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-VL-7B-Instruct
- **configs/projects/halloumi/8b_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/projects/limo/qwen2.5_7b_fft.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-7B-Instruct
- **configs/projects/limo/qwen2.5_7b_fft_yarn.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-7B-Instruct
- **configs/projects/limo/qwen2.5_7b_fft_yarn_deepspeed.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-7B-Instruct
- **configs/projects/limo/qwen2.5_7b_fft_yarn_deepspeed_memory_optimized_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-7B-Instruct
- **configs/projects/wc50m/configs/base_ultrachat.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Meta-Llama-3.1-8B
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- **configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- **configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- **configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- **configs/recipes/falcon_e/dpo/falcon_e_1b_instruct/dpo.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-E-1B-Instruct
- **configs/recipes/falcon_e/sft/falcon_e_1b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-E-1B-Base
- **configs/recipes/falcon_e/sft/falcon_e_1b_instruct/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-E-1B-Instruct
- **configs/recipes/falcon_e/sft/falcon_e_3b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-E-3B-Base
- **configs/recipes/falcon_e/sft/falcon_e_3b_instruct/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-E-3B-Instruct
- **configs/recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-H1-0.5B-Instruct
- **configs/recipes/falcon_h1/sft/falcon_h1_0_5b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-H1-0.5B-Instruct
- **configs/recipes/falcon_h1/sft/falcon_h1_1_5b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-H1-1.5B-Instruct
- **configs/recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-H1-1.5B-Deep-Instruct
- **configs/recipes/falcon_h1/sft/falcon_h1_34b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-H1-34B-Instruct
- **configs/recipes/falcon_h1/sft/falcon_h1_3b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-H1-3B-Instruct
- **configs/recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: tiiuae/Falcon-H1-7B-Instruct
- **configs/recipes/gemma2/sft/2b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: google/gemma-2-2b-it
- **configs/recipes/gemma3/sft/12b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: google/gemma-3-12b-it
- **configs/recipes/gemma3/sft/27b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: google/gemma-3-27b-it
- **configs/recipes/gemma3/sft/4b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: google/gemma-3-4b-it
- **configs/recipes/gpt2/pretraining/macos_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: gpt2
- **configs/recipes/gpt2/pretraining/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: gpt2
- **configs/recipes/gpt_oss/sft/120b_lora_multi_gpu_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: openai/gpt-oss-120b
- **configs/recipes/gpt_oss/sft/20b_lora_multi_gpu_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: openai/gpt-oss-20b
- **configs/recipes/gpt_oss/sft/20b_lora_single_gpu_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: openai/gpt-oss-20b
- **configs/recipes/llama3_1/pretraining/8b/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B
- **configs/recipes/llama3_1/sft/405b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-405B-Instruct
- **configs/recipes/llama3_1/sft/405b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-405B-Instruct
- **configs/recipes/llama3_1/sft/405b_qlora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-405B-Instruct
- **configs/recipes/llama3_1/sft/70b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-70B-Instruct
- **configs/recipes/llama3_1/sft/70b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-70B-Instruct
- **configs/recipes/llama3_1/sft/70b_qlora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-70B-Instruct
- **configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/recipes/llama3_1/sft/8b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/recipes/llama3_1/sft/8b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/recipes/llama3_1/sft/8b_qlora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.1-8B-Instruct
- **configs/recipes/llama3_2/dpo/1b_qlora_dpo.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-1B-Instruct
- **configs/recipes/llama3_2/sft/1b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-1B-Instruct
- **configs/recipes/llama3_2/sft/3b_full/fsdp_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-3B-Instruct
- **configs/recipes/llama3_2/sft/3b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-3B-Instruct
- **configs/recipes/llama3_2/sft/3b_lora/fsdp_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-3B-Instruct
- **configs/recipes/llama3_2/sft/3b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-3B-Instruct
- **configs/recipes/llama3_2/sft/3b_qlora/fsdp_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-3B-Instruct
- **configs/recipes/llama3_2/sft/3b_qlora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-3B-Instruct
- **configs/recipes/llama3_3/sft/70b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.3-70B-Instruct
- **configs/recipes/llama3_3/sft/70b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.3-70B-Instruct
- **configs/recipes/llama3_3/sft/70b_qlora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.3-70B-Instruct
- **configs/recipes/llama4/sft/scout_base_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-4-Scout-17B-16E
- **configs/recipes/llama4/sft/scout_instruct_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-4-Scout-17B-16E-Instruct
- **configs/recipes/llama4/sft/scout_instruct_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-4-Scout-17B-16E-Instruct
- **configs/recipes/llama4/sft/scout_instruct_qlora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-4-Scout-17B-16E-Instruct
- **configs/recipes/olmo3/sft/32b_lora/train.yaml**: Dry-run failed: OSError: allenai/Olmo-3-32B-Instruct is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
  - model: allenai/Olmo-3-32B-Instruct
- **configs/recipes/olmo3/sft/7b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: allenai/Olmo-3-7B-Instruct
- **configs/recipes/phi3/dpo/macos_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-mini-4k-instruct
- **configs/recipes/phi3/dpo/nvidia_24g_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-mini-4k-instruct
- **configs/recipes/phi3/dpo/nvidia_80g_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-mini-4k-instruct
- **configs/recipes/phi3/dpo/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-mini-4k-instruct
- **configs/recipes/phi3/kto/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-mini-4k-instruct
- **configs/recipes/phi3/sft/lora_macos_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-mini-4k-instruct
- **configs/recipes/phi3/sft/lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-mini-4k-instruct
- **configs/recipes/phi4/sft/reasoning_plus/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-4-reasoning-plus
- **configs/recipes/phi4/sft/reasoning_plus/lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-4-reasoning-plus
- **configs/recipes/phi4/sft/reasoning_plus/qlora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-4-reasoning-plus
- **configs/recipes/qwen2_5/sft/3b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-3B-Instruct
- **configs/recipes/qwen2_5/sft/7b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-7B-Instruct
- **configs/recipes/qwen3/sft/0.6b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-0.6B
- **configs/recipes/qwen3/sft/1.7b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-1.7B
- **configs/recipes/qwen3/sft/14b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-14B
- **configs/recipes/qwen3/sft/235b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-235B-A22B
- **configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-30B-A3B
- **configs/recipes/qwen3/sft/32b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-32B
- **configs/recipes/qwen3/sft/4b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-4B
- **configs/recipes/qwen3/sft/8b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-8B
- **configs/recipes/qwen3_5/sft/0.8b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3.5-0.8B
- **configs/recipes/qwen3_5/sft/0.8b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3.5-0.8B
- **configs/recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-30B-A3B-Instruct-2507
- **configs/recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-Next-80B-A3B-Instruct
- **configs/recipes/qwen3_next/sft/80b_a3b_lora/train.yaml**: Dry-run failed: OSError: Qwen/Qwen3-Next-80B-A3B is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
  - model: Qwen/Qwen3-Next-80B-A3B
- **configs/recipes/qwq/sft/full_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/QwQ-32B
- **configs/recipes/qwq/sft/lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/QwQ-32B
- **configs/recipes/qwq/sft/qlora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/QwQ-32B
- **configs/recipes/smollm/sft/135m/quickstart_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: HuggingFaceTB/SmolLM2-135M-Instruct
- **configs/recipes/smollm/sft/135m/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: HuggingFaceTB/SmolLM2-135M-Instruct
- **configs/recipes/vision/internvl3/sft/full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: OpenGVLab/InternVL3-1B-hf
- **configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-11B-Vision-Instruct
- **configs/recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-11B-Vision-Instruct
- **configs/recipes/vision/llama3_2_vision/sft/90b_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: meta-llama/Llama-3.2-90B-Vision-Instruct
- **configs/recipes/vision/llava_7b/dpo/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: llava-hf/llava-1.5-7b-hf
- **configs/recipes/vision/llava_7b/sft/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: llava-hf/llava-1.5-7b-hf
- **configs/recipes/vision/molmo/grpo/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: oumi-ai/Molmo-7B-O-0924
- **configs/recipes/vision/molmo/sft/molmo_d_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: oumi-ai/Molmo-7B-D-0924
- **configs/recipes/vision/molmo/sft/molmo_o_full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: oumi-ai/Molmo-7B-O-0924
- **configs/recipes/vision/phi3/dpo/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-vision-128k-instruct
- **configs/recipes/vision/phi3/sft/full/completions_only_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-vision-128k-instruct
- **configs/recipes/vision/phi3/sft/full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-vision-128k-instruct
- **configs/recipes/vision/phi3/sft/lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-3-vision-128k-instruct
- **configs/recipes/vision/phi4/sft/full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-4-multimodal-instruct
- **configs/recipes/vision/phi4/sft/lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: microsoft/Phi-4-multimodal-instruct
- **configs/recipes/vision/qwen2_5_vl_3b/dpo/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-VL-3B-Instruct
- **configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-VL-3B-Instruct
- **configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-VL-3B-Instruct
- **configs/recipes/vision/qwen2_5_vl_7b/sft/full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2.5-VL-7B-Instruct
- **configs/recipes/vision/qwen2_vl_2b/dpo/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2-VL-2B-Instruct
- **configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2-VL-2B-Instruct
- **configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen2-VL-2B-Instruct
- **configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_fft_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-VL-2B-Instruct
- **configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-VL-2B-Instruct
- **configs/recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_fft_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-VL-30B-A3B-Instruct
- **configs/recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-VL-30B-A3B-Instruct
- **configs/recipes/vision/qwen3_vl_4b/sft/4b_instruct_fft_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-VL-4B-Instruct
- **configs/recipes/vision/qwen3_vl_4b/sft/4b_instruct_lora_train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: Qwen/Qwen3-VL-4B-Instruct
- **configs/recipes/vision/smolvlm/sft/full/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: HuggingFaceTB/SmolVLM-Instruct
- **configs/recipes/vision/smolvlm/sft/lora/train.yaml**: Dry-run failed: TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
  - model: HuggingFaceTB/SmolVLM-Instruct

## Warnings (69)

- **configs/examples/fineweb_ablation_pretraining/ddp/train.yaml**: pad_token is set to eos_token. The model may learn to ignore EOS, causing generation to never terminate. Consider using a dedicated pad token.
- **configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml**: pad_token is set to eos_token. The model may learn to ignore EOS, causing generation to never terminate. Consider using a dedicated pad token.
- **configs/examples/gkd/train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/examples/gold/train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/examples/grpo_verl_countdown/train.yaml**: Tokenizer's pad_token == eos_token ('<|end_of_text|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/projects/chatqa/chatqa_stage1_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/projects/chatqa/chatqa_stage2_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/gemma3/sft/12b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/gemma3/sft/27b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/glm4/inference/air_vllm_infer.yaml**: Model 'THUDM/glm-4-9b-chat' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/gpt_oss/sft/120b_lora_multi_gpu_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/gpt_oss/sft/20b_lora_multi_gpu_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_1/sft/405b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_1/sft/8b_qlora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_3/inference/nemotron_super_49b_vllm_infer.yaml**: Model 'nvidia/Llama-3.3-Nemotron-Super-49B-v1' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/olmo3/evaluation/32b/eval.yaml**: Cannot load model config: allenai/Olmo-3-32B-Instruct (may be unreleased, gated, or renamed)
- **configs/recipes/olmo3/inference/32b_infer.yaml**: Cannot load model config: allenai/Olmo-3-32B-Instruct (may be unreleased, gated, or renamed)
- **configs/recipes/olmo3/sft/32b_lora/train.yaml**: Cannot load model config: allenai/Olmo-3-32B-Instruct (may be unreleased, gated, or renamed)
- **configs/recipes/phi3/dpo/macos_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/dpo/nvidia_24g_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/dpo/nvidia_80g_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/dpo/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/kto/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/sft/lora_macos_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/sft/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi4/sft/reasoning_plus/lora_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/inference/30b_a3b_instruct_vllm_infer.yaml**: Cannot load model config: Qwen/Qwen3-30B-A3B-Instruct (may be unreleased, gated, or renamed)
- **configs/recipes/qwen3/sft/14b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/235b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/32b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3_next/evaluation/80b_a3b_eval.yaml**: Cannot load model config: Qwen/Qwen3-Next-80B-A3B (may be unreleased, gated, or renamed)
- **configs/recipes/qwen3_next/inference/80b_a3b_infer.yaml**: Cannot load model config: Qwen/Qwen3-Next-80B-A3B (may be unreleased, gated, or renamed)
- **configs/recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3_next/sft/80b_a3b_lora/train.yaml**: Cannot load model config: Qwen/Qwen3-Next-80B-A3B (may be unreleased, gated, or renamed)
- **configs/recipes/qwq/sft/lora_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/smollm/sft/135m/quickstart_train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/smollm/sft/135m/train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/vision/molmo/sft/molmo_d_full/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/dpo/train.yaml**: Model 'microsoft/Phi-3-vision-128k-instruct' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/vision/phi3/dpo/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/inference/vllm_infer.yaml**: Model 'microsoft/Phi-3-vision-128k-instruct' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/vision/phi3/sft/full/completions_only_train.yaml**: Model 'microsoft/Phi-3-vision-128k-instruct' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/vision/phi3/sft/full/completions_only_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/sft/full/train.yaml**: Model 'microsoft/Phi-3-vision-128k-instruct' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/vision/phi3/sft/full/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/sft/lora/train.yaml**: Model 'microsoft/Phi-3-vision-128k-instruct' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/vision/phi3/sft/lora/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi4/inference/infer.yaml**: Model 'microsoft/Phi-4-multimodal-instruct' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/vision/phi4/inference/vllm_infer.yaml**: Model 'microsoft/Phi-4-multimodal-instruct' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/vision/phi4/sft/full/train.yaml**: Model 'microsoft/Phi-4-multimodal-instruct' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/vision/phi4/sft/full/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi4/sft/lora/train.yaml**: Model 'microsoft/Phi-4-multimodal-instruct' requires trust_remote_code=True but the config doesn't set it.
- **configs/recipes/vision/phi4/sft/lora/train.yaml**: Could not load architecture: Could not instantiate model on meta device
- **configs/recipes/vision/phi4/sft/lora/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_lora_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006

## Coverage Gaps (11)

- **openrouter** (apis): missing evaluation
- **falcon_e** (recipes): missing inference
- **gemma2** (recipes): missing evaluation, inference
- **glm4** (recipes): missing evaluation, training
- **gpt2** (recipes): missing evaluation
- **gpt_oss** (recipes): missing evaluation
- **phi3** (recipes): missing inference
- **qwen2_5** (recipes): missing evaluation, inference
- **qwen3_5** (recipes): missing evaluation
- **qwen3_coder** (recipes): missing evaluation, training
- **qwen3_instruct** (recipes): missing evaluation, inference

## VRAM Estimates (147 training configs)

| Config | VRAM | Min VRAM | Params | Type |
|--------|------|----------|--------|------|
| examples/deepspeed/llama3_1_8b_deepspeed_z2_train.yaml | 294.2 GB | 46.3 GB | 7.5B | FFT |
| examples/deepspeed/llama3_1_8b_deepspeed_z3_offload_train.yaml | 168.8 GB | 46.3 GB | 7.5B | FFT |
| examples/deepspeed/llama3_1_8b_deepspeed_z3_train.yaml | 545.1 GB | 46.3 GB | 7.5B | FFT |
| examples/fineweb_ablation_pretraining/ddp/train.yaml | 45.2 GB | 20.7 GB | 1.7B | FFT |
| examples/fineweb_ablation_pretraining/fsdp/train.yaml | 68.4 GB | 11.4 GB | 1.7B | FFT |
| examples/gkd/train.yaml | 3.6 GB | 2.4 GB | 0.1B | FFT |
| examples/gold/train.yaml | 5.2 GB | 2.4 GB | 0.1B | FFT |
| examples/gold/train_gptoss120b_qwen06b.yaml | 8.6 GB | 6.8 GB | 0.5B | FFT |
| examples/gold/train_gptoss120b_qwen8b.yaml | 97.5 GB | 88.2 GB | 7.6B | FFT |
| examples/grpo_tldr/train.yaml | 28.3 GB | 6.5 GB | 0.5B | FFT |
| examples/grpo_verl_countdown/train.yaml | 14455.4 GB | 38.3 GB | 3.2B | FFT |
| examples/grpo_verl_geometry3k/train.yaml | 11835.0 GB | 36.7 GB | 3.1B | FFT |
| examples/grpo_verl_gsm8k/train.yaml | 470.7 GB | 6.6 GB | 0.5B | FFT |
| examples/letter_counting/grpo/train.yaml | 125.7 GB | 38.2 GB | 3.2B | FFT |
| examples/misc/tulu3_sft_mini.yaml | 0.8 GB | 0.6 GB | 0.0B | FFT |
| projects/aya/sft/train.yaml | 150.7 GB | 87.1 GB | 7.5B | FFT |
| projects/chatqa/chatqa_stage1_train.yaml | 58.7 GB | 44.2 GB | 3.7B | FFT |
| projects/chatqa/chatqa_stage2_train.yaml | 58.7 GB | 44.2 GB | 3.7B | FFT |
| projects/coalm/405b_train.yaml | 1087.5 GB | 434.9 GB | 403.8B | LoRA+Q |
| projects/coalm/70b_train.yaml | 1165.6 GB | 405.9 GB | 69.5B | FFT |
| projects/coalm/8b_train.yaml | 362.2 GB | 11.3 GB | 7.5B | LoRA |
| projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml | 395.3 GB | 103.6 GB | 12.8B | FFT |
| projects/dcvlr/starter_kit/molmo-o-train-openr1.yaml | 466.7 GB | 92.4 GB | 11.2B | FFT |
| projects/dcvlr/starter_kit/qwenvl-openr1.yaml | 82.3 GB | 43.1 GB | 7.1B | FFT |
| projects/halloumi/8b_train.yaml | 750.5 GB | 46.1 GB | 7.5B | FFT |
| projects/limo/qwen2.5_7b_fft.yaml | 360.9 GB | 43.2 GB | 7.1B | FFT |
| projects/limo/qwen2.5_7b_fft_yarn.yaml | 16869.1 GB | 43.2 GB | 7.1B | FFT |
| projects/limo/qwen2.5_7b_fft_yarn_deepspeed.yaml | 16869.1 GB | 43.2 GB | 7.1B | FFT |
| projects/limo/qwen2.5_7b_fft_yarn_deepspeed_memory_optimized_train.yaml | 16869.1 GB | 43.2 GB | 7.1B | FFT |
| projects/wc50m/configs/base_ultrachat.yaml | 128.6 GB | 87.1 GB | 7.5B | FFT |
| recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml | 502.6 GB | 405.9 GB | 69.5B | FFT |
| recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml | 179.2 GB | 82.5 GB | 69.5B | LoRA |
| recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml | 179.9 GB | 83.2 GB | 69.5B | LoRA+Q |
| recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml | 131.7 GB | 46.1 GB | 7.5B | FFT |
| recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml | 150.5 GB | 17.3 GB | 7.5B | LoRA |
| recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml | 103.1 GB | 17.4 GB | 7.5B | LoRA+Q |
| recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml | 75.8 GB | 18.7 GB | 1.5B | FFT |
| recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml | 93.3 GB | 4.9 GB | 1.5B | LoRA |
| recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml | 87.5 GB | 39.2 GB | 32.0B | LoRA |
| recipes/falcon_e/dpo/falcon_e_1b_instruct/dpo.yaml | 31.7 GB | 20.0 GB | 1.7B | FFT |
| recipes/falcon_e/sft/falcon_e_1b/full_train.yaml | 25.3 GB | 20.0 GB | 1.7B | FFT |
| recipes/falcon_e/sft/falcon_e_1b_instruct/full_train.yaml | 25.3 GB | 20.0 GB | 1.7B | FFT |
| recipes/falcon_e/sft/falcon_e_3b/full_train.yaml | 42.4 GB | 35.3 GB | 3.0B | FFT |
| recipes/falcon_e/sft/falcon_e_3b_instruct/full_train.yaml | 42.4 GB | 35.3 GB | 3.0B | FFT |
| recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml | 77.7 GB | 2.0 GB | 0.4B | LoRA+Q |
| recipes/falcon_h1/sft/falcon_h1_0_5b/full_train.yaml | 9.2 GB | 5.2 GB | 0.4B | FFT |
| recipes/falcon_h1/sft/falcon_h1_1_5b/full_train.yaml | 37.9 GB | 13.4 GB | 1.1B | FFT |
| recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_train.yaml | 35.3 GB | 15.1 GB | 1.2B | FFT |
| recipes/falcon_h1/sft/falcon_h1_34b/full_train.yaml | 292.0 GB | 175.5 GB | 29.6B | FFT |
| recipes/falcon_h1/sft/falcon_h1_3b/full_train.yaml | 34.2 GB | 14.7 GB | 2.2B | FFT |
| recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml | 47.9 GB | 39.4 GB | 6.4B | FFT |
| recipes/gemma2/sft/2b_full/train.yaml | 111.0 GB | 31.5 GB | 2.7B | FFT |
| recipes/gemma3/sft/12b_lora/train.yaml | 74.5 GB | 16.2 GB | 11.6B | LoRA |
| recipes/gemma3/sft/27b_lora/train.yaml | 84.9 GB | 35.7 GB | 28.3B | LoRA |
| recipes/gemma3/sft/4b_full/train.yaml | 82.1 GB | 25.2 GB | 4.0B | FFT |
| recipes/gpt2/pretraining/macos_train.yaml | 1.7 GB | 1.6 GB | 0.1B | FFT |
| recipes/gpt2/pretraining/train.yaml | 39.8 GB | 1.8 GB | 0.1B | FFT |
| recipes/gpt_oss/sft/120b_lora_multi_gpu_train.yaml | 103.2 GB | 42.4 GB | 115.9B | LoRA+Q |
| recipes/gpt_oss/sft/20b_lora_multi_gpu_train.yaml | 155.6 GB | 10.2 GB | 20.1B | LoRA+Q |
| recipes/gpt_oss/sft/20b_lora_single_gpu_train.yaml | 157.2 GB | 11.9 GB | 20.1B | LoRA+Q |
| recipes/llama3_1/pretraining/8b/train.yaml | 220.1 GB | 46.1 GB | 7.5B | FFT |
| recipes/llama3_1/sft/405b_full/train.yaml | 2442.6 GB | 2312.1 GB | 403.8B | FFT |
| recipes/llama3_1/sft/405b_lora/train.yaml | 563.0 GB | 432.4 GB | 403.8B | LoRA |
| recipes/llama3_1/sft/405b_qlora/train.yaml | 1085.4 GB | 432.7 GB | 403.8B | LoRA+Q |
| recipes/llama3_1/sft/70b_full/train.yaml | 778.9 GB | 405.9 GB | 69.5B | FFT |
| recipes/llama3_1/sft/70b_lora/train.yaml | 179.2 GB | 82.5 GB | 69.5B | LoRA |
| recipes/llama3_1/sft/70b_qlora/train.yaml | 179.9 GB | 83.2 GB | 69.5B | LoRA+Q |
| recipes/llama3_1/sft/8b_full/longctx_train.yaml | 356.4 GB | 87.1 GB | 7.5B | FFT |
| recipes/llama3_1/sft/8b_full/train.yaml | 131.7 GB | 46.1 GB | 7.5B | FFT |
| recipes/llama3_1/sft/8b_lora/fsdp_train.yaml | 362.0 GB | 11.1 GB | 7.5B | LoRA |
| recipes/llama3_1/sft/8b_lora/train.yaml | 150.5 GB | 17.3 GB | 7.5B | LoRA |
| recipes/llama3_1/sft/8b_qlora/train.yaml | 96.9 GB | 11.2 GB | 7.5B | LoRA+Q |
| recipes/llama3_2/dpo/1b_qlora_dpo.yaml | 546.9 GB | 3.6 GB | 1.2B | LoRA+Q |
| recipes/llama3_2/sft/1b_full/train.yaml | 82.3 GB | 15.0 GB | 1.2B | FFT |
| recipes/llama3_2/sft/3b_full/fsdp_train.yaml | 483.0 GB | 20.7 GB | 3.2B | FFT |
| recipes/llama3_2/sft/3b_full/train.yaml | 152.4 GB | 38.2 GB | 3.2B | FFT |
| recipes/llama3_2/sft/3b_lora/fsdp_train.yaml | 468.4 GB | 6.2 GB | 3.2B | LoRA |
| recipes/llama3_2/sft/3b_lora/train.yaml | 123.3 GB | 9.1 GB | 3.2B | LoRA |
| recipes/llama3_2/sft/3b_qlora/fsdp_train.yaml | 468.4 GB | 6.2 GB | 3.2B | LoRA+Q |
| recipes/llama3_2/sft/3b_qlora/train.yaml | 123.3 GB | 9.1 GB | 3.2B | LoRA+Q |
| recipes/llama3_3/sft/70b_full/train.yaml | 502.6 GB | 405.9 GB | 69.5B | FFT |
| recipes/llama3_3/sft/70b_lora/train.yaml | 179.2 GB | 82.5 GB | 69.5B | LoRA |
| recipes/llama3_3/sft/70b_qlora/train.yaml | 179.9 GB | 83.2 GB | 69.5B | LoRA+Q |
| recipes/llama4/sft/scout_base_full/train.yaml | 591.7 GB | 576.2 GB | 100.7B | FFT |
| recipes/llama4/sft/scout_instruct_full/train.yaml | 591.7 GB | 576.2 GB | 100.7B | FFT |
| recipes/llama4/sft/scout_instruct_lora/train.yaml | 123.1 GB | 107.5 GB | 100.7B | LoRA |
| recipes/llama4/sft/scout_instruct_qlora/train.yaml | 123.1 GB | 107.5 GB | 100.7B | LoRA+Q |
| recipes/olmo3/sft/7b_full/train.yaml | 84.0 GB | 42.5 GB | 6.9B | FFT |
| recipes/phi3/dpo/macos_train.yaml | 62.9 GB | 9.8 GB | 3.7B | LoRA |
| recipes/phi3/dpo/nvidia_24g_train.yaml | 118.0 GB | 9.7 GB | 3.7B | LoRA |
| recipes/phi3/dpo/nvidia_80g_train.yaml | 228.6 GB | 9.8 GB | 3.7B | LoRA |
| recipes/phi3/dpo/train.yaml | 62.9 GB | 9.8 GB | 3.7B | LoRA |
| recipes/phi3/kto/train.yaml | 62.9 GB | 9.8 GB | 3.7B | LoRA |
| recipes/phi3/sft/lora_macos_train.yaml | 62.9 GB | 9.8 GB | 3.7B | LoRA |
| recipes/phi3/sft/lora_train.yaml | 118.1 GB | 9.8 GB | 3.7B | LoRA |
| recipes/phi4/sft/reasoning_plus/full_train.yaml | 115.4 GB | 85.2 GB | 14.1B | FFT |
| recipes/phi4/sft/reasoning_plus/lora_train.yaml | 49.6 GB | 19.3 GB | 14.1B | LoRA |
| recipes/phi4/sft/reasoning_plus/qlora_train.yaml | 61.5 GB | 31.3 GB | 14.1B | LoRA+Q |
| recipes/qwen2_5/sft/3b_full/train.yaml | 117.5 GB | 19.6 GB | 3.1B | FFT |
| recipes/qwen2_5/sft/7b_full/train.yaml | 176.3 GB | 43.1 GB | 7.1B | FFT |
| recipes/qwen3/sft/0.6b_full/train.yaml | 160.9 GB | 6.8 GB | 0.5B | FFT |
| recipes/qwen3/sft/1.7b_full/train.yaml | 174.4 GB | 20.9 GB | 1.7B | FFT |
| recipes/qwen3/sft/14b_lora/train.yaml | 567.4 GB | 19.2 GB | 14.0B | LoRA |
| recipes/qwen3/sft/235b_lora/train.yaml | 2287.9 GB | 1776.6 GB | 1820.8B | LoRA |
| recipes/qwen3/sft/30b_a3b_lora/train.yaml | 500.5 GB | 237.4 GB | 232.7B | LoRA |
| recipes/qwen3/sft/32b_lora/train.yaml | 914.0 GB | 36.9 GB | 29.7B | LoRA |
| recipes/qwen3/sft/4b_full/train.yaml | 270.0 GB | 23.3 GB | 3.7B | FFT |
| recipes/qwen3/sft/8b_full/train.yaml | 441.4 GB | 46.7 GB | 7.6B | FFT |
| recipes/qwen3_5/sft/0.8b_full/train.yaml | 139.6 GB | 7.5 GB | 0.6B | FFT |
| recipes/qwen3_5/sft/0.8b_lora/train.yaml | 134.2 GB | 2.1 GB | 0.6B | LoRA |
| recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml | 500.5 GB | 237.4 GB | 232.7B | LoRA |
| recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml | 1046.6 GB | 783.4 GB | 773.9B | LoRA |
| recipes/qwq/sft/full_train.yaml | 236.3 GB | 188.0 GB | 32.0B | FFT |
| recipes/qwq/sft/lora_train.yaml | 87.5 GB | 39.2 GB | 32.0B | LoRA |
| recipes/qwq/sft/qlora_train.yaml | 88.0 GB | 39.7 GB | 32.0B | LoRA+Q |
| recipes/smollm/sft/135m/quickstart_train.yaml | 11.0 GB | 2.4 GB | 0.1B | FFT |
| recipes/smollm/sft/135m/train.yaml | 11.0 GB | 2.4 GB | 0.1B | FFT |
| recipes/vision/internvl3/sft/full/train.yaml | 13.3 GB | 6.5 GB | 0.5B | FFT |
| recipes/vision/llama3_2_vision/sft/11b_full/train.yaml | 74.3 GB | 56.5 GB | 9.2B | FFT |
| recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml | 31.3 GB | 13.6 GB | 9.2B | LoRA |
| recipes/vision/llama3_2_vision/sft/90b_full/train.yaml | 594.0 GB | 505.0 GB | 86.6B | FFT |
| recipes/vision/llava_7b/dpo/train.yaml | 60.3 GB | 41.0 GB | 6.6B | FFT |
| recipes/vision/llava_7b/sft/train.yaml | 107.5 GB | 77.1 GB | 6.6B | FFT |
| recipes/vision/molmo/grpo/train.yaml | 293.6 GB | 173.3 GB | 11.2B | FFT |
| recipes/vision/molmo/sft/molmo_d_full/train.yaml | 173.7 GB | 103.6 GB | 12.8B | FFT |
| recipes/vision/molmo/sft/molmo_o_full/train.yaml | 180.7 GB | 92.4 GB | 11.2B | FFT |
| recipes/vision/phi3/dpo/train.yaml | 43.1 GB | 9.8 GB | 3.7B | LoRA |
| recipes/vision/phi3/sft/full/completions_only_train.yaml | 91.8 GB | 44.2 GB | 3.7B | FFT |
| recipes/vision/phi3/sft/full/train.yaml | 91.8 GB | 44.2 GB | 3.7B | FFT |
| recipes/vision/phi3/sft/lora/train.yaml | 32.4 GB | 9.6 GB | 3.7B | LoRA |
| recipes/vision/phi4/sft/full/train.yaml | 76.5 GB | 45.4 GB | 3.8B | FFT |
| recipes/vision/phi4/sft/lora/train.yaml | 40.9 GB | 9.8 GB | 3.8B | LoRA |
| recipes/vision/qwen2_5_vl_3b/dpo/train.yaml | 36342.6 GB | 36.7 GB | 3.1B | FFT |
| recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml | 67.8 GB | 19.6 GB | 3.1B | FFT |
| recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml | 18.8 GB | 7.9 GB | 3.1B | LoRA |
| recipes/vision/qwen2_5_vl_7b/sft/full/train.yaml | 108.6 GB | 43.1 GB | 7.1B | FFT |
| recipes/vision/qwen2_vl_2b/dpo/train.yaml | 1533.3 GB | 18.7 GB | 1.5B | FFT |
| recipes/vision/qwen2_vl_2b/sft/full/train.yaml | 25.0 GB | 18.7 GB | 1.5B | FFT |
| recipes/vision/qwen2_vl_2b/sft/lora/train.yaml | 10.7 GB | 4.4 GB | 1.5B | LoRA |
| recipes/vision/qwen3_vl_2b/sft/2b_instruct_fft_train.yaml | 29.4 GB | 20.9 GB | 1.7B | FFT |
| recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml | 23.1 GB | 5.0 GB | 1.7B | LoRA |
| recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_fft_train.yaml | 1335.5 GB | 1321.0 GB | 232.7B | FFT |
| recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_lora_train.yaml | 251.9 GB | 237.4 GB | 232.7B | LoRA |
| recipes/vision/qwen3_vl_4b/sft/4b_instruct_fft_train.yaml | 57.0 GB | 43.4 GB | 3.7B | FFT |
| recipes/vision/qwen3_vl_4b/sft/4b_instruct_lora_train.yaml | 23.0 GB | 9.4 GB | 3.7B | LoRA |
| recipes/vision/smolvlm/sft/full/train.yaml | 25.8 GB | 20.7 GB | 1.7B | FFT |
| recipes/vision/smolvlm/sft/lora/train.yaml | 10.0 GB | 4.8 GB | 1.7B | LoRA |

## Dry-Run Results (0 passed, 150 failed)

- examples/deepspeed/llama3_1_8b_deepspeed_z2_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/deepspeed/llama3_1_8b_deepspeed_z3_offload_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/deepspeed/llama3_1_8b_deepspeed_z3_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/fineweb_ablation_pretraining/ddp/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/fineweb_ablation_pretraining/fsdp/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/gkd/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/gold/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/gold/train_gptoss120b_qwen06b.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/gold/train_gptoss120b_qwen8b.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/grpo_tldr/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/grpo_verl_countdown/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/grpo_verl_geometry3k/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/grpo_verl_gsm8k/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/letter_counting/grpo/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- examples/misc/tulu3_sft_mini.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/aya/sft/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/chatqa/chatqa_stage1_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/chatqa/chatqa_stage2_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/coalm/405b_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/coalm/70b_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/coalm/8b_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/dcvlr/starter_kit/molmo-o-train-openr1.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/dcvlr/starter_kit/qwenvl-openr1.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/halloumi/8b_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/limo/qwen2.5_7b_fft.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/limo/qwen2.5_7b_fft_yarn.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/limo/qwen2.5_7b_fft_yarn_deepspeed.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/limo/qwen2.5_7b_fft_yarn_deepspeed_memory_optimized_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- projects/wc50m/configs/base_ultrachat.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_e/dpo/falcon_e_1b_instruct/dpo.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_e/sft/falcon_e_1b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_e/sft/falcon_e_1b_instruct/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_e/sft/falcon_e_3b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_e/sft/falcon_e_3b_instruct/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_h1/sft/falcon_h1_0_5b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_h1/sft/falcon_h1_1_5b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_h1/sft/falcon_h1_34b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_h1/sft/falcon_h1_3b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/gemma2/sft/2b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/gemma3/sft/12b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/gemma3/sft/27b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/gemma3/sft/4b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/gpt2/pretraining/macos_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/gpt2/pretraining/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/gpt_oss/sft/120b_lora_multi_gpu_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/gpt_oss/sft/20b_lora_multi_gpu_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/gpt_oss/sft/20b_lora_single_gpu_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/pretraining/8b/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/405b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/405b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/405b_qlora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/70b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/70b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/70b_qlora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/8b_full/accelerate.yaml: **FAILED** — No model_name
- recipes/llama3_1/sft/8b_full/longctx_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/8b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/8b_lora/fsdp_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/8b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_1/sft/8b_qlora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_2/dpo/1b_qlora_dpo.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_2/sft/1b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_2/sft/3b_full/fsdp_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_2/sft/3b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_2/sft/3b_lora/fsdp_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_2/sft/3b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_2/sft/3b_qlora/fsdp_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_2/sft/3b_qlora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_3/sft/70b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_3/sft/70b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama3_3/sft/70b_qlora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama4/sft/scout_base_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama4/sft/scout_instruct_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama4/sft/scout_instruct_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/llama4/sft/scout_instruct_qlora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/olmo3/sft/32b_lora/train.yaml: **FAILED** — OSError: allenai/Olmo-3-32B-Instruct is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
- recipes/olmo3/sft/7b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi3/dpo/macos_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi3/dpo/nvidia_24g_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi3/dpo/nvidia_80g_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi3/dpo/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi3/kto/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi3/sft/lora_macos_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi3/sft/lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi4/sft/reasoning_plus/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi4/sft/reasoning_plus/lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/phi4/sft/reasoning_plus/qlora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen2_5/sft/3b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen2_5/sft/7b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3/sft/0.6b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3/sft/1.7b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3/sft/14b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3/sft/235b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3/sft/30b_a3b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3/sft/32b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3/sft/4b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3/sft/8b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3_5/sft/0.8b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3_5/sft/0.8b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwen3_next/sft/80b_a3b_lora/train.yaml: **FAILED** — OSError: Qwen/Qwen3-Next-80B-A3B is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
- recipes/qwq/sft/full_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwq/sft/lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/qwq/sft/qlora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/smollm/sft/135m/quickstart_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/smollm/sft/135m/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/internvl3/sft/full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/llama3_2_vision/sft/11b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/llama3_2_vision/sft/90b_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/llava_7b/dpo/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/llava_7b/sft/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/molmo/grpo/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/molmo/sft/molmo_d_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/molmo/sft/molmo_o_full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/phi3/dpo/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/phi3/sft/full/completions_only_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/phi3/sft/full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/phi3/sft/lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/phi4/sft/full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/phi4/sft/lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen2_5_vl_3b/dpo/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen2_5_vl_7b/sft/full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen2_vl_2b/dpo/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen2_vl_2b/sft/full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen2_vl_2b/sft/lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen3_vl_2b/sft/2b_instruct_fft_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_fft_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen3_vl_4b/sft/4b_instruct_fft_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/qwen3_vl_4b/sft/4b_instruct_lora_train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/smolvlm/sft/full/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'
- recipes/vision/smolvlm/sft/lora/train.yaml: **FAILED** — TypeError: _LazyAutoMapping.get() missing 1 required positional argument: 'default'

## Optimization Suggestions (225)

- performance: 183
- efficiency: 37
- best_practice: 5

## Environment

- available_ram_gb: 57.7
- bitsandbytes_version: 0.49.2
- cgroup_memory_limit_gb: 57.7
- cuda_available: True
- cuda_capability: 8.9
- cuda_device: NVIDIA L40S
- cuda_device_count: 1
- cuda_total_memory_gb: 44.4
- cuda_version: 12.8
- cudnn_enabled: True
- cudnn_version: 91002
- deepspeed_version: 0.18.8
- flash_attn_version: 2.8.3
- git_commit: 188b7b33
- nccl_version: 2.27.5
- nvidia_driver: 570.195.03
- oumi_version: ?
- peft_version: 0.18.1
- platform: Linux-6.8.0-45-generic-x86_64-with-glibc2.35
- python: 3.11.15
- torch_version: 2.10.0+cu128
- transformers_version: 5.3.0
- trl_version: 0.28.0

## Phase Timing

- classify: 13.17s
- static: 32.87s
- hub: 0.41s
- tier0: 136.44s
- vram: 13.29s
- dry_run: 69.61s
