# Config Health Summary

Scanned **495** configs in 3829.2s.

## Results

| Status | Count |
|--------|-------|
| Healthy | 443 |
| Warnings | 52 |
| Failing | 0 |
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

## Warnings (52)

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
- **configs/recipes/gpt_oss/sft/120b_lora_multi_gpu_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/gpt_oss/sft/20b_lora_multi_gpu_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_1/sft/405b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_1/sft/8b_qlora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/olmo3/sft/32b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/phi3/dpo/macos_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/dpo/nvidia_24g_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/dpo/nvidia_80g_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/dpo/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/kto/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/sft/lora_macos_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/sft/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi4/sft/reasoning_plus/lora_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/14b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/235b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/32b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3_next/sft/80b_a3b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwq/sft/lora_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/smollm/sft/135m/quickstart_train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/smollm/sft/135m/train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/vision/molmo/sft/molmo_d_full/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/dpo/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/sft/full/completions_only_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/sft/full/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/sft/lora/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi4/sft/full/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
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

## VRAM Estimates (149 training configs)

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
| recipes/olmo3/sft/32b_lora/train.yaml | 142.5 GB | 38.9 GB | 31.7B | LoRA |
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
| recipes/qwen3_next/sft/80b_a3b_lora/train.yaml | 1046.6 GB | 783.4 GB | 773.9B | LoRA |
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

## Dry-Run Results (63 passed, 87 failed)

- examples/deepspeed/llama3_1_8b_deepspeed_z2_train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 44.0 GB free (7.5B params)
- examples/deepspeed/llama3_1_8b_deepspeed_z3_offload_train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 44.0 GB free (7.5B params)
- examples/deepspeed/llama3_1_8b_deepspeed_z3_train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 44.0 GB free (7.5B params)
- examples/fineweb_ablation_pretraining/ddp/train.yaml: passed in 25.4s (16.0 GB)
- examples/fineweb_ablation_pretraining/fsdp/train.yaml: passed in 23.3s (16.0 GB)
- examples/gkd/train.yaml: passed in 2.4s (1.3 GB)
- examples/gold/train.yaml: passed in 2.5s (1.3 GB)
- examples/gold/train_gptoss120b_qwen06b.yaml: passed in 8.7s (5.6 GB)
- examples/gold/train_gptoss120b_qwen8b.yaml: **FAILED** — Skipped: model needs ~84.6 GB GPU but only 43.7 GB free (7.6B params)
- examples/grpo_tldr/train.yaml: passed in 7.3s (4.7 GB)
- examples/grpo_verl_countdown/train.yaml: passed in 45.3s (30.0 GB)
- examples/grpo_verl_geometry3k/train.yaml: passed in 117.0s (35.6 GB)
- examples/grpo_verl_gsm8k/train.yaml: passed in 7.7s (4.7 GB)
- examples/letter_counting/grpo/train.yaml: passed in 43.0s (30.0 GB)
- examples/misc/tulu3_sft_mini.yaml: passed in 1.3s (0.2 GB)
- projects/aya/sft/train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 43.8 GB free (7.5B params)
- projects/chatqa/chatqa_stage1_train.yaml: **FAILED** — Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
- projects/chatqa/chatqa_stage2_train.yaml: **FAILED** — Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
- projects/coalm/405b_train.yaml: **FAILED** — Skipped: model needs ~1504 GB CPU RAM but only 58 GB available (403.8B params)
- projects/coalm/70b_train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- projects/coalm/8b_train.yaml: passed in 94.9s (16.8 GB)
- projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml: **FAILED** — Skipped: model needs ~95 GB CPU RAM but only 58 GB available (12.8B params)
- projects/dcvlr/starter_kit/molmo-o-train-openr1.yaml: **FAILED** — Skipped: model needs ~84 GB CPU RAM but only 58 GB available (11.2B params)
- projects/dcvlr/starter_kit/qwenvl-openr1.yaml: **FAILED** — Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
- projects/halloumi/8b_train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
- projects/limo/qwen2.5_7b_fft.yaml: **FAILED** — Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
- projects/limo/qwen2.5_7b_fft_yarn.yaml: **FAILED** — Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
- projects/limo/qwen2.5_7b_fft_yarn_deepspeed.yaml: **FAILED** — Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
- projects/limo/qwen2.5_7b_fft_yarn_deepspeed_memory_optimized_train.yaml: **FAILED** — Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
- projects/wc50m/configs/base_ultrachat.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
- recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
- recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml: passed in 91.4s (15.9 GB)
- recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml: passed in 93.9s (16.6 GB)
- recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml: passed in 23.6s (16.6 GB)
- recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml: passed in 25.8s (4.9 GB)
- recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml: **FAILED** — Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
- recipes/falcon_e/dpo/falcon_e_1b_instruct/dpo.yaml: passed in 20.1s (16.0 GB)
- recipes/falcon_e/sft/falcon_e_1b/full_train.yaml: passed in 22.1s (16.0 GB)
- recipes/falcon_e/sft/falcon_e_1b_instruct/full_train.yaml: passed in 20.6s (16.0 GB)
- recipes/falcon_e/sft/falcon_e_3b/full_train.yaml: passed in 38.8s (28.5 GB)
- recipes/falcon_e/sft/falcon_e_3b_instruct/full_train.yaml: passed in 36.2s (28.5 GB)
- recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_0_5b/full_train.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_1_5b/full_train.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_train.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_34b/full_train.yaml: **FAILED** — Skipped: model needs ~110 GB CPU RAM but only 58 GB available (29.6B params)
- recipes/falcon_h1/sft/falcon_h1_3b/full_train.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml: **FAILED** — Skipped: model needs ~71.0 GB GPU but only 43.8 GB free (6.4B params)
- recipes/gemma2/sft/2b_full/train.yaml: passed in 33.5s (24.4 GB)
- recipes/gemma3/sft/12b_lora/train.yaml: passed in 371.0s (31.3 GB)
- recipes/gemma3/sft/27b_lora/train.yaml: **FAILED** — Skipped: model needs ~105 GB CPU RAM but only 58 GB available (28.3B params)
- recipes/gemma3/sft/4b_full/train.yaml: **FAILED** — Skipped: model needs ~44.9 GB GPU but only 43.9 GB free (4.0B params)
- recipes/gpt2/pretraining/macos_train.yaml: passed in 2.6s (1.2 GB)
- recipes/gpt2/pretraining/train.yaml: passed in 2.5s (1.2 GB)
- recipes/gpt_oss/sft/120b_lora_multi_gpu_train.yaml: **FAILED** — Skipped: model needs ~432 GB CPU RAM but only 58 GB available (115.9B params)
- recipes/gpt_oss/sft/20b_lora_multi_gpu_train.yaml: **FAILED** — Skipped: model needs ~75 GB CPU RAM but only 58 GB available (20.1B params)
- recipes/gpt_oss/sft/20b_lora_single_gpu_train.yaml: **FAILED** — Skipped: model needs ~75 GB CPU RAM but only 58 GB available (20.1B params)
- recipes/llama3_1/pretraining/8b/train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
- recipes/llama3_1/sft/405b_full/train.yaml: **FAILED** — Skipped: model needs ~1504 GB CPU RAM but only 58 GB available (403.8B params)
- recipes/llama3_1/sft/405b_lora/train.yaml: **FAILED** — Skipped: model needs ~1504 GB CPU RAM but only 58 GB available (403.8B params)
- recipes/llama3_1/sft/405b_qlora/train.yaml: **FAILED** — Skipped: model needs ~1504 GB CPU RAM but only 58 GB available (403.8B params)
- recipes/llama3_1/sft/70b_full/train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/llama3_1/sft/70b_lora/train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/llama3_1/sft/70b_qlora/train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/llama3_1/sft/8b_full/accelerate.yaml: **FAILED** — No model_name
- recipes/llama3_1/sft/8b_full/longctx_train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
- recipes/llama3_1/sft/8b_full/train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
- recipes/llama3_1/sft/8b_lora/fsdp_train.yaml: passed in 90.1s (15.9 GB)
- recipes/llama3_1/sft/8b_lora/train.yaml: passed in 90.1s (15.9 GB)
- recipes/llama3_1/sft/8b_qlora/train.yaml: passed in 92.2s (16.6 GB)
- recipes/llama3_2/dpo/1b_qlora_dpo.yaml: passed in 15.3s (3.0 GB)
- recipes/llama3_2/sft/1b_full/train.yaml: passed in 15.1s (11.6 GB)
- recipes/llama3_2/sft/3b_full/fsdp_train.yaml: passed in 41.0s (30.0 GB)
- recipes/llama3_2/sft/3b_full/train.yaml: passed in 37.3s (30.0 GB)
- recipes/llama3_2/sft/3b_lora/fsdp_train.yaml: passed in 43.6s (7.9 GB)
- recipes/llama3_2/sft/3b_lora/train.yaml: passed in 44.5s (7.9 GB)
- recipes/llama3_2/sft/3b_qlora/fsdp_train.yaml: passed in 42.8s (7.9 GB)
- recipes/llama3_2/sft/3b_qlora/train.yaml: passed in 41.8s (7.9 GB)
- recipes/llama3_3/sft/70b_full/train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/llama3_3/sft/70b_lora/train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/llama3_3/sft/70b_qlora/train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/llama4/sft/scout_base_full/train.yaml: **FAILED** — Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
- recipes/llama4/sft/scout_instruct_full/train.yaml: **FAILED** — Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
- recipes/llama4/sft/scout_instruct_lora/train.yaml: **FAILED** — Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
- recipes/llama4/sft/scout_instruct_qlora/train.yaml: **FAILED** — Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
- recipes/olmo3/sft/32b_lora/train.yaml: **FAILED** — Skipped: model needs ~118 GB CPU RAM but only 58 GB available (31.7B params)
- recipes/olmo3/sft/7b_full/train.yaml: **FAILED** — Skipped: model needs ~77.0 GB GPU but only 43.9 GB free (6.9B params)
- recipes/phi3/dpo/macos_train.yaml: passed in 41.1s (8.0 GB)
- recipes/phi3/dpo/nvidia_24g_train.yaml: passed in 38.2s (7.7 GB)
- recipes/phi3/dpo/nvidia_80g_train.yaml: passed in 39.2s (8.0 GB)
- recipes/phi3/dpo/train.yaml: passed in 39.7s (8.0 GB)
- recipes/phi3/kto/train.yaml: passed in 37.7s (8.0 GB)
- recipes/phi3/sft/lora_macos_train.yaml: passed in 38.4s (8.0 GB)
- recipes/phi3/sft/lora_train.yaml: passed in 37.6s (8.0 GB)
- recipes/phi4/sft/reasoning_plus/full_train.yaml: **FAILED** — Skipped: model needs ~53 GB CPU RAM but only 58 GB available (14.1B params)
- recipes/phi4/sft/reasoning_plus/lora_train.yaml: **FAILED** — Skipped: model needs ~53 GB CPU RAM but only 58 GB available (14.1B params)
- recipes/phi4/sft/reasoning_plus/qlora_train.yaml: **FAILED** — Skipped: model needs ~53 GB CPU RAM but only 58 GB available (14.1B params)
- recipes/qwen2_5/sft/3b_full/train.yaml: passed in 33.5s (29.3 GB)
- recipes/qwen2_5/sft/7b_full/train.yaml: **FAILED** — Skipped: model needs ~79.0 GB GPU but only 43.8 GB free (7.1B params)
- recipes/qwen3/sft/0.6b_full/train.yaml: passed in 7.5s (5.6 GB)
- recipes/qwen3/sft/1.7b_full/train.yaml: passed in 18.7s (16.1 GB)
- recipes/qwen3/sft/14b_lora/train.yaml: **FAILED** — Skipped: model needs ~52 GB CPU RAM but only 58 GB available (14.0B params)
- recipes/qwen3/sft/235b_lora/train.yaml: **FAILED** — Skipped: model needs ~6783 GB CPU RAM but only 58 GB available (1820.8B params)
- recipes/qwen3/sft/30b_a3b_lora/train.yaml: **FAILED** — Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
- recipes/qwen3/sft/32b_lora/train.yaml: **FAILED** — Skipped: model needs ~111 GB CPU RAM but only 58 GB available (29.7B params)
- recipes/qwen3/sft/4b_full/train.yaml: **FAILED** — Skipped: model needs ~41.0 GB GPU but only 43.9 GB free (3.7B params)
- recipes/qwen3/sft/8b_full/train.yaml: **FAILED** — Skipped: model needs ~84.6 GB GPU but only 43.9 GB free (7.6B params)
- recipes/qwen3_5/sft/0.8b_full/train.yaml: passed in 37.2s (8.2 GB)
- recipes/qwen3_5/sft/0.8b_lora/train.yaml: passed in 37.1s (2.4 GB)
- recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml: **FAILED** — Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
- recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml: **FAILED** — Skipped: model needs ~2883 GB CPU RAM but only 58 GB available (773.9B params)
- recipes/qwen3_next/sft/80b_a3b_lora/train.yaml: **FAILED** — Skipped: model needs ~2883 GB CPU RAM but only 58 GB available (773.9B params)
- recipes/qwq/sft/full_train.yaml: **FAILED** — Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
- recipes/qwq/sft/lora_train.yaml: **FAILED** — Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
- recipes/qwq/sft/qlora_train.yaml: **FAILED** — Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
- recipes/smollm/sft/135m/quickstart_train.yaml: passed in 2.5s (1.3 GB)
- recipes/smollm/sft/135m/train.yaml: passed in 2.7s (1.3 GB)
- recipes/vision/internvl3/sft/full/train.yaml: passed in 28.5s (7.6 GB)
- recipes/vision/llama3_2_vision/sft/11b_full/train.yaml: **FAILED** — Skipped: model needs ~103.4 GB GPU but only 43.6 GB free (9.2B params)
- recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml: passed in 294.3s (36.3 GB)
- recipes/vision/llama3_2_vision/sft/90b_full/train.yaml: **FAILED** — Skipped: model needs ~323 GB CPU RAM but only 58 GB available (86.6B params)
- recipes/vision/llava_7b/dpo/train.yaml: **FAILED** — Skipped: model needs ~73.8 GB GPU but only 43.8 GB free (6.6B params)
- recipes/vision/llava_7b/sft/train.yaml: **FAILED** — Skipped: model needs ~73.8 GB GPU but only 43.8 GB free (6.6B params)
- recipes/vision/molmo/grpo/train.yaml: **FAILED** — Skipped: model needs ~84 GB CPU RAM but only 58 GB available (11.2B params)
- recipes/vision/molmo/sft/molmo_d_full/train.yaml: **FAILED** — Skipped: model needs ~95 GB CPU RAM but only 58 GB available (12.8B params)
- recipes/vision/molmo/sft/molmo_o_full/train.yaml: **FAILED** — Skipped: model needs ~84 GB CPU RAM but only 58 GB available (11.2B params)
- recipes/vision/phi3/dpo/train.yaml: **FAILED** — ImportError: cannot import name 'is_flash_attn_greater_or_equal_2_10' from 'transformers.utils' (/root/miniconda3/envs/oumi/lib/python3.11/site-packages/transformers/utils/__init__.py)
- recipes/vision/phi3/sft/full/completions_only_train.yaml: **FAILED** — Skipped: model needs ~41.6 GB GPU but only 43.8 GB free (3.7B params)
- recipes/vision/phi3/sft/full/train.yaml: **FAILED** — Skipped: model needs ~41.6 GB GPU but only 43.8 GB free (3.7B params)
- recipes/vision/phi3/sft/lora/train.yaml: **FAILED** — ImportError: cannot import name 'is_flash_attn_greater_or_equal_2_10' from 'transformers.utils' (/root/miniconda3/envs/oumi/lib/python3.11/site-packages/transformers/utils/__init__.py)
- recipes/vision/phi4/sft/full/train.yaml: **FAILED** — Skipped: model needs ~42.9 GB GPU but only 43.8 GB free (3.8B params)
- recipes/vision/phi4/sft/lora/train.yaml: **FAILED** — ImportError: cannot import name 'SlidingWindowCache' from 'transformers.cache_utils' (/root/miniconda3/envs/oumi/lib/python3.11/site-packages/transformers/cache_utils.py)
- recipes/vision/qwen2_5_vl_3b/dpo/train.yaml: passed in 117.5s (35.6 GB)
- recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml: passed in 115.6s (35.6 GB)
- recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml: passed in 118.0s (8.7 GB)
- recipes/vision/qwen2_5_vl_7b/sft/full/train.yaml: **FAILED** — Skipped: model needs ~79.0 GB GPU but only 43.6 GB free (7.1B params)
- recipes/vision/qwen2_vl_2b/dpo/train.yaml: passed in 69.2s (20.6 GB)
- recipes/vision/qwen2_vl_2b/sft/full/train.yaml: passed in 69.8s (20.6 GB)
- recipes/vision/qwen2_vl_2b/sft/lora/train.yaml: passed in 68.4s (4.8 GB)
- recipes/vision/qwen3_vl_2b/sft/2b_instruct_fft_train.yaml: passed in 69.1s (19.9 GB)
- recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml: passed in 69.4s (4.5 GB)
- recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_fft_train.yaml: **FAILED** — Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
- recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_lora_train.yaml: **FAILED** — Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
- recipes/vision/qwen3_vl_4b/sft/4b_instruct_fft_train.yaml: **FAILED** — Skipped: model needs ~41.0 GB GPU but only 43.8 GB free (3.7B params)
- recipes/vision/qwen3_vl_4b/sft/4b_instruct_lora_train.yaml: passed in 134.1s (9.3 GB)
- recipes/vision/smolvlm/sft/full/train.yaml: passed in 58.7s (29.2 GB)
- recipes/vision/smolvlm/sft/lora/train.yaml: passed in 63.2s (24.4 GB)

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
- transformers_version: 5.4.0
- trl_version: 0.28.0

## Phase Timing

- classify: 12.44s
- static: 44.51s
- hub: 0.07s
- tier0: 148.75s
- vram: 18.81s
- dry_run: 3536.32s
