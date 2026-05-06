# Config Health Summary

Scanned **495** configs in 2765.1s.

## Results

| Status | Count |
|--------|-------|
| Healthy | 230 |
| Warnings | 152 |
| Failing | 113 |
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

## Failures (136)

- **configs/examples/deploy/fireworks_deploy.yaml**: Could not parse with any oumi config class
- **configs/recipes/llama3_1/sft/8b_full/accelerate.yaml**: Could not parse with any oumi config class
- **configs/projects/coalm/405b_train.yaml**: Dataset not found on HF Hub: text_sft_jsonl
- **configs/projects/coalm/70b_train.yaml**: Dataset not found on HF Hub: text_sft_jsonl
- **configs/projects/coalm/8b_train.yaml**: Dataset not found on HF Hub: text_sft_jsonl
- **configs/projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml**: Dataset not found on HF Hub: hf_vision
- **configs/projects/dcvlr/starter_kit/molmo-o-train-openr1.yaml**: Dataset not found on HF Hub: hf_vision
- **configs/projects/dcvlr/starter_kit/qwenvl-openr1.yaml**: Dataset not found on HF Hub: hf_vision
- **configs/projects/halloumi/8b_train.yaml**: Dataset not found on HF Hub: HuggingFaceDataset
- **configs/projects/halloumi/8b_train.yaml**: Dataset not found on HF Hub: HuggingFaceDataset
- **configs/projects/halloumi/8b_train.yaml**: Dataset not found on HF Hub: HuggingFaceDataset
- **configs/projects/halloumi/8b_train.yaml**: Dataset not found on HF Hub: HuggingFaceDataset
- **configs/projects/limo/qwen2.5_7b_fft.yaml**: Dataset not found on HF Hub: PromptResponseDataset
- **configs/projects/limo/qwen2.5_7b_fft_yarn.yaml**: Dataset not found on HF Hub: PromptResponseDataset
- **configs/projects/limo/qwen2.5_7b_fft_yarn_deepspeed.yaml**: Dataset not found on HF Hub: PromptResponseDataset
- **configs/projects/limo/qwen2.5_7b_fft_yarn_deepspeed_memory_optimized_train.yaml**: Dataset not found on HF Hub: PromptResponseDataset
- **configs/recipes/falcon_e/evaluation/falcon_e_1b/eval.yaml**: Model not found on HF Hub: output/falcon_e_1b.fft/checkpoint-800-quantized
- **configs/recipes/falcon_e/evaluation/falcon_e_1b_instruct/eval.yaml**: Model not found on HF Hub: output/falcon_e_1b_instruct.fft/checkpoint-800-quantized
- **configs/recipes/falcon_e/evaluation/falcon_e_3b/eval.yaml**: Model not found on HF Hub: output/falcon_e_3b.fft/checkpoint-800-quantized
- **configs/recipes/falcon_e/evaluation/falcon_e_3b_instruct/eval.yaml**: Model not found on HF Hub: output/falcon_e_3b_instruct.fft/checkpoint-800-quantized
- **configs/recipes/olmo3/evaluation/32b/eval.yaml**: Model not found on HF Hub: allenai/Olmo-3-32B-Instruct
- **configs/recipes/olmo3/inference/32b_infer.yaml**: Model not found on HF Hub: allenai/Olmo-3-32B-Instruct
- **configs/recipes/olmo3/sft/32b_lora/train.yaml**: Model not found on HF Hub: allenai/Olmo-3-32B-Instruct
- **configs/recipes/qwen3/inference/30b_a3b_instruct_vllm_infer.yaml**: Model not found on HF Hub: Qwen/Qwen3-30B-A3B-Instruct
- **configs/recipes/qwen3_next/evaluation/80b_a3b_eval.yaml**: Model not found on HF Hub: Qwen/Qwen3-Next-80B-A3B
- **configs/recipes/qwen3_next/inference/80b_a3b_infer.yaml**: Model not found on HF Hub: Qwen/Qwen3-Next-80B-A3B
- **configs/recipes/qwen3_next/sft/80b_a3b_lora/train.yaml**: Model not found on HF Hub: Qwen/Qwen3-Next-80B-A3B
- **configs/recipes/vision/llava_7b/dpo/train.yaml**: Dataset not found on HF Hub: vision_dpo_jsonl
- **configs/recipes/vision/phi3/dpo/train.yaml**: Dataset not found on HF Hub: vision_dpo_jsonl
- **configs/examples/misc/tulu3_sft_mini.yaml**: model_max_length (4096) exceeds model's max_position_embeddings (2048). Training will produce garbage or fail with RoPE extrapolation errors.
- **configs/projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml**: model_max_length (8192) exceeds model's max_position_embeddings (4096). Training will produce garbage or fail with RoPE extrapolation errors.
- **configs/projects/dcvlr/starter_kit/molmo-o-train-openr1.yaml**: model_max_length (8192) exceeds model's max_position_embeddings (4096). Training will produce garbage or fail with RoPE extrapolation errors.
- **configs/recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml**: model_max_length (16385) exceeds model's max_position_embeddings (16384). Training will produce garbage or fail with RoPE extrapolation errors.
- **configs/recipes/phi3/dpo/train.yaml**: LoRA target modules not found in model: ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']
  - Available: ['', '0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '4', '5', '6']
- **configs/recipes/phi3/kto/train.yaml**: LoRA target modules not found in model: ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']
  - Available: ['', '0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '4', '5', '6']
- **configs/recipes/phi3/sft/lora_macos_train.yaml**: LoRA target modules not found in model: ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']
  - Available: ['', '0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '4', '5', '6']
- **configs/recipes/phi3/sft/lora_train.yaml**: LoRA target modules not found in model: ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']
  - Available: ['', '0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '4', '5', '6']
- **configs/recipes/phi4/sft/reasoning_plus/qlora_train.yaml**: LoRA target modules not found in model: ['gate_proj', 'up_proj']
  - Available: ['', '0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34']
- **configs/recipes/vision/phi3/dpo/train.yaml**: LoRA target modules not found in model: ['gate_proj', 'up_proj']
  - Available: ['', '0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '4', '5', '6']
- **configs/recipes/vision/phi3/sft/lora/train.yaml**: LoRA target modules not found in model: ['gate_proj', 'up_proj']
  - Available: ['', '0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '4', '5', '6']
- **configs/examples/deepspeed/llama3_1_8b_deepspeed_z2_train.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 44.0 GB free (7.5B params)
  - model: meta-llama/Llama-3.1-8B-Instruct
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/examples/deepspeed/llama3_1_8b_deepspeed_z3_offload_train.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 44.0 GB free (7.5B params)
  - model: meta-llama/Llama-3.1-8B-Instruct
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/examples/deepspeed/llama3_1_8b_deepspeed_z3_train.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 44.0 GB free (7.5B params)
  - model: meta-llama/Llama-3.1-8B-Instruct
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/examples/gold/train_gptoss120b_qwen8b.yaml**: Dry-run failed: Skipped: model needs ~84.6 GB GPU but only 43.7 GB free (7.6B params)
  - model: Qwen/Qwen3-8B
estimated: 7.6B params, 14.1 GB weights, 28.2 GB CPU RAM needed
- **configs/examples/grpo_verl_geometry3k/train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerati
  - model: Qwen/Qwen2.5-VL-3B-Instruct
estimated: 3.1B params, 5.7 GB weights, 11.5 GB CPU RAM needed
- **configs/projects/aya/sft/train.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 43.8 GB free (7.5B params)
  - model: meta-llama/Llama-3.1-8B-Instruct
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/projects/chatqa/chatqa_stage1_train.yaml**: Dry-run failed: Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
  - model: microsoft/Phi-3-mini-4k-instruct
estimated: 3.7B params, 6.9 GB weights, 13.9 GB CPU RAM needed
- **configs/projects/chatqa/chatqa_stage2_train.yaml**: Dry-run failed: Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
  - model: microsoft/Phi-3-mini-4k-instruct
estimated: 3.7B params, 6.9 GB weights, 13.9 GB CPU RAM needed
- **configs/projects/coalm/405b_train.yaml**: Dry-run failed: Skipped: model needs ~1504 GB CPU RAM but only 58 GB available (403.8B params)
  - model: meta-llama/Llama-3.1-405B-Instruct
estimated: 403.8B params, 752.0 GB weights, 1504.1 GB CPU RAM needed
- **configs/projects/coalm/70b_train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: meta-llama/Llama-3.3-70B-Instruct
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml**: Dry-run failed: Skipped: model needs ~95 GB CPU RAM but only 58 GB available (12.8B params)
  - model: oumi-ai/Molmo-7B-D-0924
estimated: 12.8B params, 47.6 GB weights, 95.2 GB CPU RAM needed
- **configs/projects/dcvlr/starter_kit/molmo-o-train-openr1.yaml**: Dry-run failed: Skipped: model needs ~84 GB CPU RAM but only 58 GB available (11.2B params)
  - model: oumi-ai/Molmo-7B-O-0924
estimated: 11.2B params, 41.8 GB weights, 83.6 GB CPU RAM needed
- **configs/projects/dcvlr/starter_kit/qwenvl-openr1.yaml**: Dry-run failed: Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
  - model: Qwen/Qwen2.5-VL-7B-Instruct
estimated: 7.1B params, 13.2 GB weights, 26.3 GB CPU RAM needed
- **configs/projects/halloumi/8b_train.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
  - model: meta-llama/Llama-3.1-8B-Instruct
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/projects/limo/qwen2.5_7b_fft.yaml**: Dry-run failed: Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
  - model: Qwen/Qwen2.5-7B-Instruct
estimated: 7.1B params, 13.2 GB weights, 26.3 GB CPU RAM needed
- **configs/projects/limo/qwen2.5_7b_fft_yarn.yaml**: Dry-run failed: Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
  - model: Qwen/Qwen2.5-7B-Instruct
estimated: 7.1B params, 13.2 GB weights, 26.3 GB CPU RAM needed
- **configs/projects/limo/qwen2.5_7b_fft_yarn_deepspeed.yaml**: Dry-run failed: Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
  - model: Qwen/Qwen2.5-7B-Instruct
estimated: 7.1B params, 13.2 GB weights, 26.3 GB CPU RAM needed
- **configs/projects/limo/qwen2.5_7b_fft_yarn_deepspeed_memory_optimized_train.yaml**: Dry-run failed: Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
  - model: Qwen/Qwen2.5-7B-Instruct
estimated: 7.1B params, 13.2 GB weights, 26.3 GB CPU RAM needed
- **configs/projects/wc50m/configs/base_ultrachat.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
  - model: meta-llama/Meta-Llama-3.1-8B
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
  - model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml**: Dry-run failed: Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
  - model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
estimated: 32.0B params, 59.6 GB weights, 119.2 GB CPU RAM needed
- **configs/recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml**: Dry-run failed: AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
  - model: tiiuae/Falcon-H1-0.5B-Instruct
estimated: 0.4B params, 0.7 GB weights, 1.3 GB CPU RAM needed
params: 0.5B
LoRA trainable: 8.4M
device: cuda
- **configs/recipes/falcon_h1/sft/falcon_h1_0_5b/full_train.yaml**: Dry-run failed: AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
  - model: tiiuae/Falcon-H1-0.5B-Instruct
estimated: 0.4B params, 0.7 GB weights, 1.3 GB CPU RAM needed
params: 0.5B
device: cuda
- **configs/recipes/falcon_h1/sft/falcon_h1_1_5b/full_train.yaml**: Dry-run failed: AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
  - model: tiiuae/Falcon-H1-1.5B-Instruct
estimated: 1.1B params, 2.0 GB weights, 4.0 GB CPU RAM needed
params: 1.6B
device: cuda
- **configs/recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_train.yaml**: Dry-run failed: AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
  - model: tiiuae/Falcon-H1-1.5B-Deep-Instruct
estimated: 1.2B params, 2.1 GB weights, 4.3 GB CPU RAM needed
params: 1.6B
device: cuda
- **configs/recipes/falcon_h1/sft/falcon_h1_34b/full_train.yaml**: Dry-run failed: Skipped: model needs ~110 GB CPU RAM but only 58 GB available (29.6B params)
  - model: tiiuae/Falcon-H1-34B-Instruct
estimated: 29.6B params, 55.2 GB weights, 110.5 GB CPU RAM needed
- **configs/recipes/falcon_h1/sft/falcon_h1_3b/full_train.yaml**: Dry-run failed: AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
  - model: tiiuae/Falcon-H1-3B-Instruct
estimated: 2.2B params, 4.1 GB weights, 8.1 GB CPU RAM needed
params: 3.1B
device: cuda
- **configs/recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml**: Dry-run failed: Skipped: model needs ~71.0 GB GPU but only 43.8 GB free (6.4B params)
  - model: tiiuae/Falcon-H1-7B-Instruct
estimated: 6.4B params, 11.8 GB weights, 23.7 GB CPU RAM needed
- **configs/recipes/gemma3/sft/27b_lora/train.yaml**: Dry-run failed: Skipped: model needs ~105 GB CPU RAM but only 58 GB available (28.3B params)
  - model: google/gemma-3-27b-it
estimated: 28.3B params, 52.7 GB weights, 105.4 GB CPU RAM needed
- **configs/recipes/gemma3/sft/4b_full/train.yaml**: Dry-run failed: Skipped: model needs ~44.9 GB GPU but only 43.9 GB free (4.0B params)
  - model: google/gemma-3-4b-it
estimated: 4.0B params, 7.5 GB weights, 15.0 GB CPU RAM needed
- **configs/recipes/gpt_oss/sft/120b_lora_multi_gpu_train.yaml**: Dry-run failed: Skipped: model needs ~432 GB CPU RAM but only 58 GB available (115.9B params)
  - model: openai/gpt-oss-120b
estimated: 115.9B params, 215.9 GB weights, 431.9 GB CPU RAM needed
- **configs/recipes/gpt_oss/sft/20b_lora_multi_gpu_train.yaml**: Dry-run failed: Skipped: model needs ~75 GB CPU RAM but only 58 GB available (20.1B params)
  - model: openai/gpt-oss-20b
estimated: 20.1B params, 37.5 GB weights, 75.0 GB CPU RAM needed
- **configs/recipes/gpt_oss/sft/20b_lora_single_gpu_train.yaml**: Dry-run failed: Skipped: model needs ~75 GB CPU RAM but only 58 GB available (20.1B params)
  - model: openai/gpt-oss-20b
estimated: 20.1B params, 37.5 GB weights, 75.0 GB CPU RAM needed
- **configs/recipes/llama3_1/pretraining/8b/train.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
  - model: meta-llama/Llama-3.1-8B
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/recipes/llama3_1/sft/405b_full/train.yaml**: Dry-run failed: Skipped: model needs ~1504 GB CPU RAM but only 58 GB available (403.8B params)
  - model: meta-llama/Llama-3.1-405B-Instruct
estimated: 403.8B params, 752.0 GB weights, 1504.1 GB CPU RAM needed
- **configs/recipes/llama3_1/sft/405b_lora/train.yaml**: Dry-run failed: Skipped: model needs ~1504 GB CPU RAM but only 58 GB available (403.8B params)
  - model: meta-llama/Llama-3.1-405B-Instruct
estimated: 403.8B params, 752.0 GB weights, 1504.1 GB CPU RAM needed
- **configs/recipes/llama3_1/sft/405b_qlora/train.yaml**: Dry-run failed: Skipped: model needs ~1504 GB CPU RAM but only 58 GB available (403.8B params)
  - model: meta-llama/Llama-3.1-405B-Instruct
estimated: 403.8B params, 752.0 GB weights, 1504.1 GB CPU RAM needed
- **configs/recipes/llama3_1/sft/70b_full/train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: meta-llama/Llama-3.1-70B-Instruct
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/recipes/llama3_1/sft/70b_lora/train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: meta-llama/Llama-3.1-70B-Instruct
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/recipes/llama3_1/sft/70b_qlora/train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: meta-llama/Llama-3.1-70B-Instruct
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
  - model: meta-llama/Llama-3.1-8B-Instruct
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/recipes/llama3_1/sft/8b_full/train.yaml**: Dry-run failed: Skipped: model needs ~83.9 GB GPU but only 43.9 GB free (7.5B params)
  - model: meta-llama/Llama-3.1-8B-Instruct
estimated: 7.5B params, 14.0 GB weights, 28.0 GB CPU RAM needed
- **configs/recipes/llama3_3/sft/70b_full/train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: meta-llama/Llama-3.3-70B-Instruct
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/recipes/llama3_3/sft/70b_lora/train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: meta-llama/Llama-3.3-70B-Instruct
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/recipes/llama3_3/sft/70b_qlora/train.yaml**: Dry-run failed: Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
  - model: meta-llama/Llama-3.3-70B-Instruct
estimated: 69.5B params, 129.5 GB weights, 258.9 GB CPU RAM needed
- **configs/recipes/llama4/sft/scout_base_full/train.yaml**: Dry-run failed: Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
  - model: meta-llama/Llama-4-Scout-17B-16E
estimated: 100.7B params, 187.6 GB weights, 375.1 GB CPU RAM needed
- **configs/recipes/llama4/sft/scout_instruct_full/train.yaml**: Dry-run failed: Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
  - model: meta-llama/Llama-4-Scout-17B-16E-Instruct
estimated: 100.7B params, 187.6 GB weights, 375.1 GB CPU RAM needed
- **configs/recipes/llama4/sft/scout_instruct_lora/train.yaml**: Dry-run failed: Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
  - model: meta-llama/Llama-4-Scout-17B-16E-Instruct
estimated: 100.7B params, 187.6 GB weights, 375.1 GB CPU RAM needed
- **configs/recipes/llama4/sft/scout_instruct_qlora/train.yaml**: Dry-run failed: Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
  - model: meta-llama/Llama-4-Scout-17B-16E-Instruct
estimated: 100.7B params, 187.6 GB weights, 375.1 GB CPU RAM needed
- **configs/recipes/olmo3/sft/32b_lora/train.yaml**: Dry-run failed: OSError: allenai/Olmo-3-32B-Instruct is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
  - model: allenai/Olmo-3-32B-Instruct
- **configs/recipes/olmo3/sft/7b_full/train.yaml**: Dry-run failed: Skipped: model needs ~77.0 GB GPU but only 43.9 GB free (6.9B params)
  - model: allenai/Olmo-3-7B-Instruct
estimated: 6.9B params, 12.8 GB weights, 25.7 GB CPU RAM needed
- **configs/recipes/phi4/sft/reasoning_plus/full_train.yaml**: Dry-run failed: Skipped: model needs ~53 GB CPU RAM but only 58 GB available (14.1B params)
  - model: microsoft/Phi-4-reasoning-plus
estimated: 14.1B params, 26.3 GB weights, 52.7 GB CPU RAM needed
- **configs/recipes/phi4/sft/reasoning_plus/lora_train.yaml**: Dry-run failed: Skipped: model needs ~53 GB CPU RAM but only 58 GB available (14.1B params)
  - model: microsoft/Phi-4-reasoning-plus
estimated: 14.1B params, 26.3 GB weights, 52.7 GB CPU RAM needed
- **configs/recipes/phi4/sft/reasoning_plus/qlora_train.yaml**: Dry-run failed: Skipped: model needs ~53 GB CPU RAM but only 58 GB available (14.1B params)
  - model: microsoft/Phi-4-reasoning-plus
estimated: 14.1B params, 26.3 GB weights, 52.7 GB CPU RAM needed
- **configs/recipes/qwen2_5/sft/7b_full/train.yaml**: Dry-run failed: Skipped: model needs ~79.0 GB GPU but only 43.8 GB free (7.1B params)
  - model: Qwen/Qwen2.5-7B-Instruct
estimated: 7.1B params, 13.2 GB weights, 26.3 GB CPU RAM needed
- **configs/recipes/qwen3/sft/14b_lora/train.yaml**: Dry-run failed: Skipped: model needs ~52 GB CPU RAM but only 58 GB available (14.0B params)
  - model: Qwen/Qwen3-14B
estimated: 14.0B params, 26.1 GB weights, 52.1 GB CPU RAM needed
- **configs/recipes/qwen3/sft/235b_lora/train.yaml**: Dry-run failed: Skipped: model needs ~6783 GB CPU RAM but only 58 GB available (1820.8B params)
  - model: Qwen/Qwen3-235B-A22B
estimated: 1820.8B params, 3391.5 GB weights, 6783.0 GB CPU RAM needed
- **configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml**: Dry-run failed: Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
  - model: Qwen/Qwen3-30B-A3B
estimated: 232.7B params, 433.4 GB weights, 866.9 GB CPU RAM needed
- **configs/recipes/qwen3/sft/32b_lora/train.yaml**: Dry-run failed: Skipped: model needs ~111 GB CPU RAM but only 58 GB available (29.7B params)
  - model: Qwen/Qwen3-32B
estimated: 29.7B params, 55.4 GB weights, 110.7 GB CPU RAM needed
- **configs/recipes/qwen3/sft/4b_full/train.yaml**: Dry-run failed: Skipped: model needs ~41.0 GB GPU but only 43.9 GB free (3.7B params)
  - model: Qwen/Qwen3-4B
estimated: 3.7B params, 6.8 GB weights, 13.7 GB CPU RAM needed
- **configs/recipes/qwen3/sft/8b_full/train.yaml**: Dry-run failed: Skipped: model needs ~84.6 GB GPU but only 43.9 GB free (7.6B params)
  - model: Qwen/Qwen3-8B
estimated: 7.6B params, 14.1 GB weights, 28.2 GB CPU RAM needed
- **configs/recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml**: Dry-run failed: Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
  - model: Qwen/Qwen3-30B-A3B-Instruct-2507
estimated: 232.7B params, 433.4 GB weights, 866.9 GB CPU RAM needed
- **configs/recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml**: Dry-run failed: Skipped: model needs ~2883 GB CPU RAM but only 58 GB available (773.9B params)
  - model: Qwen/Qwen3-Next-80B-A3B-Instruct
estimated: 773.9B params, 1441.5 GB weights, 2883.0 GB CPU RAM needed
- **configs/recipes/qwen3_next/sft/80b_a3b_lora/train.yaml**: Dry-run failed: OSError: Qwen/Qwen3-Next-80B-A3B is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
  - model: Qwen/Qwen3-Next-80B-A3B
- **configs/recipes/qwq/sft/full_train.yaml**: Dry-run failed: Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
  - model: Qwen/QwQ-32B
estimated: 32.0B params, 59.6 GB weights, 119.2 GB CPU RAM needed
- **configs/recipes/qwq/sft/lora_train.yaml**: Dry-run failed: Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
  - model: Qwen/QwQ-32B
estimated: 32.0B params, 59.6 GB weights, 119.2 GB CPU RAM needed
- **configs/recipes/qwq/sft/qlora_train.yaml**: Dry-run failed: Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
  - model: Qwen/QwQ-32B
estimated: 32.0B params, 59.6 GB weights, 119.2 GB CPU RAM needed
- **configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml**: Dry-run failed: Skipped: model needs ~103.4 GB GPU but only 43.8 GB free (9.2B params)
  - model: meta-llama/Llama-3.2-11B-Vision-Instruct
estimated: 9.2B params, 17.2 GB weights, 34.5 GB CPU RAM needed
- **configs/recipes/vision/llama3_2_vision/sft/90b_full/train.yaml**: Dry-run failed: Skipped: model needs ~323 GB CPU RAM but only 58 GB available (86.6B params)
  - model: meta-llama/Llama-3.2-90B-Vision-Instruct
estimated: 86.6B params, 161.3 GB weights, 322.7 GB CPU RAM needed
- **configs/recipes/vision/llava_7b/dpo/train.yaml**: Dry-run failed: Skipped: model needs ~73.8 GB GPU but only 43.9 GB free (6.6B params)
  - model: llava-hf/llava-1.5-7b-hf
estimated: 6.6B params, 12.3 GB weights, 24.6 GB CPU RAM needed
- **configs/recipes/vision/llava_7b/sft/train.yaml**: Dry-run failed: Skipped: model needs ~73.8 GB GPU but only 43.9 GB free (6.6B params)
  - model: llava-hf/llava-1.5-7b-hf
estimated: 6.6B params, 12.3 GB weights, 24.6 GB CPU RAM needed
- **configs/recipes/vision/molmo/grpo/train.yaml**: Dry-run failed: Skipped: model needs ~84 GB CPU RAM but only 58 GB available (11.2B params)
  - model: oumi-ai/Molmo-7B-O-0924
estimated: 11.2B params, 41.8 GB weights, 83.6 GB CPU RAM needed
- **configs/recipes/vision/molmo/sft/molmo_d_full/train.yaml**: Dry-run failed: Skipped: model needs ~95 GB CPU RAM but only 58 GB available (12.8B params)
  - model: oumi-ai/Molmo-7B-D-0924
estimated: 12.8B params, 47.6 GB weights, 95.2 GB CPU RAM needed
- **configs/recipes/vision/molmo/sft/molmo_o_full/train.yaml**: Dry-run failed: Skipped: model needs ~84 GB CPU RAM but only 58 GB available (11.2B params)
  - model: oumi-ai/Molmo-7B-O-0924
estimated: 11.2B params, 41.8 GB weights, 83.6 GB CPU RAM needed
- **configs/recipes/vision/phi3/dpo/train.yaml**: Dry-run failed: AttributeError: type object 'DynamicCache' has no attribute 'from_legacy_cache'
  - model: microsoft/Phi-3-vision-128k-instruct
estimated: 3.7B params, 6.9 GB weights, 13.9 GB CPU RAM needed
params: 4.1B
LoRA trainable: 11.3M
device: cuda
- **configs/recipes/vision/phi3/sft/full/completions_only_train.yaml**: Dry-run failed: Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
  - model: microsoft/Phi-3-vision-128k-instruct
estimated: 3.7B params, 6.9 GB weights, 13.9 GB CPU RAM needed
- **configs/recipes/vision/phi3/sft/full/train.yaml**: Dry-run failed: Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
  - model: microsoft/Phi-3-vision-128k-instruct
estimated: 3.7B params, 6.9 GB weights, 13.9 GB CPU RAM needed
- **configs/recipes/vision/phi3/sft/lora/train.yaml**: Dry-run failed: AttributeError: type object 'DynamicCache' has no attribute 'from_legacy_cache'
  - model: microsoft/Phi-3-vision-128k-instruct
estimated: 3.7B params, 6.9 GB weights, 13.9 GB CPU RAM needed
params: 4.1B
LoRA trainable: 5.6M
device: cuda
- **configs/recipes/vision/phi4/sft/full/train.yaml**: Dry-run failed: Skipped: model needs ~42.9 GB GPU but only 43.9 GB free (3.8B params)
  - model: microsoft/Phi-4-multimodal-instruct
estimated: 3.8B params, 7.1 GB weights, 14.3 GB CPU RAM needed
- **configs/recipes/vision/phi4/sft/lora/train.yaml**: Dry-run failed: ImportError: cannot import name 'SlidingWindowCache' from 'transformers.cache_utils' (/root/miniconda3/envs/oumi/lib/python3.11/site-packages/transformers/cache_utils.py)
  - model: microsoft/Phi-4-multimodal-instruct
estimated: 3.8B params, 7.1 GB weights, 14.3 GB CPU RAM needed
- **configs/recipes/vision/qwen2_5_vl_3b/dpo/train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerati
  - model: Qwen/Qwen2.5-VL-3B-Instruct
estimated: 3.1B params, 5.7 GB weights, 11.5 GB CPU RAM needed
- **configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerati
  - model: Qwen/Qwen2.5-VL-3B-Instruct
estimated: 3.1B params, 5.7 GB weights, 11.5 GB CPU RAM needed
- **configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerati
  - model: Qwen/Qwen2.5-VL-3B-Instruct
estimated: 3.1B params, 5.7 GB weights, 11.5 GB CPU RAM needed
- **configs/recipes/vision/qwen2_5_vl_7b/sft/full/train.yaml**: Dry-run failed: Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
  - model: Qwen/Qwen2.5-VL-7B-Instruct
estimated: 7.1B params, 13.2 GB weights, 26.3 GB CPU RAM needed
- **configs/recipes/vision/qwen2_vl_2b/dpo/train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
  - model: Qwen/Qwen2-VL-2B-Instruct
estimated: 1.5B params, 2.9 GB weights, 5.8 GB CPU RAM needed
- **configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
  - model: Qwen/Qwen2-VL-2B-Instruct
estimated: 1.5B params, 2.9 GB weights, 5.8 GB CPU RAM needed
- **configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
  - model: Qwen/Qwen2-VL-2B-Instruct
estimated: 1.5B params, 2.9 GB weights, 5.8 GB CPU RAM needed
- **configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_fft_train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
  - model: Qwen/Qwen3-VL-2B-Instruct
estimated: 1.7B params, 3.2 GB weights, 6.4 GB CPU RAM needed
- **configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
  - model: Qwen/Qwen3-VL-2B-Instruct
estimated: 1.7B params, 3.2 GB weights, 6.4 GB CPU RAM needed
- **configs/recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_fft_train.yaml**: Dry-run failed: Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
  - model: Qwen/Qwen3-VL-30B-A3B-Instruct
estimated: 232.7B params, 433.4 GB weights, 866.9 GB CPU RAM needed
- **configs/recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_lora_train.yaml**: Dry-run failed: Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
  - model: Qwen/Qwen3-VL-30B-A3B-Instruct
estimated: 232.7B params, 433.4 GB weights, 866.9 GB CPU RAM needed
- **configs/recipes/vision/qwen3_vl_4b/sft/4b_instruct_fft_train.yaml**: Dry-run failed: Skipped: model needs ~41.0 GB GPU but only 43.9 GB free (3.7B params)
  - model: Qwen/Qwen3-VL-4B-Instruct
estimated: 3.7B params, 6.8 GB weights, 13.7 GB CPU RAM needed
- **configs/recipes/vision/qwen3_vl_4b/sft/4b_instruct_lora_train.yaml**: Dry-run failed: ValueError: Unrecognized configuration class <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
  - model: Qwen/Qwen3-VL-4B-Instruct
estimated: 3.7B params, 6.8 GB weights, 13.7 GB CPU RAM needed

## Warnings (311)

- **configs/projects/coalm/8b_train.yaml**: PEFT enabled but no lora_target_modules specified
- **configs/recipes/gpt_oss/inference/120b_together_infer.yaml**: No max_new_tokens set — generation may run indefinitely
- **configs/recipes/qwen3/inference/235b_a22b_fireworks_infer.yaml**: No max_new_tokens set — generation may run indefinitely
- **configs/recipes/qwen3/inference/235b_a22b_together_infer.yaml**: No max_new_tokens set — generation may run indefinitely
- **configs/recipes/vision/llava_7b/dpo/train.yaml**: FSDP TRANSFORMER_BASED_WRAP without transformer_layer_cls
- **configs/examples/berry_bench/evaluation/eval.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/examples/deepspeed/llama3_1_8b_deepspeed_z2_train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/examples/deepspeed/llama3_1_8b_deepspeed_z3_offload_train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/examples/deepspeed/llama3_1_8b_deepspeed_z3_train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/examples/fineweb_ablation_pretraining/ddp/train.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceFW/ablation-model-fineweb-v1' does not require it. Prefer trust_remote_code=False for security.
- **configs/examples/fineweb_ablation_pretraining/ddp/train.yaml**: pad_token is set to eos_token. The model may learn to ignore EOS, causing generation to never terminate. Consider using a dedicated pad token.
- **configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceFW/ablation-model-fineweb-v1' does not require it. Prefer trust_remote_code=False for security.
- **configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml**: pad_token is set to eos_token. The model may learn to ignore EOS, causing generation to never terminate. Consider using a dedicated pad token.
- **configs/examples/gkd/train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/examples/gold/train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/examples/gold/train_gptoss120b_qwen06b.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-0.6B' does not require it. Prefer trust_remote_code=False for security.
- **configs/examples/gold/train_gptoss120b_qwen8b.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/examples/grpo_verl_countdown/train.yaml**: Tokenizer's pad_token == eos_token ('<|end_of_text|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/examples/letter_counting/evaluation/eval.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/examples/letter_counting/grpo/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/examples/misc/tulu3_sft_mini.yaml**: Config sets trust_remote_code=True but model 'EleutherAI/pythia-14m' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/aya/evaluation/eval.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/aya/sft/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/aya/sft/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/projects/chatqa/chatqa_stage1_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/projects/chatqa/chatqa_stage2_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/projects/coalm/405b_train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-405B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/coalm/70b_infer.yaml**: Config sets trust_remote_code=True but model 'uiuc-convai/CoALM-70B' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/coalm/70b_train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.3-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/coalm/8b_infer.yaml**: Config sets trust_remote_code=True but model 'uiuc-convai/CoALM-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/coalm/8b_train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/dcvlr/starter_kit/molmo-d-train-openr1.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/projects/dcvlr/starter_kit/qwenvl-openr1.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-VL-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/halloumi/8b_train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/limo/qwen2.5_7b_fft.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/limo/qwen2.5_7b_fft_yarn.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/limo/qwen2.5_7b_fft_yarn_deepspeed.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/limo/qwen2.5_7b_fft_yarn_deepspeed_memory_optimized_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/wc50m/configs/base_ultrachat.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Meta-Llama-3.1-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/projects/wc50m/configs/base_ultrachat.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/deepseek_r1/evaluation/distill_llama_70b/eval.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/evaluation/distill_llama_8b/eval.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/evaluation/distill_qwen_1_5b/eval.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/evaluation/distill_qwen_32b/eval.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/inference/distill_llama_70b/infer.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/inference/distill_llama_8b/infer.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/inference/distill_qwen_1_5b/infer.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/inference/distill_qwen_32b/infer.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/inference/distill_qwen_32b/vllm_infer.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml**: Config sets trust_remote_code=True but model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<｜end▁of▁sentence｜>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/falcon_e/dpo/falcon_e_1b_instruct/dpo.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-E-1B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_e/sft/falcon_e_1b/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-E-1B-Base' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_e/sft/falcon_e_1b_instruct/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-E-1B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_e/sft/falcon_e_3b/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-E-3B-Base' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_e/sft/falcon_e_3b_instruct/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-E-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-0.5B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/evaluation/falcon_h1_0_5b/eval.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-0.5B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/evaluation/falcon_h1_1_5b/eval.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-1.5B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/evaluation/falcon_h1_1_5b_deep/eval.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-1.5B-Deep-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/evaluation/falcon_h1_34b/eval.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-34B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/evaluation/falcon_h1_3b/eval.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/evaluation/falcon_h1_7b/eval.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/inference/0_5b_infer.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-0.5B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/inference/1_5b_deep_infer.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-1.5B-Deep-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/inference/1_5b_infer.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-1.5B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/inference/34b_infer.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-34B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/inference/3b_infer.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/inference/7b_infer.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/sft/falcon_h1_0_5b/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-0.5B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/sft/falcon_h1_1_5b/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-1.5B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-1.5B-Deep-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/sft/falcon_h1_34b/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-34B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/sft/falcon_h1_3b/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml**: Config sets trust_remote_code=True but model 'tiiuae/Falcon-H1-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/evaluation/12b/eval.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3-12b-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/evaluation/27b/eval.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3-27b-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/evaluation/4b/eval.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3-4b-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/inference/12b_instruct_infer.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3-12b-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/inference/27b_instruct_infer.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3-27b-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/inference/3n_e4b_it_infer.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3n-E4B-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/inference/3n_e4b_it_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3n-E4B-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/inference/4b_instruct_infer.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3-4b-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/sft/12b_lora/train.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3-12b-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/sft/12b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/gemma3/sft/27b_lora/train.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3-27b-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gemma3/sft/27b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/gemma3/sft/4b_full/train.yaml**: Config sets trust_remote_code=True but model 'google/gemma-3-4b-it' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gpt2/inference/infer.yaml**: Config sets trust_remote_code=True but model 'openai-community/gpt2' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gpt_oss/inference/120b_infer.yaml**: Config sets trust_remote_code=True but model 'openai/gpt-oss-120b' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gpt_oss/inference/120b_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'openai/gpt-oss-120b' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gpt_oss/inference/20b_infer.yaml**: Config sets trust_remote_code=True but model 'openai/gpt-oss-20b' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gpt_oss/inference/20b_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'openai/gpt-oss-20b' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gpt_oss/sft/120b_lora_multi_gpu_train.yaml**: Config sets trust_remote_code=True but model 'openai/gpt-oss-120b' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gpt_oss/sft/120b_lora_multi_gpu_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/gpt_oss/sft/20b_lora_multi_gpu_train.yaml**: Config sets trust_remote_code=True but model 'openai/gpt-oss-20b' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/gpt_oss/sft/20b_lora_multi_gpu_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/gpt_oss/sft/20b_lora_single_gpu_train.yaml**: Config sets trust_remote_code=True but model 'openai/gpt-oss-20b' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/evaluation/70b_eval.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/evaluation/8b_eval.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/inference/70b_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/inference/8b_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/inference/8b_sglang_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/pretraining/8b/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/pretraining/8b/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_1/sft/405b_full/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-405B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/405b_lora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-405B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/405b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_1/sft/405b_qlora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-405B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/70b_full/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/70b_full/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_1/sft/70b_lora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/70b_lora/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_1/sft/70b_qlora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/70b_qlora/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_1/sft/8b_full/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/8b_full/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_1/sft/8b_lora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/8b_lora/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_1/sft/8b_qlora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.1-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_1/sft/8b_qlora/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_1/sft/8b_qlora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/llama3_2/dpo/1b_qlora_dpo.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-1B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/dpo/1b_qlora_dpo.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_2/evaluation/1b_eval.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-1B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/evaluation/3b_eval.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/inference/1b_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-1B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/inference/1b_sglang_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-1B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/inference/1b_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-1B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/inference/3b_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/inference/3b_sglang_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/inference/3b_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/sft/1b_full/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-1B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/sft/1b_full/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_2/sft/3b_full/fsdp_train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/sft/3b_full/fsdp_train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_2/sft/3b_full/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/sft/3b_full/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_2/sft/3b_lora/fsdp_train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/sft/3b_lora/fsdp_train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_2/sft/3b_lora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/sft/3b_lora/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_2/sft/3b_qlora/fsdp_train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/sft/3b_qlora/fsdp_train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_2/sft/3b_qlora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_2/sft/3b_qlora/train.yaml**: Tokenizer has no pad_token. Set model.tokenizer_pad_token in config.
- **configs/recipes/llama3_3/evaluation/70b_eval.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.3-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_3/inference/70b_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.3-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_3/inference/70b_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.3-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_3/sft/70b_full/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.3-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_3/sft/70b_lora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.3-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama3_3/sft/70b_qlora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.3-70B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama4/evaluation/scout_instruct_eval.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-4-Scout-17B-16E-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama4/inference/scout_instruct_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-4-Scout-17B-16E-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama4/inference/scout_instruct_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-4-Scout-17B-16E-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama4/sft/scout_base_full/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-4-Scout-17B-16E' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama4/sft/scout_instruct_full/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-4-Scout-17B-16E-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama4/sft/scout_instruct_lora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-4-Scout-17B-16E-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/llama4/sft/scout_instruct_qlora/train.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-4-Scout-17B-16E-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/olmo3/evaluation/32b/eval.yaml**: Cannot load model config: allenai/Olmo-3-32B-Instruct (may be unreleased, gated, or renamed)
- **configs/recipes/olmo3/evaluation/7b/eval.yaml**: Config sets trust_remote_code=True but model 'allenai/Olmo-3-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/olmo3/inference/32b_infer.yaml**: Cannot load model config: allenai/Olmo-3-32B-Instruct (may be unreleased, gated, or renamed)
- **configs/recipes/olmo3/inference/7b_infer.yaml**: Config sets trust_remote_code=True but model 'allenai/Olmo-3-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/olmo3/sft/32b_lora/train.yaml**: Cannot load model config: allenai/Olmo-3-32B-Instruct (may be unreleased, gated, or renamed)
- **configs/recipes/olmo3/sft/7b_full/train.yaml**: Config sets trust_remote_code=True but model 'allenai/Olmo-3-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi3/dpo/macos_train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-3-mini-4k-instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi3/dpo/macos_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/dpo/nvidia_24g_train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-3-mini-4k-instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi3/dpo/nvidia_24g_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/dpo/nvidia_80g_train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-3-mini-4k-instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi3/dpo/nvidia_80g_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/dpo/train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-3-mini-4k-instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi3/dpo/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/kto/train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-3-mini-4k-instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi3/kto/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/sft/lora_macos_train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-3-mini-4k-instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi3/sft/lora_macos_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi3/sft/lora_train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-3-mini-4k-instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi3/sft/lora_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/phi4/evaluation/reasoning_plus_eval.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-4-reasoning-plus' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi4/inference/reasoning_plus_infer.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-4-reasoning-plus' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi4/sft/reasoning_plus/full_train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-4-reasoning-plus' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi4/sft/reasoning_plus/lora_train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-4-reasoning-plus' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/phi4/sft/reasoning_plus/lora_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/phi4/sft/reasoning_plus/qlora_train.yaml**: Config sets trust_remote_code=True but model 'microsoft/Phi-4-reasoning-plus' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen2_5/sft/3b_full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen2_5/sft/7b_full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/evaluation/0.6b_eval.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-0.6B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/evaluation/1.7b_eval.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-1.7B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/evaluation/14b_eval.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-14B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/evaluation/30b_a3b_eval.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-30B-A3B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/evaluation/32b_eval.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/evaluation/4b_eval.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-4B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/evaluation/8b_eval.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/inference/0.6b_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-0.6B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/inference/1.7b_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-1.7B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/inference/14b_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-14B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/inference/30b_a3b_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-30B-A3B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/inference/30b_a3b_instruct_vllm_infer.yaml**: Cannot load model config: Qwen/Qwen3-30B-A3B-Instruct (may be unreleased, gated, or renamed)
- **configs/recipes/qwen3/inference/32b_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/inference/4b_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-4B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/inference/4b_instruct_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-4B-Instruct-2507' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/inference/4b_instruct_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-4B-Instruct-2507' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/inference/8b_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/sft/0.6b_full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-0.6B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/sft/1.7b_full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-1.7B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/sft/14b_lora/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-14B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/sft/14b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/235b_lora/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-235B-A22B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/sft/235b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-30B-A3B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/32b_lora/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/sft/32b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3/sft/4b_full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-4B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3/sft/8b_full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3_5/inference/0.8b_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3.5-0.8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3_5/inference/0.8b_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3.5-0.8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3_5/sft/0.8b_full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3.5-0.8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3_5/sft/0.8b_lora/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3.5-0.8B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3_coder/inference/30b_a3b_instruct_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-Coder-30B-A3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-30B-A3B-Instruct-2507' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3_next/evaluation/80b_a3b_eval.yaml**: Cannot load model config: Qwen/Qwen3-Next-80B-A3B (may be unreleased, gated, or renamed)
- **configs/recipes/qwen3_next/inference/80b_a3b_infer.yaml**: Cannot load model config: Qwen/Qwen3-Next-80B-A3B (may be unreleased, gated, or renamed)
- **configs/recipes/qwen3_next/inference/80b_a3b_instruct_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-Next-80B-A3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-Next-80B-A3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwen3_next/sft/80b_a3b_lora/train.yaml**: Cannot load model config: Qwen/Qwen3-Next-80B-A3B (may be unreleased, gated, or renamed)
- **configs/recipes/qwq/evaluation/eval.yaml**: Config sets trust_remote_code=True but model 'Qwen/QwQ-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwq/inference/infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/QwQ-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwq/sft/full_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/QwQ-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwq/sft/lora_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/QwQ-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/qwq/sft/lora_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/qwq/sft/qlora_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/QwQ-32B' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/smollm/evaluation/135m/eval.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolLM2-135M-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v1_eval.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolLM2-135M-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v2_eval.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolLM2-135M-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolLM2-135M-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/smollm/inference/135m_infer.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolLM2-135M-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/smollm/sft/135m/quickstart_train.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolLM2-135M-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/smollm/sft/135m/quickstart_train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/smollm/sft/135m/train.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolLM2-135M-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/smollm/sft/135m/train.yaml**: Tokenizer's pad_token == eos_token ('<|im_end|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/smollm/tuning/135m/tune.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolLM2-135M-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/internvl3/sft/full/train.yaml**: Config sets trust_remote_code=True but model 'OpenGVLab/InternVL3-1B-hf' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-11B-Vision-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-11B-Vision-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml**: Config sets trust_remote_code=True but model 'meta-llama/Llama-3.2-11B-Vision-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/vision/llava_7b/dpo/train.yaml**: Config sets trust_remote_code=True but model 'llava-hf/llava-1.5-7b-hf' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/llava_7b/inference/infer.yaml**: Config sets trust_remote_code=True but model 'llava-hf/llava-1.5-7b-hf' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/llava_7b/inference/vllm_infer.yaml**: Config sets trust_remote_code=True but model 'llava-hf/llava-1.5-7b-hf' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/llava_7b/sft/train.yaml**: Config sets trust_remote_code=True but model 'llava-hf/llava-1.5-7b-hf' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/molmo/sft/molmo_d_full/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/dpo/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/sft/full/completions_only_train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/sft/full/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi3/sft/lora/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi4/sft/full/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/phi4/sft/lora/train.yaml**: Could not load architecture: Could not instantiate model on meta device
- **configs/recipes/vision/phi4/sft/lora/train.yaml**: Tokenizer's pad_token == eos_token ('<|endoftext|>'). The model may learn to ignore EOS during training. Consider setting a dedicated pad token via model.tokenizer_pad_token.
- **configs/recipes/vision/qwen2_5_vl_3b/dpo/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-VL-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-VL-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-VL-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-VL-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-VL-3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_5_vl_7b/sft/full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2.5-VL-7B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_vl_2b/dpo/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_vl_2b/evaluation/eval.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen2-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_2b/inference/infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_2b/inference/vllm_infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_fft_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-2B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_fft_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-30B-A3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_lora_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-30B-A3B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_lora_train.yaml**: gradient_accumulation + FSDP + PEFT LoRA + bf16 can cause RuntimeError (dtype mismatch). See: https://discuss.huggingface.co/t/105006
- **configs/recipes/vision/qwen3_vl_4b/inference/infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-4B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_4b/sft/4b_instruct_fft_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-4B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_4b/sft/4b_instruct_lora_train.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-4B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/qwen3_vl_8b/inference/infer.yaml**: Config sets trust_remote_code=True but model 'Qwen/Qwen3-VL-8B-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/smolvlm/inference/infer.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolVLM-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/smolvlm/inference/vllm_infer.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolVLM-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/smolvlm/sft/full/train.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolVLM-Instruct' does not require it. Prefer trust_remote_code=False for security.
- **configs/recipes/vision/smolvlm/sft/lora/train.yaml**: Config sets trust_remote_code=True but model 'HuggingFaceTB/SmolVLM-Instruct' does not require it. Prefer trust_remote_code=False for security.

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
| examples/misc/tulu3_sft_mini.yaml | 1.0 GB | 0.6 GB | 0.0B | FFT |
| projects/aya/sft/train.yaml | 150.7 GB | 87.1 GB | 7.5B | FFT |
| projects/chatqa/chatqa_stage1_train.yaml | 58.7 GB | 44.2 GB | 3.7B | FFT |
| projects/chatqa/chatqa_stage2_train.yaml | 58.7 GB | 44.2 GB | 3.7B | FFT |
| projects/coalm/405b_train.yaml | 1087.5 GB | 434.9 GB | 403.8B | LoRA+Q |
| projects/coalm/70b_train.yaml | 1165.6 GB | 405.9 GB | 69.5B | FFT |
| projects/coalm/8b_train.yaml | 362.0 GB | 11.1 GB | 7.5B | LoRA |
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
| recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml | 77.8 GB | 2.0 GB | 0.4B | LoRA+Q |
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
| recipes/phi3/dpo/train.yaml | 63.0 GB | 9.9 GB | 3.7B | LoRA |
| recipes/phi3/kto/train.yaml | 63.0 GB | 9.9 GB | 3.7B | LoRA |
| recipes/phi3/sft/lora_macos_train.yaml | 63.0 GB | 9.9 GB | 3.7B | LoRA |
| recipes/phi3/sft/lora_train.yaml | 118.2 GB | 9.9 GB | 3.7B | LoRA |
| recipes/phi4/sft/reasoning_plus/full_train.yaml | 115.4 GB | 85.2 GB | 14.1B | FFT |
| recipes/phi4/sft/reasoning_plus/lora_train.yaml | 49.6 GB | 19.3 GB | 14.1B | LoRA |
| recipes/phi4/sft/reasoning_plus/qlora_train.yaml | 61.7 GB | 31.4 GB | 14.1B | LoRA+Q |
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
| recipes/vision/phi3/dpo/train.yaml | 43.1 GB | 9.9 GB | 3.7B | LoRA |
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

## Dry-Run Results (53 passed, 97 failed)

- examples/deepspeed/llama3_1_8b_deepspeed_z2_train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 44.0 GB free (7.5B params)
- examples/deepspeed/llama3_1_8b_deepspeed_z3_offload_train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 44.0 GB free (7.5B params)
- examples/deepspeed/llama3_1_8b_deepspeed_z3_train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 44.0 GB free (7.5B params)
- examples/fineweb_ablation_pretraining/ddp/train.yaml: passed in 23.3s (16.0 GB)
- examples/fineweb_ablation_pretraining/fsdp/train.yaml: passed in 22.2s (16.0 GB)
- examples/gkd/train.yaml: passed in 3.0s (1.3 GB)
- examples/gold/train.yaml: passed in 2.5s (1.3 GB)
- examples/gold/train_gptoss120b_qwen06b.yaml: passed in 9.5s (5.6 GB)
- examples/gold/train_gptoss120b_qwen8b.yaml: **FAILED** — Skipped: model needs ~84.6 GB GPU but only 43.7 GB free (7.6B params)
- examples/grpo_tldr/train.yaml: passed in 7.7s (4.7 GB)
- examples/grpo_verl_countdown/train.yaml: passed in 43.2s (30.0 GB)
- examples/grpo_verl_geometry3k/train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerati
- examples/grpo_verl_gsm8k/train.yaml: passed in 7.3s (4.7 GB)
- examples/letter_counting/grpo/train.yaml: passed in 43.1s (30.0 GB)
- examples/misc/tulu3_sft_mini.yaml: passed in 1.3s (0.2 GB)
- projects/aya/sft/train.yaml: **FAILED** — Skipped: model needs ~83.9 GB GPU but only 43.8 GB free (7.5B params)
- projects/chatqa/chatqa_stage1_train.yaml: **FAILED** — Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
- projects/chatqa/chatqa_stage2_train.yaml: **FAILED** — Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
- projects/coalm/405b_train.yaml: **FAILED** — Skipped: model needs ~1504 GB CPU RAM but only 58 GB available (403.8B params)
- projects/coalm/70b_train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- projects/coalm/8b_train.yaml: passed in 89.2s (15.9 GB)
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
- recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml: passed in 87.7s (15.9 GB)
- recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml: passed in 90.7s (16.6 GB)
- recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml: passed in 22.2s (16.6 GB)
- recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml: passed in 23.3s (4.9 GB)
- recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml: **FAILED** — Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
- recipes/falcon_e/dpo/falcon_e_1b_instruct/dpo.yaml: passed in 20.0s (16.0 GB)
- recipes/falcon_e/sft/falcon_e_1b/full_train.yaml: passed in 19.9s (16.0 GB)
- recipes/falcon_e/sft/falcon_e_1b_instruct/full_train.yaml: passed in 19.7s (16.0 GB)
- recipes/falcon_e/sft/falcon_e_3b/full_train.yaml: passed in 39.2s (28.5 GB)
- recipes/falcon_e/sft/falcon_e_3b_instruct/full_train.yaml: passed in 39.0s (28.5 GB)
- recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_0_5b/full_train.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_1_5b/full_train.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_1_5b_deep/full_train.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_34b/full_train.yaml: **FAILED** — Skipped: model needs ~110 GB CPU RAM but only 58 GB available (29.6B params)
- recipes/falcon_h1/sft/falcon_h1_3b/full_train.yaml: **FAILED** — AssertionError: causal_conv1d_cuda is not available. Please install causal-conv1d.
- recipes/falcon_h1/sft/falcon_h1_7b/full_train.yaml: **FAILED** — Skipped: model needs ~71.0 GB GPU but only 43.8 GB free (6.4B params)
- recipes/gemma2/sft/2b_full/train.yaml: passed in 37.5s (24.4 GB)
- recipes/gemma3/sft/12b_lora/train.yaml: passed in 163.9s (24.1 GB)
- recipes/gemma3/sft/27b_lora/train.yaml: **FAILED** — Skipped: model needs ~105 GB CPU RAM but only 58 GB available (28.3B params)
- recipes/gemma3/sft/4b_full/train.yaml: **FAILED** — Skipped: model needs ~44.9 GB GPU but only 43.9 GB free (4.0B params)
- recipes/gpt2/pretraining/macos_train.yaml: passed in 4.7s (1.2 GB)
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
- recipes/llama3_1/sft/8b_lora/fsdp_train.yaml: passed in 98.9s (15.9 GB)
- recipes/llama3_1/sft/8b_lora/train.yaml: passed in 104.7s (15.9 GB)
- recipes/llama3_1/sft/8b_qlora/train.yaml: passed in 95.9s (16.6 GB)
- recipes/llama3_2/dpo/1b_qlora_dpo.yaml: passed in 22.3s (3.0 GB)
- recipes/llama3_2/sft/1b_full/train.yaml: passed in 17.7s (11.6 GB)
- recipes/llama3_2/sft/3b_full/fsdp_train.yaml: passed in 41.3s (30.0 GB)
- recipes/llama3_2/sft/3b_full/train.yaml: passed in 40.6s (30.0 GB)
- recipes/llama3_2/sft/3b_lora/fsdp_train.yaml: passed in 48.7s (7.9 GB)
- recipes/llama3_2/sft/3b_lora/train.yaml: passed in 49.4s (7.9 GB)
- recipes/llama3_2/sft/3b_qlora/fsdp_train.yaml: passed in 49.6s (7.9 GB)
- recipes/llama3_2/sft/3b_qlora/train.yaml: passed in 48.3s (7.9 GB)
- recipes/llama3_3/sft/70b_full/train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/llama3_3/sft/70b_lora/train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/llama3_3/sft/70b_qlora/train.yaml: **FAILED** — Skipped: model needs ~259 GB CPU RAM but only 58 GB available (69.5B params)
- recipes/llama4/sft/scout_base_full/train.yaml: **FAILED** — Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
- recipes/llama4/sft/scout_instruct_full/train.yaml: **FAILED** — Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
- recipes/llama4/sft/scout_instruct_lora/train.yaml: **FAILED** — Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
- recipes/llama4/sft/scout_instruct_qlora/train.yaml: **FAILED** — Skipped: model needs ~375 GB CPU RAM but only 58 GB available (100.7B params)
- recipes/olmo3/sft/32b_lora/train.yaml: **FAILED** — OSError: allenai/Olmo-3-32B-Instruct is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
- recipes/olmo3/sft/7b_full/train.yaml: **FAILED** — Skipped: model needs ~77.0 GB GPU but only 43.9 GB free (6.9B params)
- recipes/phi3/dpo/macos_train.yaml: passed in 42.6s (8.0 GB)
- recipes/phi3/dpo/nvidia_24g_train.yaml: passed in 42.3s (7.7 GB)
- recipes/phi3/dpo/nvidia_80g_train.yaml: passed in 45.5s (8.0 GB)
- recipes/phi3/dpo/train.yaml: passed in 42.2s (7.9 GB)
- recipes/phi3/kto/train.yaml: passed in 40.8s (7.9 GB)
- recipes/phi3/sft/lora_macos_train.yaml: passed in 40.7s (7.9 GB)
- recipes/phi3/sft/lora_train.yaml: passed in 41.1s (7.9 GB)
- recipes/phi4/sft/reasoning_plus/full_train.yaml: **FAILED** — Skipped: model needs ~53 GB CPU RAM but only 58 GB available (14.1B params)
- recipes/phi4/sft/reasoning_plus/lora_train.yaml: **FAILED** — Skipped: model needs ~53 GB CPU RAM but only 58 GB available (14.1B params)
- recipes/phi4/sft/reasoning_plus/qlora_train.yaml: **FAILED** — Skipped: model needs ~53 GB CPU RAM but only 58 GB available (14.1B params)
- recipes/qwen2_5/sft/3b_full/train.yaml: passed in 38.8s (29.3 GB)
- recipes/qwen2_5/sft/7b_full/train.yaml: **FAILED** — Skipped: model needs ~79.0 GB GPU but only 43.8 GB free (7.1B params)
- recipes/qwen3/sft/0.6b_full/train.yaml: passed in 8.3s (5.6 GB)
- recipes/qwen3/sft/1.7b_full/train.yaml: passed in 22.2s (16.1 GB)
- recipes/qwen3/sft/14b_lora/train.yaml: **FAILED** — Skipped: model needs ~52 GB CPU RAM but only 58 GB available (14.0B params)
- recipes/qwen3/sft/235b_lora/train.yaml: **FAILED** — Skipped: model needs ~6783 GB CPU RAM but only 58 GB available (1820.8B params)
- recipes/qwen3/sft/30b_a3b_lora/train.yaml: **FAILED** — Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
- recipes/qwen3/sft/32b_lora/train.yaml: **FAILED** — Skipped: model needs ~111 GB CPU RAM but only 58 GB available (29.7B params)
- recipes/qwen3/sft/4b_full/train.yaml: **FAILED** — Skipped: model needs ~41.0 GB GPU but only 43.9 GB free (3.7B params)
- recipes/qwen3/sft/8b_full/train.yaml: **FAILED** — Skipped: model needs ~84.6 GB GPU but only 43.9 GB free (7.6B params)
- recipes/qwen3_5/sft/0.8b_full/train.yaml: passed in 13.0s (7.2 GB)
- recipes/qwen3_5/sft/0.8b_lora/train.yaml: passed in 13.1s (2.5 GB)
- recipes/qwen3_instruct/sft/30b_a3b_lora/train.yaml: **FAILED** — Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
- recipes/qwen3_next/sft/80b_a3b_instruct_lora/train.yaml: **FAILED** — Skipped: model needs ~2883 GB CPU RAM but only 58 GB available (773.9B params)
- recipes/qwen3_next/sft/80b_a3b_lora/train.yaml: **FAILED** — OSError: Qwen/Qwen3-Next-80B-A3B is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
- recipes/qwq/sft/full_train.yaml: **FAILED** — Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
- recipes/qwq/sft/lora_train.yaml: **FAILED** — Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
- recipes/qwq/sft/qlora_train.yaml: **FAILED** — Skipped: model needs ~119 GB CPU RAM but only 58 GB available (32.0B params)
- recipes/smollm/sft/135m/quickstart_train.yaml: passed in 3.6s (1.3 GB)
- recipes/smollm/sft/135m/train.yaml: passed in 2.7s (1.3 GB)
- recipes/vision/internvl3/sft/full/train.yaml: passed in 7.9s (6.0 GB)
- recipes/vision/llama3_2_vision/sft/11b_full/train.yaml: **FAILED** — Skipped: model needs ~103.4 GB GPU but only 43.8 GB free (9.2B params)
- recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml: passed in 262.2s (20.0 GB)
- recipes/vision/llama3_2_vision/sft/90b_full/train.yaml: **FAILED** — Skipped: model needs ~323 GB CPU RAM but only 58 GB available (86.6B params)
- recipes/vision/llava_7b/dpo/train.yaml: **FAILED** — Skipped: model needs ~73.8 GB GPU but only 43.9 GB free (6.6B params)
- recipes/vision/llava_7b/sft/train.yaml: **FAILED** — Skipped: model needs ~73.8 GB GPU but only 43.9 GB free (6.6B params)
- recipes/vision/molmo/grpo/train.yaml: **FAILED** — Skipped: model needs ~84 GB CPU RAM but only 58 GB available (11.2B params)
- recipes/vision/molmo/sft/molmo_d_full/train.yaml: **FAILED** — Skipped: model needs ~95 GB CPU RAM but only 58 GB available (12.8B params)
- recipes/vision/molmo/sft/molmo_o_full/train.yaml: **FAILED** — Skipped: model needs ~84 GB CPU RAM but only 58 GB available (11.2B params)
- recipes/vision/phi3/dpo/train.yaml: **FAILED** — AttributeError: type object 'DynamicCache' has no attribute 'from_legacy_cache'
- recipes/vision/phi3/sft/full/completions_only_train.yaml: **FAILED** — Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
- recipes/vision/phi3/sft/full/train.yaml: **FAILED** — Skipped: model needs ~41.6 GB GPU but only 43.9 GB free (3.7B params)
- recipes/vision/phi3/sft/lora/train.yaml: **FAILED** — AttributeError: type object 'DynamicCache' has no attribute 'from_legacy_cache'
- recipes/vision/phi4/sft/full/train.yaml: **FAILED** — Skipped: model needs ~42.9 GB GPU but only 43.9 GB free (3.8B params)
- recipes/vision/phi4/sft/lora/train.yaml: **FAILED** — ImportError: cannot import name 'SlidingWindowCache' from 'transformers.cache_utils' (/root/miniconda3/envs/oumi/lib/python3.11/site-packages/transformers/cache_utils.py)
- recipes/vision/qwen2_5_vl_3b/dpo/train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerati
- recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerati
- recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerati
- recipes/vision/qwen2_5_vl_7b/sft/full/train.yaml: **FAILED** — Skipped: model needs ~79.0 GB GPU but only 43.9 GB free (7.1B params)
- recipes/vision/qwen2_vl_2b/dpo/train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
- recipes/vision/qwen2_vl_2b/sft/full/train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
- recipes/vision/qwen2_vl_2b/sft/lora/train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
- recipes/vision/qwen3_vl_2b/sft/2b_instruct_fft_train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
- recipes/vision/qwen3_vl_2b/sft/2b_instruct_lora_train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
- recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_fft_train.yaml: **FAILED** — Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
- recipes/vision/qwen3_vl_30b_a3b/sft/30b_a3b_instruct_lora_train.yaml: **FAILED** — Skipped: model needs ~867 GB CPU RAM but only 58 GB available (232.7B params)
- recipes/vision/qwen3_vl_4b/sft/4b_instruct_fft_train.yaml: **FAILED** — Skipped: model needs ~41.0 GB GPU but only 43.9 GB free (3.7B params)
- recipes/vision/qwen3_vl_4b/sft/4b_instruct_lora_train.yaml: **FAILED** — ValueError: Unrecognized configuration class <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ApertusConfig, ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfi
- recipes/vision/smolvlm/sft/full/train.yaml: passed in 19.9s (16.9 GB)
- recipes/vision/smolvlm/sft/lora/train.yaml: passed in 20.2s (4.1 GB)

## Optimization Suggestions (248)

- performance: 194
- efficiency: 37
- best_practice: 17

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
- git_commit: ff6314d0
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

- classify: 12.57s
- static: 42.23s
- hub: 0.01s
- tier0: 244.78s
- vram: 15.39s
- dry_run: 2386.35s
