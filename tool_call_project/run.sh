#!/bin/bash
# Run inference for tool call evaluation.
# Each model runs on a separate GPU in parallel.

cd "$(dirname "$0")"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# CUDA_VISIBLE_DEVICES=0 python run_inference.py \
#   --input_file $DATA \
#   --output_file output/smollm2_135m_preds.jsonl \
#   --inference_config configs/smollm2_135m.yaml \
#   2>&1 | tee infer_logs/smollm2_135m.log &

# CUDA_VISIBLE_DEVICES=1 python run_inference.py \
#   --input_file $DATA \
#   --output_file output/qwen2.5_1.5b_preds.jsonl \
#   --inference_config configs/qwen2.5_1.5b.yaml \
#   2>&1 | tee infer_logs/qwen2.5_1.5b.log &

# CUDA_VISIBLE_DEVICES=2 python run_inference.py \
#   --input_file $DATA \
#   --output_file output/llama3.1_8b_preds.jsonl \
#   --inference_config configs/llama3.1_8b_instruct.yaml \
#   2>&1 | tee infer_logs/llama3.1_8b.log &

# CUDA_VISIBLE_DEVICES=3 python run_inference.py \
#   --input_file $DATA \
#   --output_file output/smollm2_135m_sft_preds.jsonl \
#   --inference_config configs/smollm2_135m_sft.yaml \
#   2>&1 | tee infer_logs/smollm2_135m_sft.log &


# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_inference.py \
#   --input_file $DATA \
#   --output_file output/llama3.3_70b_preds.jsonl \
#   --inference_config configs/llama3.3_70b.yaml \
#   2>&1 | tee infer_logs/llama3.3_70b.log &

# python run_inference.py \
#   --input_file $DATA \
#   --output_file output/gpt5_mini_preds.jsonl \
#   --inference_config configs/gpt5_mini.yaml \
#   2>&1 | tee infer_logs/gpt5_mini.log &

# CUDA_VISIBLE_DEVICES=0 python run_inference.py \
#   --input_file $DATA \
#   --output_file output/qwen2.5_1.5b_sft_preds.jsonl \
#   --inference_config infer_configs/qwen2.5_1.5b_sft.yaml \
#   2>&1 | tee infer_logs/qwen2.5_1.5b_sft.log &

# CUDA_VISIBLE_DEVICES=0 python run_inference.py \
#   --input_file data/hermes_reasoning_tool_use_test_split_tool_calls_only.jsonl \
#   --output_file output/hermes_llama3.1_8b_preds.jsonl \
#   --inference_config infer_configs/hermes_llama3.1_8b.yaml \
#   2>&1 | tee infer_logs/hermes_llama3.1_8b.log &

# CUDA_VISIBLE_DEVICES=1 python run_inference.py \
#   --input_file data/tatqa_test.jsonl \
#   --output_file output/tatqa_llama3.1_8b_preds.jsonl \
#   --inference_config infer_configs/tatqa_llama3.1_8b.yaml \
#   2>&1 | tee infer_logs/tatqa_llama3.1_8b.log &

# CUDA_VISIBLE_DEVICES=2 python run_inference.py \
#   --input_file data/hermes_reasoning_tool_use_test_split_tool_calls_only.jsonl \
#   --output_file output/hermes_llama3.1_8b_base_preds.jsonl \
#   --inference_config infer_configs/llama3.1_8b.yaml \
#   2>&1 | tee infer_logs/hermes_llama3.1_8b_base.log &

# CUDA_VISIBLE_DEVICES=3 python run_inference.py \
#   --input_file data/tatqa_test.jsonl \
#   --output_file output/tatqa_llama3.1_8b_base_preds.jsonl \
#   --inference_config infer_configs/llama3.1_8b.yaml \
#   2>&1 | tee infer_logs/tatqa_llama3.1_8b_base.log

# wait

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_inference.py \
#   --input_file data/hermes_reasoning_tool_use_test_split_tool_calls_only.jsonl \
#   --output_file output/hermes_llama3.3_70b_base_preds.jsonl \
#   --inference_config infer_configs/llama3.3_70b.yaml \
#   2>&1 | tee infer_logs/hermes_llama3.3_70b_base.log &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python run_inference.py \
#   --input_file data/tatqa_test.jsonl \
#   --output_file output/tatqa_llama3.3_70b_base_preds.jsonl \
#   --inference_config infer_configs/llama3.3_70b.yaml \
#   2>&1 | tee infer_logs/tatqa_llama3.3_70b_base.log &


CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  --input_file data/hermes_reasoning_tool_use_test_split_tool_calls_only.jsonl \
  --output_file output/hermes_llama3.1_8b_oldcollator_preds.jsonl \
  --inference_config infer_configs/hermes_llama3.1_8b_oldcollator.yaml \
  2>&1 | tee infer_logs/hermes_llama3.1_8b_oldcollator.log &

CUDA_VISIBLE_DEVICES=1 python run_inference.py \
  --input_file data/tatqa_test.jsonl \
  --output_file output/tatqa_llama3.1_8b_oldcollator_preds.jsonl \
  --inference_config infer_configs/tatqa_llama3.1_8b_oldcollator.yaml \
  2>&1 | tee infer_logs/tatqa_llama3.1_8b_oldcollator.log &

wait
echo "All inference jobs complete."
