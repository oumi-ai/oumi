#!/bin/bash

# Mlflow environment variables
export DATABRICKS_HOST="xxx"
export DATABRICKS_TOKEN="xxx"
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_EXPERIMENT_ID="1360406024681671"
export MLFLOW_RUN_NAME="oumi_enterprise_experiments"

# Run inference with NativeTextInferenceEngine (supports LoRA on all layers including unembed_tokens)
CUDA_VISIBLE_DEVICES=0 python run_native.py --input_file tatqa_data/test_modified.jsonl --output_file output/tinker_llama3.1_8b_instruct_tatqa.jsonl --inference_config infer_configs/tinker_llama3.1_8b_instruct.yaml | 2>&1 tee infer_logs/tinker_llama3.1_8b_instruct_tatqa.log &
CUDA_VISIBLE_DEVICES=1 python run_native.py --input_file tatqa_data/test_modified.jsonl --output_file output/tinker_qwen3_4b_instruct_tatqa.jsonl --inference_config infer_configs/tinker_qwen3_4b_instruct.yaml | 2>&1 tee infer_logs/tinker_qwen3_4b_instruct_tatqa.log &
wait
