# Mlflow environment variables
export DATABRICKS_HOST="xxx"
export DATABRICKS_TOKEN="xxx"
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_EXPERIMENT_ID="1360406024681671"
export MLFLOW_RUN_NAME="oumi_enterprise_experiments"


CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt200.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt200.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt200.log &
CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt400.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt400.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt400.log &
CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt800.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt800.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt800.log &
CUDA_VISIBLE_DEVICES=3 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt1600.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt1600.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt1600.log &
CUDA_VISIBLE_DEVICES=4 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt3200.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt3200.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt3200.log &
CUDA_VISIBLE_DEVICES=5 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt6400.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt6400.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt6400.log &
CUDA_VISIBLE_DEVICES=6 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt12000.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt12000.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt12000.log &

wait