# Mlflow environment variables
export DATABRICKS_HOST="xxx"
export DATABRICKS_TOKEN="xxx"
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_EXPERIMENT_ID="1360406024681671"
export MLFLOW_RUN_NAME="oumi_enterprise_experiments"

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/1k_arxiv_val.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen3_8b_arxiv_val.json --inference_config infer_configs/qwen3_8b_arxiv.yaml --num_samples 10

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/multilingual_thinking_val.json --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen3_8b_multilingual_val.json --inference_config infer_configs/qwen3_8b_multilingual.yaml --num_samples 10 &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/multilingual_thinking_val.json --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen3_8b_tuned_multilingual_val.json --inference_config infer_configs/qwen3_8b_multilingual.yaml --num_samples 10 &

# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/docqa_1000_val.json --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen3_8b_docqa_val.json --inference_config infer_configs/qwen3_8b.yaml --num_samples 10 &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file data/docqa_1000_val.json --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen3_8b_tuned_docqa_val.json --inference_config infer_configs/qwen3_8b_docqa.yaml --num_samples 10

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/docqa_1000_val.json --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_docqa_val.json --inference_config infer_configs/qwen2.5_7b.yaml --num_samples 10 &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/docqa_1000_val.json --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tuned_docqa_val.json --inference_config infer_configs/qwen2.5_7b_docqa.yaml --num_samples 10 &

# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/1k_arxiv_val.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_arxiv_val.json --inference_config infer_configs/qwen2.5_7b.yaml --num_samples 10 &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file data/1k_arxiv_val.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tuned_arxiv_val.json --inference_config infer_configs/qwen2.5_7b_arxiv.yaml --num_samples 10

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen3_8b_tatqa_test_baseline_nothink.json --inference_config infer_configs/qwen3_8b.yaml &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen3_8b_tatqa_test_ep1_nothink.json --inference_config infer_configs/qwen3_8b_tatqa_ep1.yaml &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen3_8b_tatqa_test_ep2_nothink.json --inference_config infer_configs/qwen3_8b_tatqa_ep2.yaml &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen3_8b_tatqa_test_ep3_nothink.json --inference_config infer_configs/qwen3_8b_tatqa_ep3.yaml
# wait

CUDA_VISIBLE_DEVICES=4 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tatqa_test_baseline.json --inference_config infer_configs/qwen2.5_7b.yaml &
CUDA_VISIBLE_DEVICES=5 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tatqa_test_ep1.json --inference_config infer_configs/qwen2.5_7b_tatqa_ep1.yaml &
CUDA_VISIBLE_DEVICES=6 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tatqa_test_ep2.json --inference_config infer_configs/qwen2.5_7b_tatqa_ep2.yaml &
CUDA_VISIBLE_DEVICES=7 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tatqa_test_ep3.json --inference_config infer_configs/qwen2.5_7b_tatqa_ep3.yaml
wait
