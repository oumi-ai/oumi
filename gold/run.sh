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

# CUDA_VISIBLE_DEVICES=4 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tatqa_test_baseline.json --inference_config infer_configs/qwen2.5_7b.yaml &
# CUDA_VISIBLE_DEVICES=5 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tatqa_test_ep1.json --inference_config infer_configs/qwen2.5_7b_tatqa_ep1.yaml &
# CUDA_VISIBLE_DEVICES=6 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tatqa_test_ep2.json --inference_config infer_configs/qwen2.5_7b_tatqa_ep2.yaml &
# CUDA_VISIBLE_DEVICES=7 python run.py --input_file data/tatqa_test.jsonl --output_file /home/shanghong/oumi/enterprise_experiments/output/qwen2.5_7b_tatqa_test_ep3.json --inference_config infer_configs/qwen2.5_7b_tatqa_ep3.yaml
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/tatqa_test.jsonl --output_file output/qwen3_8b_tatqa_think2_ep1.json --inference_config infer_configs/qwen3_8b_tatqa_think2_ep1.yaml &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/tatqa_test.jsonl --output_file output/qwen3_8b_tatqa_think2_ep2.json --inference_config infer_configs/qwen3_8b_tatqa_think2_ep2.yaml &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/tatqa_test.jsonl --output_file output/qwen3_8b_tatqa_think2_ep3.json --inference_config infer_configs/qwen3_8b_tatqa_think2_ep3.yaml &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/tatqa_test.jsonl --output_file output/qwen3_8b_tatqa_think2_ep1.json --inference_config infer_configs/qwen3_8b_tatqa_think2_ep1.yaml &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/tatqa_test.jsonl --output_file output/qwen3_8b_tatqa_think2_ep2.json --inference_config infer_configs/qwen3_8b_tatqa_think2_ep2.yaml &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/tatqa_test.jsonl --output_file output/qwen3_8b_tatqa_think2_ep3.json --inference_config infer_configs/qwen3_8b_tatqa_think2_ep3.yaml &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/test.jsonl --output_file output/qwen3_4b_baseline.json --inference_config infer_configs/qwen3_4b.yaml --chat_format &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_1.5b_baseline.jsonl --inference_config infer_configs/qwen2.5_1.5b.yaml --chat_format &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_7b_baseline.json --inference_config infer_configs/qwen2.5_7b.yaml --chat_format &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_1.5b_ckpt500.json --inference_config infer_configs/qwen2.5_1.5b_ckpt500.yaml --chat_format &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_1.5b_ckpt1000.json --inference_config infer_configs/qwen2.5_1.5b_ckpt1000.yaml --chat_format &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_1.5b_ckpt500.json --inference_config infer_configs/qwen2.5_1.5b_ckpt500.yaml --chat_format | 2>&1 tee infer_logs/qwen2.5_1.5b_ckpt500.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_1.5b_ckpt1000.json --inference_config infer_configs/qwen2.5_1.5b_ckpt1000.yaml --chat_format | 2>&1 tee infer_logs/qwen2.5_1.5b_ckpt1000.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_1.5b_ckpt2000.json --inference_config infer_configs/qwen2.5_1.5b_ckpt2000.yaml --chat_format | 2>&1 tee infer_logs/qwen2.5_1.5b_ckpt2000.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_1.5b_ckpt3000.json --inference_config infer_configs/qwen2.5_1.5b_ckpt3000.yaml --chat_format | 2>&1 tee infer_logs/qwen2.5_1.5b_ckpt3000.log &
# CUDA_VISIBLE_DEVICES=4 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_1.5b_ckpt1500.json --inference_config infer_configs/qwen2.5_1.5b_ckpt1500.yaml --chat_format | 2>&1 tee infer_logs/qwen2.5_1.5b_ckpt4000.log &
# CUDA_VISIBLE_DEVICES=4 python run.py --input_file data/test.jsonl --output_file output/qwen2.5_1.5b_ckpt1700.json --inference_config infer_configs/qwen2.5_1.5b_ckpt1700.yaml --chat_format | 2>&1 tee infer_logs/qwen2.5_1.5b_ckpt1700.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/tatqa_train.jsonl --output_file output/tatqa_train.json --inference_config infer_configs/qwen3_4b.yaml --chat_format | 2>&1 tee infer_logs/tatqa_train.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda0_ckpt400.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda0_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda0_ckpt400.log &
# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda0_ckpt200.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda0_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda0_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda0_ckpt799.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda0_ckpt799.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda0_ckpt799.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen3_4b.jsonl --inference_config infer_configs/qwen3_4b.yaml | 2>&1 tee infer_logs/tatqa_qwen3_4b.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen2.5_1.5b.jsonl --inference_config infer_configs/qwen2.5_1.5b.yaml | 2>&1 tee infer_logs/tatqa_qwen2.5_1.5b.log &
# wait


# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt200.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt400.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt400.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt800.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt800.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt800.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt200.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt200.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt400.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt400.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt800.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt800.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt800.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt1100.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt1100.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda0.5_ckpt1100.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa/test_modified.jsonl --output_file output/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt1100.jsonl --inference_config infer_configs/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt1100.yaml | 2>&1 tee infer_logs/tatqa_qwen25_15b_qwen34b_lambda1.0_ckpt1100.log &
# wait

CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa/test_modified.jsonl --output_file output/qwen2.5_1.5b.sft_disable_completions_only_ckpt200.jsonl --inference_config infer_configs/qwen2.5_1.5b.sft_disable_completions_only_ckpt200.yaml | 2>&1 tee infer_logs/qwen2.5_1.5b.sft_disable_completions_only_ckpt200.log &
wait
