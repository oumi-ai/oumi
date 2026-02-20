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

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/test.jsonl --output_file output/gold_llama32_1b_qwen34b_cross_ckpt200.jsonl --inference_config infer_configs/gold_llama32_1b_qwen34b_cross_ckpt200.yaml | 2>&1 tee infer_logs/gold_llama32_1b_qwen34b_cross_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/test.jsonl --output_file output/gold_llama32_1b_qwen34b_cross_ckpt400.jsonl --inference_config infer_configs/gold_llama32_1b_qwen34b_cross_ckpt400.yaml | 2>&1 tee infer_logs/gold_llama32_1b_qwen34b_cross_ckpt400.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/test.jsonl --output_file output/gold_llama32_1b_qwen34b_cross_ckpt800.jsonl --inference_config infer_configs/gold_llama32_1b_qwen34b_cross_ckpt800.yaml | 2>&1 tee infer_logs/gold_llama32_1b_qwen34b_cross_ckpt800.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file data/test.jsonl --output_file output/gold_llama32_1b_qwen34b_cross_ckpt1600.jsonl --inference_config infer_configs/gold_llama32_1b_qwen34b_cross_ckpt1600.yaml | 2>&1 tee infer_logs/gold_llama32_1b_qwen34b_cross_ckpt1600.log &
# CUDA_VISIBLE_DEVICES=4 python run.py --input_file data/test.jsonl --output_file output/gold_llama32_1b_qwen34b_cross_ckpt3200.jsonl --inference_config infer_configs/gold_llama32_1b_qwen34b_cross_ckpt3200.yaml | 2>&1 tee infer_logs/gold_llama32_1b_qwen34b_cross_ckpt3200.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt200.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt200.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt400.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt400.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt400.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt800.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt800.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt800.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt1600.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt1600.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt1600.log &
# CUDA_VISIBLE_DEVICES=4 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt3200.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt3200.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt3200.log &
# CUDA_VISIBLE_DEVICES=5 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt6400.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt6400.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt6400.log &
# CUDA_VISIBLE_DEVICES=6 python run.py --input_file data/test.jsonl --output_file output/grpo_qwen34b_ckpt12000.jsonl --inference_config infer_configs/grpo_qwen34b_ckpt12000.yaml | 2>&1 tee infer_logs/grpo_qwen34b_ckpt12000.log &

# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/test.jsonl --output_file output/countdown_qwen3_8b_baseline.jsonl --inference_config infer_configs/qwen3_8b.yaml | 2>&1 tee infer_logs/countdown_qwen3_8b_baseline.log &
# CUDA_VISIBLE_DEVICES=1,2 python run.py --input_file data/test.jsonl --output_file output/countdown_qwen3_30b_instruct_baseline.jsonl --inference_config infer_configs/qwen3_30b_instruct.yaml | 2>&1 tee infer_logs/countdown_qwen3_30b_instruct_baseline.log &
# CUDA_VISIBLE_DEVICES=3,4 python run.py --input_file data/test.jsonl --output_file output/countdown_qwen3_32b_baseline.jsonl --inference_config infer_configs/qwen3_32b.yaml | 2>&1 tee infer_logs/countdown_qwen3_32b_baseline.log &

# wait

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py --input_file data/test.jsonl --output_file output/countdown_qwen3_235b_instruct_baseline.jsonl --inference_config infer_configs/qwen3_235b_instruct.yaml | 2>&1 tee infer_logs/countdown_qwen3_235b_instruct_baseline.log &

# wait

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/qwen3_235b_instruct_baseline_tatqa.jsonl --inference_config infer_configs/qwen3_235b_instruct.yaml | 2>&1 tee infer_logs/qwen3_235b_instruct_baseline_tatqa.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/gold_llama3.1_8b_lambda0.0_ckpt200.jsonl --inference_config infer_configs/gold_llama3.1_8b_lambda0.0_ckpt200.yaml | 2>&1 tee infer_logs/gold_llama3.1_8b_lambda0.0_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/gold_llama3.1_8b_lambda0.0_ckpt400.jsonl --inference_config infer_configs/gold_llama3.1_8b_lambda0.0_ckpt400.yaml | 2>&1 tee infer_logs/gold_llama3.1_8b_lambda0.0_ckpt400.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/gold_llama3.1_8b_lambda0.0_ckpt800.jsonl --inference_config infer_configs/gold_llama3.1_8b_lambda0.0_ckpt800.yaml | 2>&1 tee infer_logs/gold_llama3.1_8b_lambda0.0_ckpt800.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/gold_llama3.1_8b_lambda0.0_ckpt1598.jsonl --inference_config infer_configs/gold_llama3.1_8b_lambda0.0_ckpt1598.yaml | 2>&1 tee infer_logs/gold_llama3.1_8b_lambda0.0_ckpt1598.log &
# wait


# CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --input_file tatqa_data/train_modified.jsonl --output_file tatqa_data/train_final_llama3.3_70b_instruct.jsonl --inference_config infer_configs/llama3.3_70b_instruct.yaml | 2>&1 tee infer_logs/tatqa_train_modified_llama3.3_70b_instruct.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_sft.jsonl --inference_config infer_configs/tatqa_llama_sft.yaml | 2>&1 tee infer_logs/tatqa_llama_sft.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_sft.jsonl --inference_config infer_configs/tatqa_llama_sft.yaml | 2>&1 tee infer_logs/tatqa_llama_sft.log &

# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.0_2epoch.jsonl --inference_config infer_configs/tatqa_llama_lambda0.0_2epoch.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.0_2epoch.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_2epoch.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_2epoch.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_2epoch.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda1.0_2epoch.jsonl --inference_config infer_configs/tatqa_llama_lambda1.0_2epoch.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda1.0_2epoch.log &

# CUDA_VISIBLE_DEVICES=4 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_ckpt200.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_ckpt200.log &
# CUDA_VISIBLE_DEVICES=5 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_ckpt400.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_ckpt400.log &
# CUDA_VISIBLE_DEVICES=6 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_ckpt819.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_ckpt819.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_ckpt819.log &

# CUDA_VISIBLE_DEVICES=7 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_lora_ckpt200.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_lora_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_lora_ckpt200.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_lora_ckpt400.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_lora_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_lora_ckpt400.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_lora_ckpt819.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_lora_ckpt819.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_lora_ckpt819.log &
# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_lora.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_lora.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_lora.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tinker_llama3.1_8b_ckpt6200.jsonl --inference_config infer_configs/tinker_llama3.1_8b_ckpt6200.yaml | 2>&1 tee infer_logs/tinker_llama3.1_8b_ckpt6200.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_inversekl_lora_ckpt200.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_inversekl_lora_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_inversekl_lora_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_inversekl_lora_ckpt400.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_inversekl_lora_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_inversekl_lora_ckpt400.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_inversekl_lora_ckpt815.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_inversekl_lora_ckpt815.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_inversekl_lora_ckpt815.log &

# CUDA_VISIBLE_DEVICES=3 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_lora_ckpt200.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_lora_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_lora_ckpt200.log &
# CUDA_VISIBLE_DEVICES=4 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_lora_ckpt400.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_lora_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_lora_ckpt400.log &
# CUDA_VISIBLE_DEVICES=5 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_1epoch_lora_ckpt819.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_1epoch_lora_ckpt819.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_1epoch_lora_ckpt819.log &

# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_inversekl_uld_ckpt200.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_inversekl_uld_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_inversekl_uld_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_inversekl_uld_ckpt400.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_inversekl_uld_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_inversekl_uld_ckpt400.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_inversekl_uld_ckpt800.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_inversekl_uld_ckpt800.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_inversekl_uld_ckpt800.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_inversekl_uld_ckpt1600.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_inversekl_uld_ckpt1600.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_inversekl_uld_ckpt1600.log &

# CUDA_VISIBLE_DEVICES=4 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_inversekl_jsd_ckpt200.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_inversekl_jsd_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_inversekl_jsd_ckpt200.log &
# CUDA_VISIBLE_DEVICES=6 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_inversekl_jsd_ckpt400.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_inversekl_jsd_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_inversekl_jsd_ckpt400.log &
# CUDA_VISIBLE_DEVICES=7 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_llama_lambda0.5_inversekl_jsd_ckpt800.jsonl --inference_config infer_configs/tatqa_llama_lambda0.5_inversekl_jsd_ckpt800.yaml | 2>&1 tee infer_logs/tatqa_llama_lambda0.5_inversekl_jsd_ckpt800.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/test.jsonl --output_file output/countdown_llama3.2_1b_instruct_baseline.jsonl --inference_config infer_configs/llama3.2_1b_instruct.yaml | 2>&1 tee infer_logs/countdown_llama3.2_1b_instruct_baseline.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file data/test.jsonl --output_file output/gold_llama32_1b_qwen34b_cross_ckpt4600.jsonl --inference_config infer_configs/gold_llama32_1b_qwen34b_cross_ckpt4600.yaml | 2>&1 tee infer_logs/gold_llama32_1b_qwen34b_cross_ckpt4600.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_cross_llama3.1_8b_qwen3_4b_ckpt200.jsonl --inference_config infer_configs/tatqa_cross_llama3.1_8b_qwen3_4b_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_cross_llama3.1_8b_qwen3_4b_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_cross_llama3.1_8b_qwen3_4b_ckpt400.jsonl --inference_config infer_configs/tatqa_cross_llama3.1_8b_qwen3_4b_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_cross_llama3.1_8b_qwen3_4b_ckpt400.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_cross_llama3.1_8b_qwen3_4b_lora_ckpt200.jsonl --inference_config infer_configs/tatqa_cross_llama3.1_8b_qwen3_4b_lora_ckpt200.yaml | 2>&1 tee infer_logs/tatqa_cross_llama3.1_8b_qwen3_4b_lora_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_cross_llama3.1_8b_qwen3_4b_lora_ckpt400.jsonl --inference_config infer_configs/tatqa_cross_llama3.1_8b_qwen3_4b_lora_ckpt400.yaml | 2>&1 tee infer_logs/tatqa_cross_llama3.1_8b_qwen3_4b_lora_ckpt400.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_cross_llama3.1_8b_qwen3_4b_lora_ckpt799.jsonl --inference_config infer_configs/tatqa_cross_llama3.1_8b_qwen3_4b_lora_ckpt799.yaml | 2>&1 tee infer_logs/tatqa_cross_llama3.1_8b_qwen3_4b_lora_ckpt799.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file nl2sql_data/test_modified.jsonl --output_file output/nl2sql_llama3.1_8b_sft.jsonl --inference_config infer_configs/nl2sql_llama3.1_8b_sft.yaml | 2>&1 tee infer_logs/nl2sql_llama3.1_8b_sft.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file nl2sql_data/test_modified.jsonl --output_file output/nl2sql_llama_lambda0.5_1epoch_forwardkl_lora.jsonl --inference_config infer_configs/nl2sql_llama_lambda0.5_1epoch_forwardkl_lora.yaml | 2>&1 tee infer_logs/nl2sql_llama_lambda0.5_1epoch_forwardkl_lora.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file nl2sql_data/test_modified.jsonl --output_file output/nl2sql_llama_lambda0.5_1epoch_forwardkl.jsonl --inference_config infer_configs/nl2sql_llama_lambda0.5_1epoch_forwardkl.yaml | 2>&1 tee infer_logs/nl2sql_llama_lambda0.5_1epoch_forwardkl.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file nl2sql_data/test_modified.jsonl --output_file output/nl2sql_llama_lambda0.5_1epoch_inversekl_lora.jsonl --inference_config infer_configs/nl2sql_llama_lambda0.5_1epoch_inversekl_lora.yaml | 2>&1 tee infer_logs/nl2sql_llama_lambda0.5_1epoch_inversekl_lora.log &
# CUDA_VISIBLE_DEVICES=4 python run.py --input_file nl2sql_data/test_modified.jsonl --output_file output/nl2sql_llama_lambda0.5_1epoch_inversekl.jsonl --inference_config infer_configs/nl2sql_llama_lambda0.5_1epoch_inversekl.yaml | 2>&1 tee infer_logs/nl2sql_llama_lambda0.5_1epoch_inversekl.log &
# CUDA_VISIBLE_DEVICES=5 python run.py --input_file nl2sql_data/test_modified.jsonl --output_file output/nl2sql_llama3.1_8b_baseline.jsonl --inference_config infer_configs/llama3.1_8b_instruct.yaml | 2>&1 tee infer_logs/nl2sql_llama3.1_8b_baseline.log &
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --input_file nl2sql_data/test_modified.jsonl --output_file output/nl2sql_llama3.3_70b_baseline.jsonl --inference_config infer_configs/llama3.3_70b_instruct.yaml | 2>&1 tee infer_logs/nl2sql_llama3.3_70b_baseline.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/tatqa_tinker_cross_ckpt6200.jsonl --inference_config infer_configs/tatqa_tinker_cross_ckpt6200.yaml | 2>&1 tee infer_logs/tatqa_tinker_cross_ckpt6200.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_1_beta_only_ckpt200.jsonl --inference_config infer_configs/ablation_1_beta_only_ckpt200.yaml | 2>&1 tee infer_logs/ablation_1_beta_only_ckpt200.log &
# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_1_beta_only_ckpt400.jsonl --inference_config infer_configs/ablation_1_beta_only_ckpt400.yaml | 2>&1 tee infer_logs/ablation_1_beta_only_ckpt400.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_1_beta_only_ckpt799.jsonl --inference_config infer_configs/ablation_1_beta_only_ckpt799.yaml | 2>&1 tee infer_logs/ablation_1_beta_only_ckpt799.log &

# CUDA_VISIBLE_DEVICES=3 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_2_weights_only_ckpt200.jsonl --inference_config infer_configs/ablation_2_weights_only_ckpt200.yaml | 2>&1 tee infer_logs/ablation_2_weights_only_ckpt200.log &
# CUDA_VISIBLE_DEVICES=4 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_2_weights_only_ckpt400.jsonl --inference_config infer_configs/ablation_2_weights_only_ckpt400.yaml | 2>&1 tee infer_logs/ablation_2_weights_only_ckpt400.log &
# CUDA_VISIBLE_DEVICES=5 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_2_weights_only_ckpt799.jsonl --inference_config infer_configs/ablation_2_weights_only_ckpt799.yaml | 2>&1 tee infer_logs/ablation_2_weights_only_ckpt799.log &

# CUDA_VISIBLE_DEVICES=6 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_3_full_fix_ckpt200.jsonl --inference_config infer_configs/ablation_3_full_fix_ckpt200.yaml | 2>&1 tee infer_logs/ablation_3_full_fix_ckpt200.log &
# CUDA_VISIBLE_DEVICES=7 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_3_full_fix_ckpt400.jsonl --inference_config infer_configs/ablation_3_full_fix_ckpt400.yaml | 2>&1 tee infer_logs/ablation_3_full_fix_ckpt400.log &
# wait

# CUDA_VISIBLE_DEVICES=0 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_3_full_fix_ckpt799.jsonl --inference_config infer_configs/ablation_3_full_fix_ckpt799.yaml | 2>&1 tee infer_logs/ablation_3_full_fix_ckpt799.log &

# CUDA_VISIBLE_DEVICES=1 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_4_no_hybrid_ckpt200.jsonl --inference_config infer_configs/ablation_4_no_hybrid_ckpt200.yaml | 2>&1 tee infer_logs/ablation_4_no_hybrid_ckpt200.log &
# CUDA_VISIBLE_DEVICES=2 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_4_no_hybrid_ckpt400.jsonl --inference_config infer_configs/ablation_4_no_hybrid_ckpt400.yaml | 2>&1 tee infer_logs/ablation_4_no_hybrid_ckpt400.log &
# CUDA_VISIBLE_DEVICES=3 python run.py --input_file tatqa_data/test_modified.jsonl --output_file output/ablation_4_no_hybrid_ckpt799.jsonl --inference_config infer_configs/ablation_4_no_hybrid_ckpt799.yaml | 2>&1 tee infer_logs/ablation_4_no_hybrid_ckpt799.log &
# wait

CUDA_VISIBLE_DEVICES=0 python run.py --input_file data/test.jsonl --output_file output/countdown_qwen2.5_1.5b_sft.jsonl --inference_config infer_configs/countdown_qwen2.5_1.5b_sft.yaml | 2>&1 tee infer_logs/countdown_qwen2.5_1.5b_sft.log &
wait