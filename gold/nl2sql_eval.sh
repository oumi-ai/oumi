python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen2.5_1.5b_baseline.jsonl \
    --out nl2sql_results/nl2sql_qwen2.5_1.5b_baseline_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen2.5_1.5b_baseline_summary.json

python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen3_4b_baseline.jsonl \
    --out nl2sql_results/nl2sql_qwen3_4b_baseline_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen3_4b_baseline_summary.json

python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen25_15b_qwen34b_lambda0_ckpt200.jsonl \
    --out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0_ckpt200_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0_ckpt200_summary.json

python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen25_15b_qwen34b_lambda0_ckpt400.jsonl \
    --out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0_ckpt400_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0_ckpt400_summary.json
    
python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen25_15b_qwen34b_lambda0_ckpt800.jsonl \
    --out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0_ckpt800_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0_ckpt800_summary.json

python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen25_15b_qwen34b_lambda0.5_ckpt200.jsonl \
    --out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0.5_ckpt200_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0.5_ckpt200_summary.json

python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen25_15b_qwen34b_lambda0.5_ckpt400.jsonl \
    --out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0.5_ckpt400_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0.5_ckpt400_summary.json
    
python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen25_15b_qwen34b_lambda0.5_ckpt800.jsonl \
    --out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0.5_ckpt800_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda0.5_ckpt800_summary.json

python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen25_15b_qwen34b_lambda1.0_ckpt200.jsonl \
    --out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda1.0_ckpt200_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda1.0_ckpt200_summary.json

python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen25_15b_qwen34b_lambda1.0_ckpt400.jsonl \
    --out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda1.0_ckpt400_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda1.0_ckpt400_summary.json
    
python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_qwen25_15b_qwen34b_lambda1.0_ckpt800.jsonl \
    --out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda1.0_ckpt800_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_qwen25_15b_qwen34b_lambda1.0_ckpt800_summary.json

python3 nlsql_evaluation.py \
    --test_data_path output/nl2sql_sft_ckpt200.jsonl \
    --out nl2sql_results/nl2sql_sft_ckpt200_judged.jsonl \
    --summary-out nl2sql_results/nl2sql_sft_ckpt200_summary.json