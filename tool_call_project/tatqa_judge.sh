#!/bin/bash
set -e

cd "$(dirname "$0")"
mkdir -p eval_results

# python equivalence_judge.py \
#   --ground_truth_file data/tatqa_test.jsonl \
#   --results_file output/tatqa_llama3.1_8b_preds.jsonl \
#   --judge_config_file judge_configs/equivalence_configs.yaml \
#   --use_judge \
#   --score_file eval_results/tatqa_llama3.1_8b_scores.json \
#   --out_jsonl eval_results/tatqa_llama3.1_8b_eval.jsonl \
#   2>&1 | tee eval_results/tatqa_llama3.1_8b_judge.log

# python equivalence_judge.py \
#   --ground_truth_file data/tatqa_test.jsonl \
#   --results_file output/tatqa_llama3.1_8b_base_preds.jsonl \
#   --judge_config_file judge_configs/equivalence_configs.yaml \
#   --use_judge \
#   --score_file eval_results/tatqa_llama3.1_8b_base_scores.json \
#   --out_jsonl eval_results/tatqa_llama3.1_8b_base_eval.jsonl \
#   2>&1 | tee eval_results/tatqa_llama3.1_8b_base_judge.log

# python equivalence_judge.py \
#   --ground_truth_file data/tatqa_test.jsonl \
#   --results_file output/tatqa_llama3.3_70b_base_preds.jsonl \
#   --judge_config_file judge_configs/equivalence_configs.yaml \
#   --use_judge \
#   --score_file eval_results/tatqa_llama3.3_70b_base_scores.json \
#   --out_jsonl eval_results/tatqa_llama3.3_70b_base_eval.jsonl \
#   2>&1 | tee eval_results/tatqa_llama3.3_70b_base_judge.log

python equivalence_judge.py \
  --ground_truth_file data/tatqa_test.jsonl \
  --results_file output/tatqa_llama3.1_8b_oldcollator_preds.jsonl \
  --judge_config_file judge_configs/equivalence_configs.yaml \
  --use_judge \
  --score_file eval_results/tatqa_llama3.1_8b_oldcollator_scores.json \
  --out_jsonl eval_results/tatqa_llama3.1_8b_oldcollator_eval.jsonl \
  2>&1 | tee eval_results/tatqa_llama3.1_8b_oldcollator_judge.log

echo "TatQA judge complete."
echo ""
echo "=== TatQA Qwen2.5-1.5B SFT ==="
cat eval_results/tatqa_llama3.1_8b_scores.json
