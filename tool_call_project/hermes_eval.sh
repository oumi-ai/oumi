#!/bin/bash
# Evaluate tool call predictions for all models.

cd "$(dirname "$0")"

DATA=data/hermes_reasoning_tool_use_test_split_tool_calls_only.jsonl

for pred_file in output/hermes_*_preds.jsonl; do
  name=$(basename "$pred_file" _preds.jsonl)
  echo "=============================================="
  echo "Evaluating: $name"
  echo "=============================================="
  python eval_tool_calls.py \
    --dataset_path $DATA \
    --predictions_path "$pred_file"
  mv data/eval_results.json "output/${name}.scores.json"
  echo ""
done
