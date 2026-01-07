#!/bin/bash
# Run all enterprise evals serially with organized output
#
# Usage:
#   ./scripts/enterprise/run_all_evals.sh <model_name> <data_dir>
#
# Example:
#   ./scripts/enterprise/run_all_evals.sh "Qwen/Qwen3-4B-Instruct-2507" /tmp/tim/data/ent

set -e

MODEL_NAME="${1:-Qwen/Qwen3-4B-Instruct-2507}"
DATA_DIR="${2:-data/enterprise}"

# Create run ID from timestamp + sanitized model name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(echo "$MODEL_NAME" | sed 's|.*/||' | sed 's|[^a-zA-Z0-9_-]|_|g')
RUN_ID="${TIMESTAMP}_${MODEL_SHORT}"
RUN_DIR="output/enterprise/evaluation/${RUN_ID}"

echo "=============================================="
echo "Running enterprise evals"
echo "  Model: $MODEL_NAME"
echo "  Data:  $DATA_DIR"
echo "  Run:   $RUN_ID"
echo "  Output: $RUN_DIR"
echo "=============================================="
echo ""

mkdir -p "$RUN_DIR"

# --- PubMedQA ---
echo "[1/5] Running PubMedQA..."
oumi evaluate -c configs/enterprise/evaluation/task_pubmedqa.yaml \
  --model.model_name "$MODEL_NAME" \
  --tasks.0.eval_kwargs.test_data_path "${DATA_DIR}/pubmedqa/test.jsonl" \
  --output_dir "${RUN_DIR}/pubmedqa" \
  --enable_wandb false
echo ""

# --- Banking77 ---
echo "[2/5] Running Banking77..."
oumi evaluate -c configs/enterprise/evaluation/task_banking77.yaml \
  --model.model_name "$MODEL_NAME" \
  --tasks.0.eval_kwargs.test_data_path "${DATA_DIR}/banking77/test.jsonl" \
  --output_dir "${RUN_DIR}/banking77" \
  --enable_wandb false
echo ""

# --- TAT-QA ---
# NB running tatqa with 200 samples for faster iteration while developing workflows...
echo "[3/5] Running TAT-QA..."
oumi evaluate -c configs/enterprise/evaluation/task_tatqa.yaml \
  --model.model_name "$MODEL_NAME" \
  --tasks.0.eval_kwargs.test_data_path "${DATA_DIR}/tatqa/test.jsonl" \
  --tasks.0.num_samples 200 \
  --output_dir "${RUN_DIR}/tatqa" \
  --enable_wandb false
echo ""

# --- NL2SQL ---
echo "[4/5] Running NL2SQL..."
oumi evaluate -c configs/enterprise/evaluation/task_nl2sql.yaml \
  --model.model_name "$MODEL_NAME" \
  --tasks.0.eval_kwargs.test_data_path "${DATA_DIR}/nl2sql/test.jsonl" \
  --output_dir "${RUN_DIR}/nl2sql" \
  --enable_wandb false
echo ""

# --- Control Evals (IFEval + SimpleSafetyTests) ---
echo "[5/5] Running Control Evals..."
oumi evaluate -c configs/enterprise/evaluation/control_evals.yaml \
  --model.model_name "$MODEL_NAME" \
  --output_dir "${RUN_DIR}/control" \
  --enable_wandb false
echo ""

# --- Summary ---
echo "=============================================="
echo "ALL EVALS COMPLETE!"
echo "=============================================="
echo ""
echo "Output directory: $RUN_DIR"
echo ""
ls -la "$RUN_DIR"
echo ""
echo "To copy to local:"
echo "  kubectl cp \$NAMESPACE/<pod>:$(pwd)/${RUN_DIR} ./${RUN_ID}"
