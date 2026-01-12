#!/bin/bash
# Run all enterprise evals serially with organized output
#
# Usage: ./scripts/enterprise/run_all_evals.sh [--model-name <model>] [--data-dir <dir>] [--output-dir <dir>]

set -e

# Defaults
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
DATA_DIR="data/enterprise"
OUTPUT_BASE="output/enterprise/evaluation"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--model-name <model>] [--data-dir <dir>] [--output-dir <dir>]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--model-name <model>] [--data-dir <dir>] [--output-dir <dir>]"
      exit 1
      ;;
  esac
done

# Create run ID from timestamp + sanitized model name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(echo "$MODEL_NAME" | sed 's|.*/||' | sed 's|[^a-zA-Z0-9_-]|_|g')
RUN_ID="${TIMESTAMP}_${MODEL_SHORT}"
RUN_DIR="${OUTPUT_BASE}/${RUN_ID}"

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
  --inference_engine VLLM \
  --output_dir "${RUN_DIR}/pubmedqa" \
  --enable_wandb true
echo ""

# force cleanup of GPU memory from vLLM after each eval to avoid OOM
# (especially problematic for gemma)
# pkill -9 -f vllm || true && sleep 5

# --- Banking77 ---
echo "[2/5] Running Banking77..."
oumi evaluate -c configs/enterprise/evaluation/task_banking77.yaml \
  --model.model_name "$MODEL_NAME" \
  --tasks.0.eval_kwargs.test_data_path "${DATA_DIR}/banking77/test.jsonl" \
  --inference_engine VLLM \
  --output_dir "${RUN_DIR}/banking77" \
  --enable_wandb true
echo ""

# --- TAT-QA ---
# NB running tatqa with 200 samples for faster iteration while developing workflows...
echo "[3/5] Running TAT-QA..."
oumi evaluate -c configs/enterprise/evaluation/task_tatqa.yaml \
  --model.model_name "$MODEL_NAME" \
  --tasks.0.eval_kwargs.test_data_path "${DATA_DIR}/tatqa/test.jsonl" \
  --inference_engine VLLM \
  --output_dir "${RUN_DIR}/tatqa" \
  --enable_wandb true
echo ""

# --- NL2SQL ---
echo "[4/5] Running NL2SQL..."
oumi evaluate -c configs/enterprise/evaluation/task_nl2sql.yaml \
  --model.model_name "$MODEL_NAME" \
  --tasks.0.eval_kwargs.test_data_path "${DATA_DIR}/nl2sql/test.jsonl" \
  --inference_engine VLLM \
  --output_dir "${RUN_DIR}/nl2sql" \
  --enable_wandb true
echo ""

# --- Control Evals (IFEval + SimpleSafetyTests) ---
echo "[5/5] Running Control Evals..."
oumi evaluate -c configs/enterprise/evaluation/control_evals.yaml \
  --model.model_name "$MODEL_NAME" \
  --inference_engine VLLM \
  --output_dir "${RUN_DIR}/control" \
  --enable_wandb true
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
echo "  kubectl cp \$NAMESPACE/<pod>:${RUN_DIR} ./${RUN_ID}"
