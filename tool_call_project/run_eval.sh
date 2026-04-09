#!/bin/bash
# Run inference and evaluation for tool call correctness.
#
# Usage:
#   bash tool_call_project/run_eval.sh
#
# To run a subset for quick testing, set NUM_SAMPLES:
#   NUM_SAMPLES=100 bash tool_call_project/run_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/output"
CONFIG_DIR="${SCRIPT_DIR}/configs"
TEST_DATA="${DATA_DIR}/hermes_reasoning_tool_use_test_split_tool_calls_only.jsonl"

NUM_SAMPLES="${NUM_SAMPLES:-}"
SAMPLE_ARGS=""
if [ -n "$NUM_SAMPLES" ]; then
    SAMPLE_ARGS="--num_samples $NUM_SAMPLES"
    echo "Running with $NUM_SAMPLES samples"
fi

mkdir -p "$OUTPUT_DIR"

# Define models to evaluate: config_name -> output_name
declare -A MODELS=(
    ["llama3.1_8b_instruct"]="llama3.1_8b"
    ["qwen3_4b_instruct"]="qwen3_4b"
    # ["gpt4o_mini"]="gpt4o_mini"
    # ["deepseek_v3"]="deepseek_v3"
)

for config in "${!MODELS[@]}"; do
    name="${MODELS[$config]}"
    pred_file="${OUTPUT_DIR}/${name}_preds.jsonl"
    config_file="${CONFIG_DIR}/${config}.yaml"

    echo "=============================================="
    echo "Running: $name"
    echo "Config:  $config_file"
    echo "Output:  $pred_file"
    echo "=============================================="

    # Inference
    python "${SCRIPT_DIR}/run_inference.py" \
        --input_file "$TEST_DATA" \
        --output_file "$pred_file" \
        --inference_config "$config_file" \
        $SAMPLE_ARGS

    # Evaluate
    python "${SCRIPT_DIR}/eval_tool_calls.py" \
        --dataset_path "$TEST_DATA" \
        --predictions_path "$pred_file" \
        $SAMPLE_ARGS

    echo ""
done
