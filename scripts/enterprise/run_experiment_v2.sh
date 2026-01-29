#!/bin/bash
# 
# Run a training job and evaluate the resulting model on all target and control benchmarks.
# Record key training hyperparams and dump them to experiment_meta.json in the model output directory.
# Dump all eval results bundles to a timestamped subdir of the eval output directory.
#
# Usage:
#   ./scripts/enterprise/run_experiment_v2.sh \
#     --config configs/enterprise/training/llama32_3b_instruct_train_full.yaml \
#     --task pubmedqa \
#     --data-dir /data/demo/datasets \
#     --checkpoint-dir /data/demo/checkpoints \
#     --eval-dir /data/demo/evals

set -e

# ============================================================================
# Parse arguments
# ============================================================================
CONFIG=""
TASK=""
DATA_DIR=""
CHECKPOINT_DIR=""
EVAL_DIR=""
RUN_SUFFIX=""
NPROC=8
MASTER_PORT=9010
DRY_RUN=false

print_usage() {
    echo "Usage: $0 --config <path> --task <task> --data-dir <path> --checkpoint-dir <path> --eval-dir <path> [options]"
    echo ""
    echo "Required:"
    echo "  --config <path>        Path to training config YAML"
    echo "  --task <task>          Task name (pubmedqa, banking77, tatqa, nl2sql)"
    echo "  --data-dir <path>      Dataset directory (expects <data-dir>/<task>/train.jsonl)"
    echo "  --checkpoint-dir <path> Where to save checkpoints"
    echo "  --eval-dir <path>      Where to save eval results"
    echo ""
    echo "Options:"
    echo "  --suffix <str>     Add suffix to run name (e.g., '-lr1e5')"
    echo "  --nproc <n>        Number of GPUs (default: 8)"
    echo "  --port <n>         Master port (default: 9010)"
    echo "  --dry-run          Print commands without executing"
    echo "  -h, --help         Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --eval-dir)
            EVAL_DIR="$2"
            shift 2
            ;;
        --suffix)
            RUN_SUFFIX="$2"
            shift 2
            ;;
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required args
if [[ -z "$CONFIG" ]] || [[ -z "$TASK" ]] || [[ -z "$DATA_DIR" ]] || [[ -z "$CHECKPOINT_DIR" ]] || [[ -z "$EVAL_DIR" ]]; then
    echo "Error: --config, --task, --data-dir, --checkpoint-dir, and --eval-dir are required"
    print_usage
    exit 1
fi

# Validate config exists
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Validate task data exists
TRAIN_DATASET="${DATA_DIR}/${TASK}/train.jsonl"
VAL_DATASET="${DATA_DIR}/${TASK}/val.jsonl"
if [[ ! -f "$TRAIN_DATASET" ]]; then
    echo "Error: Training data not found: $TRAIN_DATASET"
    exit 1
fi

# ============================================================================
# Extract base model name from config
# ============================================================================

HF_MODEL_NAME=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    config = yaml.safe_load(f)
print(config.get('model', {}).get('model_name', ''))
")

if [[ -z "$HF_MODEL_NAME" ]]; then
    echo "Error: Could not extract model.model_name from config"
    exit 1
fi

# ============================================================================
# Resolve paths
# ============================================================================

# Derive run name from config filename + task
CONFIG_BASENAME=$(basename "$CONFIG" .yaml)
RUN_NAME="${CONFIG_BASENAME}-${TASK}${RUN_SUFFIX}"
OUTPUT_DIR="${CHECKPOINT_DIR}/${RUN_NAME}"

# Sanitize model name for directory (replace / and . with _)
MODEL_SHORT_NAME="${HF_MODEL_NAME##*/}"
MODEL_SHORT_NAME_SANITIZED="${MODEL_SHORT_NAME//[.\/]/_}"
EVAL_OUTPUT_BASE="${EVAL_DIR}/${MODEL_SHORT_NAME_SANITIZED}-ft"

# ============================================================================
# Print plan
# ============================================================================

echo "=============================================="
echo "Enterprise Experiment Runner v2"
echo "=============================================="
echo ""
echo "Model:        $HF_MODEL_NAME"
echo "Task:         $TASK"
echo "Run name:     $RUN_NAME"
echo ""
echo "Config:       $CONFIG"
echo "Train data:   $TRAIN_DATASET"
echo "Val data:     $VAL_DATASET"
echo "Checkpoint:   $OUTPUT_DIR"
echo "Eval output:  $EVAL_OUTPUT_BASE"
echo ""
echo "Options:"
echo "  nproc:         $NPROC"
echo "  dry_run:       $DRY_RUN"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN MODE - Commands will be printed but not executed]"
    echo ""
fi

# ============================================================================
# Helper function to run or print commands
# ============================================================================

run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "$ $@"
        echo ""
    else
        echo "$ $@"
        "$@"
    fi
}

# ============================================================================
# Save experiment metadata to track impact of hyperparam changes
# ============================================================================

save_experiment_metadata() {
    local meta_file="$OUTPUT_DIR/experiment_meta.json"
    
    python3 << EOF
import yaml
import json
from datetime import datetime
from pathlib import Path

config_path = "$CONFIG"
output_path = "$meta_file"

with open(config_path) as f:
    config = yaml.safe_load(f)

training = config.get("training", {})
model = config.get("model", {})

metadata = {
    "run_name": "$RUN_NAME",
    "task": "$TASK",
    "hf_model_name": model.get("model_name", "$HF_MODEL_NAME"),
    "training_config": "$CONFIG",
    "train_dataset": "$TRAIN_DATASET",
    "val_dataset": "$VAL_DATASET",
    "checkpoint_dir": "$OUTPUT_DIR",
    "timestamp": datetime.now().isoformat(),
    "nproc_per_node": $NPROC,
    "learning_rate": training.get("learning_rate"),
    "num_train_epochs": training.get("num_train_epochs"),
    "per_device_train_batch_size": training.get("per_device_train_batch_size"),
    "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
    "weight_decay": training.get("weight_decay"),
    "warmup_ratio": training.get("warmup_ratio"),
    "lr_scheduler_type": training.get("lr_scheduler_type"),
    "optimizer": training.get("optimizer"),
}

if metadata["per_device_train_batch_size"] and metadata["gradient_accumulation_steps"]:
    metadata["effective_batch_size"] = (
        metadata["per_device_train_batch_size"] 
        * metadata["gradient_accumulation_steps"] 
        * $NPROC
    )

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved experiment metadata to {output_path}")
EOF
}

# ============================================================================
# Step 1: Training
# ============================================================================
echo "=============================================="
echo "Step 1: Training"
echo "=============================================="
echo ""

if [[ "$DRY_RUN" != "true" ]]; then
    save_experiment_metadata
fi

run_cmd oumi distributed torchrun \
    --nproc_per_node=$NPROC \
    --master-port=$MASTER_PORT \
    -m oumi train \
    -c "$CONFIG" \
    --data.train.datasets.0.dataset_path="$TRAIN_DATASET" \
    --data.validation.datasets.0.dataset_path="$VAL_DATASET" \
    --training.run_name="$RUN_NAME" \
    --training.output_dir="$OUTPUT_DIR"

echo ""
echo "Training complete! Checkpoint saved to: $OUTPUT_DIR"
echo ""

# ============================================================================
# Step 2: Evaluation
# ============================================================================
echo "=============================================="
echo "Step 2: Evaluation"
echo "=============================================="
echo ""

echo "Evaluating finetuned model: $OUTPUT_DIR"
run_cmd ./scripts/enterprise/run_all_evals.sh \
    --model-name "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$EVAL_OUTPUT_BASE"

echo ""
echo "Evaluation complete! Results saved to: $EVAL_OUTPUT_BASE"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo ""
echo "Checkpoint: $OUTPUT_DIR"
echo "Evals:      $EVAL_OUTPUT_BASE"
