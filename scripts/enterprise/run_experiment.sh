#!/bin/bash
# Run a complete enterprise experiment: train → eval → collate
#
# Usage:
#   ./scripts/enterprise/run_experiment.sh --model llama32_1b --task pubmedqa
#   ./scripts/enterprise/run_experiment.sh --model llama31_8b --task tatqa --eval-only
#   ./scripts/enterprise/run_experiment.sh --model llama32_3b --task pubmedqa --with-baseline
#
# Environment variables:
#   BASEDIR       - Base code directory (default: /data/tim/code/oumi)
#   CHECKPOINT_DIR - Where to save checkpoints (default: /data/tim/checkpoints)
#   EVAL_DIR      - Where to save eval results (default: /data/tim/evals/ent)

set -e

# ============================================================================
# Configuration
# ============================================================================

BASEDIR="${BASEDIR:-/data/tim/code/oumi}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/tim/checkpoints}"
EVAL_DIR="${EVAL_DIR:-/data/tim/evals/ent}"
DATASET_DIR="${BASEDIR}/data/enterprise"

# Model configs (model_id -> config_path, hf_name)
declare -A MODEL_CONFIGS
MODEL_CONFIGS["llama32_1b"]="configs/enterprise/training/llama32_1b_instruct_train_full.yaml"
MODEL_CONFIGS["llama32_3b"]="configs/enterprise/training/llama32_3b_instruct_train_full.yaml"
MODEL_CONFIGS["llama31_8b"]="configs/enterprise/training/llama31_8b_instruct_train_full.yaml"
MODEL_CONFIGS["gemma3_4b"]="configs/enterprise/training/gemma3_4b_it_train_full.yaml"
MODEL_CONFIGS["qwen3_4b"]="configs/enterprise/training/qwen3_4b_instruct_2507_train_full.yaml"
MODEL_CONFIGS["qwen3_8b"]="configs/enterprise/training/Qwen_Qwen3-8B_fft.yaml"
MODEL_CONFIGS["smollm2_1.7b"]="configs/enterprise/training/HuggingFaceTB_SmolLM2-1.7B-Instruct_fft.yaml"

declare -A MODEL_HF_NAMES
MODEL_HF_NAMES["llama32_1b"]="meta-llama/Llama-3.2-1B-Instruct"
MODEL_HF_NAMES["llama32_3b"]="meta-llama/Llama-3.2-3B-Instruct"
MODEL_HF_NAMES["llama31_8b"]="meta-llama/Llama-3.1-8B-Instruct"
MODEL_HF_NAMES["gemma3_4b"]="google/gemma-3-4b-it"
MODEL_HF_NAMES["qwen3_4b"]="Qwen/Qwen3-4B-Instruct-2507"
MODEL_HF_NAMES["qwen3_8b"]="Qwen/Qwen3-8B"
MODEL_HF_NAMES["smollm2_1.7b"]="HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Task data paths (task_id -> train, val)
declare -A TASK_TRAIN
TASK_TRAIN["pubmedqa"]="data/enterprise/pubmedqa/train.jsonl"
TASK_TRAIN["tatqa"]="data/enterprise/tatqa/train.jsonl"
TASK_TRAIN["banking77"]="data/enterprise/banking77/train.jsonl"
TASK_TRAIN["nl2sql"]="data/enterprise/nl2sql/train.jsonl"

declare -A TASK_VAL
TASK_VAL["pubmedqa"]="data/enterprise/pubmedqa/val.jsonl"
TASK_VAL["tatqa"]="data/enterprise/tatqa/val.jsonl"
TASK_VAL["banking77"]="data/enterprise/banking77/val.jsonl"
TASK_VAL["nl2sql"]="data/enterprise/nl2sql/val.jsonl"

# ============================================================================
# Parse arguments
# ============================================================================

MODEL=""
TASK=""
EVAL_ONLY=false
TRAIN_ONLY=false
WITH_BASELINE=false
RUN_SUFFIX=""
NPROC=8
MASTER_PORT=9010
DRY_RUN=false

print_usage() {
    echo "Usage: $0 --model <model_id> --task <task_id> [options]"
    echo ""
    echo "Models: llama32_1b, llama32_3b, llama31_8b, gemma3_4b, qwen3_4b, qwen3_8b, smollm2_1.7b"
    echo "Tasks:  pubmedqa, tatqa, banking77, nl2sql"
    echo ""
    echo "Options:"
    echo "  --model <id>       Model to train/eval (required)"
    echo "  --task <id>        Task to train on (required)"
    echo "  --eval-only        Skip training, only run evals"
    echo "  --train-only       Skip evals, only run training"
    echo "  --with-baseline    Run baseline eval (collation always includes existing baseline)"
    echo "  --suffix <str>     Add suffix to run name (e.g., '-lr1e5')"
    echo "  --nproc <n>        Number of GPUs (default: 8)"
    echo "  --port <n>         Master port (default: 9010)"
    echo "  --dry-run          Print commands without executing"
    echo "  -h, --help         Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --with-baseline)
            WITH_BASELINE=true
            shift
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
if [[ -z "$MODEL" ]] || [[ -z "$TASK" ]]; then
    echo "Error: --model and --task are required"
    print_usage
    exit 1
fi

# Validate model
if [[ -z "${MODEL_CONFIGS[$MODEL]}" ]]; then
    echo "Error: Unknown model '$MODEL'"
    echo "Valid models: ${!MODEL_CONFIGS[@]}"
    exit 1
fi

# Validate task
if [[ -z "${TASK_TRAIN[$TASK]}" ]]; then
    echo "Error: Unknown task '$TASK'"
    echo "Valid tasks: ${!TASK_TRAIN[@]}"
    exit 1
fi

# ============================================================================
# Resolve paths
# ============================================================================

TRAINING_CONFIG="${BASEDIR}/${MODEL_CONFIGS[$MODEL]}"
TRAIN_DATASET="${BASEDIR}/${TASK_TRAIN[$TASK]}"
VAL_DATASET="${BASEDIR}/${TASK_VAL[$TASK]}"
HF_MODEL_NAME="${MODEL_HF_NAMES[$MODEL]}"

RUN_NAME="${MODEL}-${TASK}${RUN_SUFFIX}"
OUTPUT_DIR="${CHECKPOINT_DIR}/${RUN_NAME}"
EVAL_OUTPUT_BASE="${EVAL_DIR}/${MODEL}-ft"
BASELINE_EVAL_DIR="${EVAL_DIR}/baselines"  # Shared baseline evals directory

# ============================================================================
# Print plan
# ============================================================================

echo "=============================================="
echo "Enterprise Experiment Runner"
echo "=============================================="
echo ""
echo "Model:      $MODEL ($HF_MODEL_NAME)"
echo "Task:       $TASK"
echo "Run name:   $RUN_NAME"
echo ""
echo "Training config:  $TRAINING_CONFIG"
echo "Train data:       $TRAIN_DATASET"
echo "Val data:         $VAL_DATASET"
echo "Checkpoint dir:   $OUTPUT_DIR"
echo "Eval output:      $EVAL_OUTPUT_BASE"
echo "Baselines:        $BASELINE_EVAL_DIR"
echo ""
echo "Options:"
echo "  eval_only:     $EVAL_ONLY"
echo "  train_only:    $TRAIN_ONLY"
echo "  with_baseline: $WITH_BASELINE"
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
# Save experiment metadata (for results tracking)
# ============================================================================

save_experiment_metadata() {
    local meta_file="$OUTPUT_DIR/experiment_meta.json"
    
    # Extract key hyperparameters from training config using Python
    python3 << EOF
import yaml
import json
from datetime import datetime
from pathlib import Path

config_path = "$TRAINING_CONFIG"
output_path = "$meta_file"

# Load training config
with open(config_path) as f:
    config = yaml.safe_load(f)

# Extract key hyperparameters
training = config.get("training", {})
model = config.get("model", {})

metadata = {
    "run_name": "$RUN_NAME",
    "model_id": "$MODEL",
    "task_id": "$TASK",
    "hf_model_name": model.get("model_name", "$HF_MODEL_NAME"),
    "training_config": "$TRAINING_CONFIG",
    "train_dataset": "$TRAIN_DATASET",
    "val_dataset": "$VAL_DATASET",
    "checkpoint_dir": "$OUTPUT_DIR",
    "timestamp": datetime.now().isoformat(),
    "nproc_per_node": $NPROC,
    # Training hyperparameters
    "learning_rate": training.get("learning_rate"),
    "num_train_epochs": training.get("num_train_epochs"),
    "per_device_train_batch_size": training.get("per_device_train_batch_size"),
    "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
    "weight_decay": training.get("weight_decay"),
    "warmup_ratio": training.get("warmup_ratio"),
    "lr_scheduler_type": training.get("lr_scheduler_type"),
    "optimizer": training.get("optimizer"),
}

# Calculate effective batch size
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
# Training
# ============================================================================

if [[ "$EVAL_ONLY" != "true" ]]; then
    echo "=============================================="
    echo "Step 1: Training"
    echo "=============================================="
    echo ""
    
    # Save metadata before training
    if [[ "$DRY_RUN" != "true" ]]; then
        save_experiment_metadata
    fi
    
    run_cmd oumi distributed torchrun \
        --nproc_per_node=$NPROC \
        --master-port=$MASTER_PORT \
        -m oumi train \
        -c "$TRAINING_CONFIG" \
        --data.train.datasets.0.dataset_path="$TRAIN_DATASET" \
        --data.validation.datasets.0.dataset_path="$VAL_DATASET" \
        --training.run_name="$RUN_NAME" \
        --training.output_dir="$OUTPUT_DIR"
    
    echo ""
    echo "Training complete! Checkpoint saved to: $OUTPUT_DIR"
    echo ""
fi

# ============================================================================
# Evaluation
# ============================================================================

if [[ "$TRAIN_ONLY" != "true" ]]; then
    echo "=============================================="
    echo "Step 2: Evaluation"
    echo "=============================================="
    echo ""
    
    # Eval the finetuned model
    echo "Evaluating finetuned model: $OUTPUT_DIR"
    run_cmd ./scripts/enterprise/run_all_evals.sh \
        --model-name "$OUTPUT_DIR" \
        --data-dir "$DATASET_DIR" \
        --output-dir "$EVAL_OUTPUT_BASE"
    
    # Optionally run baseline eval (stored in shared baselines directory)
    # Note: --with-baseline means "run baseline eval", not "include in collation"
    # Collation always includes matching baseline if one exists
    if [[ "$WITH_BASELINE" == "true" ]]; then
        echo ""
        echo "Evaluating baseline model: $HF_MODEL_NAME"
        run_cmd ./scripts/enterprise/run_all_evals.sh \
            --model-name "$HF_MODEL_NAME" \
            --data-dir "$DATASET_DIR" \
            --output-dir "$BASELINE_EVAL_DIR"
    fi
    
    echo ""
    echo "Evaluation complete! Results saved to: $EVAL_OUTPUT_BASE"
    echo ""
    
    # ============================================================================
    # Collation and Plotting
    # ============================================================================
    
    echo "=============================================="
    echo "Step 3: Collate and Plot Results"
    echo "=============================================="
    echo ""
    
    # Collate results (include corresponding baseline if it exists)
    COLLATE_DIRS="$EVAL_OUTPUT_BASE/*"
    # Extract just the model name (after the last /) for matching baseline dirs
    # Also replace dots with underscores since eval script sanitizes directory names
    MODEL_SHORT_NAME="${MODEL_HF_NAMES[$MODEL]##*/}"
    MODEL_SHORT_NAME_SANITIZED="${MODEL_SHORT_NAME//./_}"
    BASELINE_PATTERN="${BASELINE_EVAL_DIR}/*${MODEL_SHORT_NAME_SANITIZED}*"
    BASELINE_MATCHES=($(ls -d $BASELINE_PATTERN 2>/dev/null | sort -r))
    MATCHING_BASELINE="${BASELINE_MATCHES[0]:-}"
    if [[ ${#BASELINE_MATCHES[@]} -gt 1 ]]; then
        echo "Warning: Found ${#BASELINE_MATCHES[@]} matching baselines, using newest: $MATCHING_BASELINE"
    fi
    if [[ -n "$MATCHING_BASELINE" ]]; then
        COLLATE_DIRS="$EVAL_OUTPUT_BASE/* $MATCHING_BASELINE"
        echo "Including baseline: $MATCHING_BASELINE"
    fi
    
    run_cmd python scripts/enterprise/collate_eval_results.py \
        --run-dirs $COLLATE_DIRS \
        --output $EVAL_OUTPUT_BASE/collated/results.csv \
        --json $EVAL_OUTPUT_BASE/collated/results.json
    
    # Plot results
    run_cmd python scripts/enterprise/plot_eval_results.py \
        --results-json $EVAL_OUTPUT_BASE/collated/results.json \
        --output $EVAL_OUTPUT_BASE/collated/results-plot.png
    
    echo ""
    echo "Collation complete! Results at: $EVAL_OUTPUT_BASE/collated/"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================

echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo ""
echo "Checkpoint: $OUTPUT_DIR"
echo "Evals:      $EVAL_OUTPUT_BASE"
echo "Collated:   $EVAL_OUTPUT_BASE/collated/"
echo ""
echo "Next steps:"
echo "  # Append to results table (training hyperparams and performance metrics in one place)"
echo "  python configs/enterprise/ent-expts/append_results.py \\"
echo "    --results $EVAL_OUTPUT_BASE/collated/results.csv \\"
echo "    --checkpoints $CHECKPOINT_DIR \\"
echo "    --master ${EVAL_DIR}/enterprise_results_master.csv \\"
echo "    --notes \"$RUN_NAME\""
echo ""
echo "  # Copy to results bundles to local for inspection (FT + baseline)"
echo "  kubectl cp \$NAMESPACE/<pod>:$EVAL_OUTPUT_BASE ~/Downloads/${MODEL}-ft"
MODEL_SHORT_SANITIZED="${MODEL_HF_NAMES[$MODEL]##*/}"
MODEL_SHORT_SANITIZED="${MODEL_SHORT_SANITIZED//./_}"
echo "  kubectl cp \$NAMESPACE/<pod>:$BASELINE_EVAL_DIR/*${MODEL_SHORT_SANITIZED}* ~/Downloads/${MODEL}-ft/"
echo ""
echo "  # Copy master results csv to local for inspection"
echo "  kubectl cp \$NAMESPACE/<pod>:${EVAL_DIR}/enterprise_results_master.csv ~/Downloads/"
