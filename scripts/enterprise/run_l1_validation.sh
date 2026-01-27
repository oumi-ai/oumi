#!/bin/bash
# 
# NB: This is an ad hoc model-generated utility script.
# Use it for launching experiments rapidly and storing the results in a useful way.
# Checking it in for record keeping purposes but use with caution!
# 
# L1 Validation Script for Preset Training Configs
#
# This script runs training jobs to validate that preset configs work correctly.
# It trains each model on a small dataset (pubmedqa) to verify the config runs.
#
# Usage:
#   ./scripts/enterprise/run_l1_validation.sh [--model <model_id>] [--all] [--dry-run]
#
# Examples:
#   ./scripts/enterprise/run_l1_validation.sh --all              # Run all validations
#   ./scripts/enterprise/run_l1_validation.sh --model smollm2-135m  # Run single model
#   ./scripts/enterprise/run_l1_validation.sh --all --dry-run    # Print commands only

set -e

# ============================================================================
# Configuration
# ============================================================================

BASEDIR="${BASEDIR:-/data/tim/code/oumi}"
CKPT_DIR="${CKPT_DIR:-/data/tim/checkpoints/l1-validation}"
EVAL_DIR="${EVAL_DIR:-/data/tim/evals/ent/l1-validation}"
DATASET_DIR="${BASEDIR}/data/enterprise"
CONFIG_DIR="${BASEDIR}/configs/enterprise/training/preset-validation"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-9010}"

# Model definitions: id -> config file
# FFT models
declare -A FFT_MODELS
FFT_MODELS["smollm2-135m"]="HuggingFaceTB_SmolLM2-135M-Instruct_fft.yaml"
FFT_MODELS["smollm2-360m"]="HuggingFaceTB_SmolLM2-360M-Instruct_fft.yaml"
FFT_MODELS["phi35-mini"]="microsoft_Phi-3.5-mini-instruct_fft.yaml"
FFT_MODELS["phi35-moe"]="microsoft_Phi-3.5-MoE-instruct_fft.yaml"
FFT_MODELS["qwen25-1.5b"]="Qwen_Qwen2.5-1.5B-Instruct_fft.yaml"
FFT_MODELS["qwen25-3b"]="Qwen_Qwen2.5-3B-Instruct_fft.yaml"
FFT_MODELS["qwen25-7b"]="Qwen_Qwen2.5-7B_fft.yaml"
FFT_MODELS["qwen25-7b-instruct"]="Qwen_Qwen2.5-7B-Instruct_fft.yaml"
FFT_MODELS["qwen3-0.6b"]="Qwen_Qwen3-0.6B_fft.yaml"

# LoRA models
declare -A LORA_MODELS
LORA_MODELS["gemma3-4b-lora"]="gemma3_4b_it_lora.yaml"
LORA_MODELS["llama31-8b-lora"]="meta-llama_Llama-3.1-8B-Instruct_lora.yaml"
LORA_MODELS["llama32-3b-lora"]="meta-llama_Llama-3.2-3B-Instruct_lora.yaml"
LORA_MODELS["llama32-1b-lora"]="meta-llama_Llama-3.2-1B-Instruct_lora.yaml"
LORA_MODELS["smollm2-1.7b-lora"]="HuggingFaceTB_SmolLM2-1.7B-Instruct_lora.yaml"
LORA_MODELS["smollm2-135m-lora"]="HuggingFaceTB_SmolLM2-135M-Instruct_lora.yaml"
LORA_MODELS["smollm2-360m-lora"]="HuggingFaceTB_SmolLM2-360M-Instruct_lora.yaml"
LORA_MODELS["phi35-mini-lora"]="microsoft_Phi-3.5-mini-instruct_lora.yaml"
LORA_MODELS["phi35-moe-lora"]="microsoft_Phi-3.5-MoE-instruct_lora.yaml"
LORA_MODELS["qwen25-1.5b-lora"]="Qwen_Qwen2.5-1.5B-Instruct_lora.yaml"
LORA_MODELS["qwen25-3b-lora"]="Qwen_Qwen2.5-3B-Instruct_lora.yaml"
LORA_MODELS["qwen25-7b-lora"]="Qwen_Qwen2.5-7B_lora.yaml"
LORA_MODELS["qwen25-7b-instruct-lora"]="Qwen_Qwen2.5-7B-Instruct_lora.yaml"
LORA_MODELS["qwen3-0.6b-lora"]="Qwen_Qwen3-0.6B_lora.yaml"
LORA_MODELS["qwen3-4b-lora"]="Qwen_Qwen3-4B-Instruct_lora.yaml"
LORA_MODELS["qwen3-8b-lora"]="Qwen_Qwen3-8B_lora.yaml"

# Combined for backwards compatibility
declare -A MODELS
for k in "${!FFT_MODELS[@]}"; do MODELS["$k"]="${FFT_MODELS[$k]}"; done
for k in "${!LORA_MODELS[@]}"; do MODELS["$k"]="${LORA_MODELS[$k]}"; done

# Results tracking
RESULTS_FILE="${CKPT_DIR}/l1_validation_results.txt"

# ============================================================================
# Parse arguments
# ============================================================================

RUN_ALL=false
RUN_FFT=false
RUN_LORA=false
SINGLE_MODEL=""
DRY_RUN=false
EVAL_ONLY=false

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --all              Run validation for ALL models (FFT + LoRA)"
    echo "  --fft              Run validation for all FFT models"
    echo "  --lora             Run validation for all LoRA models"
    echo "  --model <id>       Run validation for a single model"
    echo "  --eval-only        Skip training, only run evals on existing checkpoints"
    echo "  --dry-run          Print commands without executing"
    echo "  --list             List available models"
    echo "  -h, --help         Show this help"
    echo ""
    echo "FFT models:"
    for model in "${!FFT_MODELS[@]}"; do
        echo "  $model -> ${FFT_MODELS[$model]}"
    done | sort
    echo ""
    echo "LoRA models:"
    for model in "${!LORA_MODELS[@]}"; do
        echo "  $model -> ${LORA_MODELS[$model]}"
    done | sort
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --fft)
            RUN_FFT=true
            shift
            ;;
        --lora)
            RUN_LORA=true
            shift
            ;;
        --model)
            SINGLE_MODEL="$2"
            shift 2
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            echo "FFT models:"
            for model in "${!FFT_MODELS[@]}"; do
                echo "  $model -> ${FFT_MODELS[$model]}"
            done | sort
            echo ""
            echo "LoRA models:"
            for model in "${!LORA_MODELS[@]}"; do
                echo "  $model -> ${LORA_MODELS[$model]}"
            done | sort
            exit 0
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

# Validate args
if [[ "$RUN_ALL" == "false" ]] && [[ "$RUN_FFT" == "false" ]] && [[ "$RUN_LORA" == "false" ]] && [[ -z "$SINGLE_MODEL" ]]; then
    echo "Error: Must specify --all, --fft, --lora, or --model <id>"
    print_usage
    exit 1
fi

if [[ -n "$SINGLE_MODEL" ]] && [[ -z "${MODELS[$SINGLE_MODEL]}" ]]; then
    echo "Error: Unknown model '$SINGLE_MODEL'"
    echo "Use --list to see available models"
    exit 1
fi

# ============================================================================
# Helper functions
# ============================================================================

run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] $@"
    else
        echo "$ $@"
        "$@"
    fi
}

log_result() {
    local model="$1"
    local status="$2"
    local notes="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [[ "$DRY_RUN" != "true" ]]; then
        mkdir -p "$(dirname "$RESULTS_FILE")"
        echo "$timestamp | $model | $status | $notes" >> "$RESULTS_FILE"
    fi
    
    if [[ "$status" == "SUCCESS" ]]; then
        echo "✅ $model: $status"
    else
        echo "❌ $model: $status - $notes"
    fi
}

# ============================================================================
# Training function
# ============================================================================

train_model() {
    local model_id="$1"
    local config_file="${MODELS[$model_id]}"
    local config_path="${CONFIG_DIR}/${config_file}"
    local run_name="l1-${model_id}-pubmedqa"
    local output_dir="${CKPT_DIR}/${model_id}-pubmedqa"
    
    echo ""
    echo "=============================================="
    echo "Training: $model_id"
    echo "Config:   $config_path"
    echo "Output:   $output_dir"
    echo "=============================================="
    
    if [[ ! -f "$config_path" ]]; then
        log_result "$model_id" "FAILED" "Config not found: $config_path"
        return 1
    fi
    
    # Run training
    if run_cmd oumi distributed torchrun \
        --nproc_per_node=$NPROC \
        --master-port=$MASTER_PORT \
        -m oumi train \
        -c "$config_path" \
        --data.train.datasets.0.dataset_path="${DATASET_DIR}/pubmedqa/train.jsonl" \
        --data.validation.datasets.0.dataset_path="${DATASET_DIR}/pubmedqa/val.jsonl" \
        --training.run_name="$run_name" \
        --training.output_dir="$output_dir"; then
        log_result "$model_id" "SUCCESS" "Training completed"
        return 0
    else
        log_result "$model_id" "FAILED" "Training error (exit code: $?)"
        return 1
    fi
}

# ============================================================================
# Eval function
# ============================================================================

eval_model() {
    local model_id="$1"
    local checkpoint_dir="${CKPT_DIR}/${model_id}-pubmedqa"
    local eval_output="${EVAL_DIR}/${model_id}"
    
    echo ""
    echo "=============================================="
    echo "Evaluating: $model_id"
    echo "Checkpoint: $checkpoint_dir"
    echo "Output:     $eval_output"
    echo "=============================================="
    
    if [[ ! -d "$checkpoint_dir" ]]; then
        echo "Warning: Checkpoint not found: $checkpoint_dir"
        log_result "$model_id" "EVAL_SKIPPED" "No checkpoint"
        return 1
    fi
    
    # Run eval on pubmedqa only (quick validation)
    if run_cmd python -m oumi evaluate \
        -c configs/enterprise/evals/pubmedqa_eval.yaml \
        --model.model_name="$checkpoint_dir" \
        --output_dir="$eval_output/pubmedqa"; then
        log_result "$model_id" "EVAL_SUCCESS" "Eval completed"
        return 0
    else
        log_result "$model_id" "EVAL_FAILED" "Eval error"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

# Determine run mode for display
if [[ "$RUN_ALL" == "true" ]]; then
    RUN_MODE="all (FFT + LoRA)"
elif [[ "$RUN_FFT" == "true" ]]; then
    RUN_MODE="FFT only"
elif [[ "$RUN_LORA" == "true" ]]; then
    RUN_MODE="LoRA only"
else
    RUN_MODE="single model: $SINGLE_MODEL"
fi

echo "=============================================="
echo "L1 Validation Runner"
echo "=============================================="
echo ""
echo "Config dir:   $CONFIG_DIR"
echo "Checkpoint:   $CKPT_DIR"
echo "Eval output:  $EVAL_DIR"
echo "Results log:  $RESULTS_FILE"
echo "Run mode:     $RUN_MODE"
echo "Dry run:      $DRY_RUN"
echo "Eval only:    $EVAL_ONLY"
echo ""

# Determine which models to run
if [[ "$RUN_ALL" == "true" ]]; then
    MODELS_TO_RUN=(${!MODELS[@]})
elif [[ "$RUN_FFT" == "true" ]]; then
    MODELS_TO_RUN=(${!FFT_MODELS[@]})
elif [[ "$RUN_LORA" == "true" ]]; then
    MODELS_TO_RUN=(${!LORA_MODELS[@]})
else
    MODELS_TO_RUN=("$SINGLE_MODEL")
fi

# Sort models for consistent ordering
IFS=$'\n' MODELS_TO_RUN=($(sort <<<"${MODELS_TO_RUN[*]}")); unset IFS

echo "Models to validate: ${MODELS_TO_RUN[*]}"
echo ""

# Track results
SUCCEEDED=()
FAILED=()

for model in "${MODELS_TO_RUN[@]}"; do
    if [[ "$EVAL_ONLY" != "true" ]]; then
        if train_model "$model"; then
            SUCCEEDED+=("$model")
        else
            FAILED+=("$model")
            continue  # Skip eval if training failed
        fi
    fi
    
    # Run eval (optional, comment out if you just want training validation)
    # eval_model "$model"
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "L1 Validation Summary"
echo "=============================================="
echo ""
echo "Succeeded (${#SUCCEEDED[@]}):"
for m in "${SUCCEEDED[@]}"; do echo "  ✅ $m"; done

echo ""
echo "Failed (${#FAILED[@]}):"
for m in "${FAILED[@]}"; do echo "  ❌ $m"; done

echo ""
echo "Results logged to: $RESULTS_FILE"
echo ""

# Exit with error if any failed
if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi

