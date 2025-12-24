#!/bin/bash

# Script to pre-download common models referenced in tests to reduce test time variance.
# Used in ".github/workflows/gpu_tests.yaml"
set -xe

export HF_HUB_ENABLE_HF_TRANSFER=1

# Function to download with retries
download_with_retry() {
    local model=$1
    local max_retries=5
    local retry_delay=30
    local attempt=0

    shift  # Remove first argument to pass remaining args

    while [ $attempt -lt $max_retries ]; do
        echo "Attempting to download $model (attempt $((attempt + 1))/$max_retries)..."

        if hf download "$model" "$@"; then
            echo "Successfully downloaded $model"
            return 0
        else
            attempt=$((attempt + 1))
            if [ $attempt -lt $max_retries ]; then
                echo "Download failed, waiting ${retry_delay}s before retry..."
                sleep $retry_delay
                # Exponential backoff
                retry_delay=$((retry_delay * 2))
            fi
        fi
    done

    echo "Failed to download $model after $max_retries attempts"
    return 1
}

# Download models with retry logic
download_with_retry "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --include "config.json" "model.safetensors" "tokenizer.json" "tokenizer_config.json" \
    "vocab.json" "merges.txt" "special_tokens_map.json" "generation_config.json"

download_with_retry "HuggingFaceTB/SmolVLM-256M-Instruct" \
    --include "config.json" "model.safetensors" "tokenizer.json" "tokenizer_config.json" \
    "vocab.json" "merges.txt" "special_tokens_map.json" "generation_config.json" \
    "preprocessor_config.json" "processor_config.json"

download_with_retry "openai-community/gpt2" \
    --include "config.json" "model.safetensors" "tokenizer.json" "tokenizer_config.json" \
    "vocab.json" "merges.txt" "generation_config.json"

download_with_retry "Qwen/Qwen3-0.6B" \
    --include "config.json" "model.safetensors" "tokenizer.json" "tokenizer_config.json" \
    "vocab.json" "merges.txt" "special_tokens_map.json" "generation_config.json"

# ========================================
# DATASETS USED IN ACTIVE TESTS
# ========================================

# MMLU dataset used in unit tests (test_data_mixtures.py)
download_with_retry "tasksource/mmlu" --repo-type dataset \
    --include "college_computer_science/*" "abstract_algebra/*"

# Alpaca dataset used in integration tests (test_train.py)
download_with_retry "yahma/alpaca-cleaned" --repo-type dataset \
    --include "alpaca_data.json" "alpaca_data_gpt4.json"
