#!/bin/bash

# Script to pre-download common models and datasets referenced in GPU integration tests.
# After running this script, tests can run fully offline with:
#   HF_HUB_OFFLINE=1 HF_HOME=$HF_HOME TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#     pytest tests/integration/ -m "not e2e and not e2e_eternal and not multi_gpu"
#
# Used in ".github/workflows/gpu_tests.yaml"
set -xe

export HF_HUB_ENABLE_HF_TRANSFER=1

# ========================================
# Download helpers using Python (snapshot_download / datasets.load_dataset)
# These correctly populate the blobs/refs/snapshots cache structure
# that transformers and datasets libraries expect for offline use.
# ========================================

download_model() {
    local model=$1
    local patterns=$2
    local max_retries=5
    local attempt=0
    local retry_delay=30

    while [ $attempt -lt $max_retries ]; do
        echo "Attempting to download $model (attempt $((attempt + 1))/$max_retries)..."
        if python -c "
from huggingface_hub import snapshot_download
import json
snapshot_download('$model', allow_patterns=json.loads('$patterns'))
" ; then
            echo "Successfully downloaded $model"
            return 0
        else
            attempt=$((attempt + 1))
            if [ $attempt -lt $max_retries ]; then
                echo "Download failed, waiting ${retry_delay}s before retry..."
                sleep $retry_delay
                retry_delay=$((retry_delay * 2))
            fi
        fi
    done

    echo "Failed to download $model after $max_retries attempts"
    return 1
}

download_dataset() {
    local dataset=$1
    local max_retries=5
    local attempt=0
    local retry_delay=30

    shift  # remaining args are Python kwargs

    while [ $attempt -lt $max_retries ]; do
        echo "Attempting to download dataset $dataset (attempt $((attempt + 1))/$max_retries)..."
        if python -c "
import datasets
datasets.load_dataset('$dataset', $@)
" ; then
            echo "Successfully downloaded dataset $dataset"
            return 0
        else
            attempt=$((attempt + 1))
            if [ $attempt -lt $max_retries ]; then
                echo "Download failed, waiting ${retry_delay}s before retry..."
                sleep $retry_delay
                retry_delay=$((retry_delay * 2))
            fi
        fi
    done

    echo "Failed to download dataset $dataset after $max_retries attempts"
    return 1
}

TOKENIZER_FILES='["config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","generation_config.json","special_tokens_map.json"]'
MODEL_FILES='["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","generation_config.json","special_tokens_map.json"]'
PROCESSOR_FILES='["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","generation_config.json","special_tokens_map.json","preprocessor_config.json","processor_config.json","chat_template.json"]'

# ========================================
# MODELS
# ========================================

# Used by: test_train (basic, unregistered_metrics, dpo, kto, pretraining, sft),
#          test_evaluate_lm_harness
download_model "openai-community/gpt2" "$MODEL_FILES"

# Used by: test_train (gkd student model), test_infer, test_native_text_inference_engine
download_model "HuggingFaceTB/SmolLM2-135M-Instruct" "$MODEL_FILES"

# Used by: test_train (gkd teacher model)
download_model "HuggingFaceTB/SmolLM2-360M-Instruct" "$MODEL_FILES"

# Used by: test_infer (with images), test_native_text_inference_engine (with images)
download_model "HuggingFaceTB/SmolVLM-256M-Instruct" "$PROCESSOR_FILES"

# Used by: test_train_gold
download_model "Qwen/Qwen3-0.6B" "$MODEL_FILES"

# Used by: test_verl_train
download_model "Qwen/Qwen2.5-0.5B" "$MODEL_FILES"

# Used by: test_vision_language_completions_only (phi3 tokenization, completions only)
# Needs *.py files for trust_remote_code=True
download_model "microsoft/Phi-3-vision-128k-instruct" '["*.json","*.py"]'

# ========================================
# DATASETS
# ========================================
# We use datasets.load_dataset() to build the arrow cache, which is needed
# for offline mode (snapshot_download alone is not sufficient for the datasets library).

# Used by: test_evaluate_lm_harness (lm_harness uses cais/mmlu, not tasksource/mmlu)
download_dataset "cais/mmlu" "'abstract_algebra', split='test'"
download_dataset "cais/mmlu" "'college_computer_science', split='test'"

# Used by: test_train (basic, unregistered_metrics)
download_dataset "yahma/alpaca-cleaned" "split='train'"

# Used by: test_verl_train (verl grpo)
download_dataset "d1shs0ap/countdown" "split='train'"
