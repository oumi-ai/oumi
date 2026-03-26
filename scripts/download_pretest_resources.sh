#!/bin/bash

# Script to pre-download HuggingFace resources needed by CPU unit tests.
# After running this script, tests can run fully offline with:
#   HF_HUB_OFFLINE=1 HF_HOME=$HF_HOME TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#     pytest tests/unit/ -m "not e2e and not e2e_eternal and not single_gpu and not multi_gpu"
#
# Used alongside ".github/workflows/pretest.yaml"
set -xe

export HF_HUB_ENABLE_HF_TRANSFER=1

TOKENIZER_FILES='["config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","generation_config.json","special_tokens_map.json"]'
PROCESSOR_FILES='["config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","generation_config.json","special_tokens_map.json","preprocessor_config.json","processor_config.json","chat_template.json"]'

# Use Python for downloads since `huggingface-cli` may not be on PATH in all envs.
# snapshot_download correctly populates the blobs/refs/snapshots cache structure
# that transformers and datasets libraries expect for offline use.
download_model() {
    local model=$1
    local patterns=$2
    local max_retries=5
    local attempt=0
    local retry_delay=10

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
    local subset=$2
    local split=$3
    local max_retries=5
    local attempt=0
    local retry_delay=10

    while [ $attempt -lt $max_retries ]; do
        echo "Attempting to download dataset $dataset/$subset (attempt $((attempt + 1))/$max_retries)..."
        if python -c "
import datasets
datasets.load_dataset('$dataset', '$subset', split='$split')
" ; then
            echo "Successfully downloaded dataset $dataset/$subset"
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

    echo "Failed to download dataset $dataset/$subset after $max_retries attempts"
    return 1
}

# ========================================
# MODELS - tokenizer/config files only
# ========================================

# Used by: test_sglang_inference_engine (collection-time parametrize),
#          test_chat_templates, test_models, test_hf_utils
download_model "openai-community/gpt2" "$TOKENIZER_FILES"

# Used by: test_chat_templates (llava template), test_sglang_inference_engine
download_model "llava-hf/llava-1.5-7b-hf" "$PROCESSOR_FILES"

# Used by: test_models (is_image_text_llm, build_tokenizer), test_hf_utils
download_model "HuggingFaceTB/SmolLM2-135M-Instruct" "$TOKENIZER_FILES"

# Used by: test_hf_utils, test_analysis_utils, test_huggingface_vision_dataset,
#          test_vision_language_jsonlines_dataset
download_model "HuggingFaceTB/SmolVLM-256M-Instruct" "$PROCESSOR_FILES"

# Used by: test_models (is_image_text_llm)
download_model "HuggingFaceTB/SmolVLM-Instruct" "$PROCESSOR_FILES"

# Used by: test_huggingface_vision_dataset, test_analysis_utils,
#          test_vision_language_jsonlines_dataset, test_supported_models
download_model "Salesforce/blip2-opt-2.7b" "$PROCESSOR_FILES"

# Used by: test_chat_templates, test_models, test_supported_models
# Needs *.py files for trust_remote_code=True
download_model "microsoft/Phi-3-vision-128k-instruct" '["*.json","*.py"]'

# Used by: test_chat_templates (phi3 template)
# Needs *.py files for trust_remote_code=True
download_model "microsoft/Phi-3-mini-4k-instruct" '["*.json","*.py"]'

# Used by: test_chat_templates, test_models, test_supported_models
download_model "Qwen/Qwen2-VL-2B-Instruct" "$PROCESSOR_FILES"

# Used by: test_chat_templates (qwen2 template)
download_model "Qwen/Qwen2.5-VL-3B-Instruct" "$PROCESSOR_FILES"

# ========================================
# DATASETS
# ========================================

# Used by: test_data_mixtures (test_data_multiple_datasets)
# We use datasets.load_dataset to build the arrow cache, which is needed
# for offline mode (snapshot_download alone is not sufficient).
download_dataset "tasksource/mmlu" "abstract_algebra" "test"
