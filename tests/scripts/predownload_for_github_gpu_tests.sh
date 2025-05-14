#!/bin/bash

# Script to pre-download common models referenced in tests to reduce test time variance.
# Used in ".github/workflows/gpu_tests.yaml"
set -xe

export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download "HuggingFaceTB/SmolLM2-135M-Instruct" --exclude "onnx/*" "runs/*"
huggingface-cli download "Qwen/Qwen2-VL-2B-Instruct"
