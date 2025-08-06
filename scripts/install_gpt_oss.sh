#!/bin/bash
# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e  # Exit on any error

echo "🚀 Installing GPT OSS dependencies for Oumi"
echo "=========================================="

# Check if we're in a conda/virtual environment
if [[ -z "$CONDA_DEFAULT_ENV" && -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  WARNING: No conda or virtual environment detected."
    echo "   It's recommended to install in an isolated environment."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Installation cancelled."
        exit 1
    fi
fi

# Step 1: Install standard GPT OSS dependencies
echo ""
echo "📦 Step 1: Installing standard GPT OSS dependencies..."
pip install uv
uv pip install "oumi[gpt_oss]"

# Step 2: Install vLLM GPT OSS build
echo ""
echo "🔧 Step 2: Installing vLLM GPT OSS build..."
echo "   This may take several minutes..."

uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Step 3: Install Flash Attention 3
echo ""
echo "⚡ Step 3: Installing Flash Attention 3..."
echo "   This compilation may take 10-15 minutes..."
uv pip install "flash-attn>=3.0.0" --no-build-isolation

# Step 4: Verify installation
echo ""
echo "🔍 Step 4: Verifying installation..."

# Check if Python can import the packages
python -c "
import sys
print('Python version:', sys.version)

try:
    import vllm
    print('✓ vLLM version:', vllm.__version__)
except ImportError as e:
    print('❌ vLLM import failed:', e)
    sys.exit(1)

try:
    import transformers
    print('✓ Transformers version:', transformers.__version__)
except ImportError as e:
    print('❌ Transformers import failed:', e)
    sys.exit(1)

try:
    import mxfp4
    print('✓ MXFP4 available')
except ImportError as e:
    print('❌ MXFP4 import failed:', e)
    sys.exit(1)

try:
    import flash_attn
    print('✓ Flash Attention version:', flash_attn.__version__)
except ImportError as e:
    print('❌ Flash Attention import failed:', e)
    sys.exit(1)

print('')
print('✅ All GPT OSS dependencies are correctly installed!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 GPT OSS installation completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Test with: python scripts/test_gpt_oss.py"
    echo "2. Run inference: oumi infer -c configs/recipes/gpt_oss/inference/20b_vllm_infer.yaml --interactive"
    echo "3. Train with LoRA: oumi train -c configs/recipes/gpt_oss/sft/20b_lora_train.yaml"
    echo ""
else
    echo ""
    echo "❌ Installation verification failed!"
    echo "   Please check the error messages above and retry."
    exit 1
fi