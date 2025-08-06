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

# Step 1: Install standard GPT OSS dependencies and triton kernels
echo ""
echo "📦 Step 1: Installing standard GPT OSS dependencies..."
pip install uv
uv pip install "oumi[gpt_oss]"
uv pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels

# Step 2: Install vLLM GPT OSS build
echo ""
echo "🔧 Step 2: Installing vLLM GPT OSS build..."
echo "   This may take several minutes..."

uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Step 3: Install Flash Attention 3 from source
echo ""
echo "⚡ Step 3: Installing Flash Attention 3 from source..."
echo "   This requires H100/H800 GPU and CUDA >= 12.3"

# Check CUDA version
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "   Detected CUDA version: $CUDA_VERSION"
else
    echo "⚠️  WARNING: nvcc not found. Flash Attention 3 requires CUDA >= 12.3"
fi

# Install required dependencies
echo "   Installing compilation dependencies..."
uv pip install packaging ninja

# Check available RAM and set MAX_JOBS if needed
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$RAM_GB" -lt 96 ] 2>/dev/null; then
    echo "   Detected ${RAM_GB}GB RAM, limiting parallel jobs to 4"
    export MAX_JOBS=4
fi

# Clone and install Flash Attention 3
echo "   Cloning Flash Attention repository..."
TEMP_DIR=$(mktemp -d)
git clone https://github.com/Dao-AILab/flash-attention.git "$TEMP_DIR"
cd "$TEMP_DIR/hopper"

echo "   Compiling Flash Attention 3 (this may take 10-20 minutes)..."
pip install .

# Clean up
cd - >/dev/null
rm -rf "$TEMP_DIR"

echo "   ✓ Flash Attention 3 installed from source"

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
    import flash_attn_interface
    print('✓ Flash Attention 3 interface available')
    # Test the function to ensure it works
    flash_attn_interface.flash_attn_func
    print('✓ Flash Attention 3 function accessible')
except ImportError as e:
    print('❌ Flash Attention 3 import failed:', e)
    print('   Note: Flash Attention 3 requires H100/H800 GPU and CUDA >= 12.3')
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