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

# Step 1: Install all GPT OSS dependencies at once
echo ""
echo "📦 Step 1: Installing all GPT OSS dependencies..."
pip install uv

# Install GPT OSS dependencies first (includes PyTorch)
echo "   Installing GPT OSS dependencies..."
# Use PyPI as primary index, PyTorch as extra index to ensure all packages are found
uv pip install "oumi[gpt_oss]" \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install vLLM GPT OSS build separately with proper index strategy
echo "   Installing vLLM GPT OSS build..."
echo "   This may take several minutes..."
uv pip install vllm==0.10.2+gptoss \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match

# Step 2: Flash Attention 3 (using pre-compiled kernel instead)
echo ""
echo "⚡ Step 2: Using pre-compiled vLLM Flash Attention 3 kernel..."
echo "   Flash Attention 3 will be loaded from kernels-community/vllm-flash-attn3"
echo "   No compilation needed - kernel will be downloaded on first use"

# TODO: Uncomment below to install Flash Attention 3 from source
# # Step 3: Install Flash Attention 3 from source
# echo ""
# echo "⚡ Step 3: Installing Flash Attention 3 from source..."
# echo "   This requires H100/H800 GPU and CUDA >= 12.3"
# 
# # Check CUDA version
# if command -v nvcc >/dev/null 2>&1; then
#     CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
#     echo "   Detected CUDA version: $CUDA_VERSION"
# else
#     echo "⚠️  WARNING: nvcc not found. Flash Attention 3 requires CUDA >= 12.3"
# fi
# 
# # Install required dependencies
# echo "   Installing compilation dependencies..."
# uv pip install packaging ninja
# 
# # Check available RAM and set MAX_JOBS if needed
# RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
# if [ "$RAM_GB" -lt 96 ] 2>/dev/null; then
#     echo "   Detected ${RAM_GB}GB RAM, limiting parallel jobs to 4"
#     export MAX_JOBS=4
# fi
# 
# # Clone and install Flash Attention 3
# echo "   Cloning Flash Attention repository..."
# TEMP_DIR=$(mktemp -d)
# 
# # Set up cleanup trap to ensure temp directory is always deleted
# cleanup_temp() {
#     if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
#         echo "   Cleaning up temporary directory..."
#         cd - >/dev/null 2>&1 || true
#         rm -rf "$TEMP_DIR"
#     fi
# }
# trap cleanup_temp EXIT ERR
# 
# git clone https://github.com/Dao-AILab/flash-attention.git "$TEMP_DIR"
# cd "$TEMP_DIR/hopper"
# 
# echo "   Compiling Flash Attention 3 (this may take 10-20 minutes)..."
# echo "   Using --no-build-isolation for CUDA environment access..."
# uv pip install . --no-build-isolation
# 
# # Clean up (trap will also handle this, but explicit cleanup for success case)
# cd - >/dev/null
# rm -rf "$TEMP_DIR"
# TEMP_DIR=""  # Clear variable so trap doesn't try to clean up again
# 
# echo "   ✓ Flash Attention 3 installed from source"

# Step 3: Verify installation
echo ""
echo "🔍 Step 3: Verifying installation..."

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
    import gpt_oss
    print('✓ GPT OSS package available (includes MXFP4 quantization support)')
except ImportError as e:
    print('❌ GPT OSS package import failed:', e)
    sys.exit(1)

try:
    from kernels import get_kernel
    # Try to verify the kernel can be loaded
    print('✓ Kernels package available')
    print('   Note: vllm-flash-attn3 kernel will be downloaded on first use')
except ImportError as e:
    print('⚠️  Kernels package not found:', e)
    print('   Flash Attention 3 will use fallback implementation')
    # Don't fail - kernels is optional

# TODO: Uncomment when using FA3 from source
# try:
#     import flash_attn_interface
#     print('✓ Flash Attention 3 interface available')
#     # Test the function to ensure it works
#     flash_attn_interface.flash_attn_func
#     print('✓ Flash Attention 3 function accessible')
# except ImportError as e:
#     print('❌ Flash Attention 3 import failed:', e)
#     print('   Note: Flash Attention 3 requires H100/H800 GPU and CUDA >= 12.3')
#     sys.exit(1)

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
