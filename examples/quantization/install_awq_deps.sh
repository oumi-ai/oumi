#!/bin/bash
# Install AWQ Dependencies for Real Quantization
# This script installs the required dependencies for AWQ quantization

echo "🔧 Installing AWQ Dependencies"
echo "==============================="

echo ""
echo "1️⃣ Installing AutoAWQ..."
echo "This will enable real AWQ quantization (not simulation mode)"

# Check if we're in a conda environment
if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo "📦 Conda environment detected: $CONDA_DEFAULT_ENV"
    echo "Installing via pip in conda environment..."
    pip install autoawq
else
    echo "🐍 Installing in current Python environment..."
    pip install autoawq
fi

echo ""
echo "2️⃣ Installing additional dependencies..."
pip install torch torchvision torchaudio
pip install transformers
pip install accelerate

echo ""
echo "3️⃣ Optional: Installing llama.cpp tools for GGUF conversion..."
echo "Note: This is optional - AWQ will use fallback conversion if not available"

# Check if user wants to install llama.cpp
read -p "Install llama.cpp Python bindings? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install llama-cpp-python
    echo "✅ llama-cpp-python installed"
else
    echo "⏭️  Skipping llama-cpp-python (using fallback conversion)"
fi

echo ""
echo "🧪 Testing AWQ installation..."

python3 -c "
try:
    import autoawq
    print(f'✅ AutoAWQ version: {autoawq.__version__}')
except ImportError:
    print('❌ AutoAWQ not found')
    exit(1)

try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name()}')
    else:
        print('⚠️  CUDA not available (CPU-only)')
except ImportError:
    print('❌ PyTorch not found')
    exit(1)

try:
    import transformers
    print(f'✅ Transformers version: {transformers.__version__}')
except ImportError:
    print('❌ Transformers not found')
    exit(1)

print('🎉 All dependencies installed successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎯 Ready for AWQ Quantization!"
    echo ""
    echo "Now you can run AWQ quantization commands:"
    echo "  oumi quantize --method awq_q4_0 --model microsoft/DialoGPT-small --output test.gguf"
    echo ""
    echo "Or run the full test suite:"
    echo "  ./examples/quantization/test_awq_examples.sh"
else
    echo ""
    echo "❌ Installation failed. Please check the error messages above."
    echo "You may need to install dependencies manually:"
    echo "  pip install autoawq torch transformers accelerate"
fi