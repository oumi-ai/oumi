#!/bin/bash
# Setup Real AWQ Quantization
# This script installs dependencies and prepares for actual AWQ quantization

echo "🔧 Setting Up Real AWQ Quantization"
echo "==================================="
echo ""

# Check current environment
echo "🔍 Environment Check:"
echo "Current conda environment: ${CONDA_DEFAULT_ENV:-none}"
echo "Python version: $(python3 --version 2>/dev/null || echo 'not found')"
echo ""

# Function to check if package is installed
check_package() {
    python3 -c "import $1; print(f'✅ $1: {$1.__version__}')" 2>/dev/null || echo "❌ $1: not installed"
}

echo "📦 Current Package Status:"
check_package torch
check_package transformers
check_package accelerate
check_package autoawq
echo ""

# Check system requirements
echo "💾 System Requirements Check:"
total_ram=$(python3 -c "
import psutil
ram_gb = psutil.virtual_memory().total / (1024**3)
print(f'{ram_gb:.1f} GB')
" 2>/dev/null || echo "unknown")
echo "Available RAM: $total_ram"

# Check CUDA
echo "🖥️  GPU Check:"
python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name()}')
        print(f'   CUDA version: {torch.version.cuda}')
        print(f'   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print('⚠️  CUDA not available - will use CPU (very slow)')
except:
    print('❓ Unable to check CUDA status')
" 2>/dev/null
echo ""

# Installation section
echo "🚀 Installing AWQ Dependencies:"
echo "==============================="
echo ""

echo "1️⃣ Installing PyTorch..."
if python3 -c "import torch" 2>/dev/null; then
    echo "   ✅ PyTorch already installed"
else
    echo "   Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi
echo ""

echo "2️⃣ Installing Transformers..."
if python3 -c "import transformers" 2>/dev/null; then
    echo "   ✅ Transformers already installed"
else
    echo "   Installing transformers..."
    pip install transformers
fi
echo ""

echo "3️⃣ Installing Accelerate..."
if python3 -c "import accelerate" 2>/dev/null; then
    echo "   ✅ Accelerate already installed"
else
    echo "   Installing accelerate..."
    pip install accelerate
fi
echo ""

echo "4️⃣ Installing AutoAWQ (this may take a while)..."
if python3 -c "import autoawq" 2>/dev/null; then
    echo "   ✅ AutoAWQ already installed"
else
    echo "   Installing autoawq..."
    echo "   Note: This compiles CUDA kernels and may take 10-20 minutes..."
    pip install autoawq
fi
echo ""

echo "5️⃣ Optional: Installing llama-cpp-python for GGUF support..."
read -p "   Install llama-cpp-python? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install llama-cpp-python
    echo "   ✅ llama-cpp-python installed"
else
    echo "   ⏭️  Skipped llama-cpp-python (will use fallback conversion)"
fi
echo ""

# Final verification
echo "✅ Final Verification:"
echo "====================="
python3 -c "
import sys

# Check all packages
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers', 
    'accelerate': 'Accelerate',
    'autoawq': 'AutoAWQ'
}

all_good = True
for package, name in packages.items():
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {name}: {version}')
    except ImportError:
        print(f'❌ {name}: not found')
        all_good = False

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA: {torch.cuda.get_device_name()}')
    else:
        print('⚠️  CUDA: not available (CPU-only)')
        print('   AWQ will be very slow on CPU')
except:
    pass

# Final status
if all_good:
    print()
    print('🎉 All dependencies installed successfully!')
    print()
    print('🚀 Ready for real AWQ quantization!')
    print()
    print('Next steps:')
    print('1. Run a quick test:')
    print('   oumi quantize --method awq_q4_0 --model microsoft/DialoGPT-small --output test.gguf')
    print()
    print('2. Run Llama 2 7B quantization:')
    print('   oumi quantize --config examples/quantization/llama2_7b_awq_example.yaml')
    print()
    print('Expected time for Llama 2 7B:')
    if torch.cuda.is_available():
        print('   - With GPU: 20-45 minutes')
    else:
        print('   - With CPU: 2-4 hours (not recommended)')
else:
    print()
    print('❌ Some dependencies missing. Please check error messages above.')
    print()
    print('Manual installation:')
    print('   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
    print('   pip install transformers accelerate autoawq')
"
echo ""

# HuggingFace setup
echo "🔑 HuggingFace Setup:"
echo "===================="
echo ""
echo "For Llama 2 models, you need to:"
echo "1. Accept the license at: https://huggingface.co/meta-llama/Llama-2-7b-hf"
echo "2. Login to HuggingFace:"
echo "   pip install huggingface_hub"
echo "   huggingface-cli login"
echo ""
echo "Alternative: Use models that don't require authentication:"
echo "   microsoft/DialoGPT-small (for testing)"
echo "   microsoft/DialoGPT-medium"
echo "   codellama/CodeLlama-7b-Python-hf"
echo ""

echo "📊 Memory Requirements:"
echo "======================"
echo ""
echo "For different models:"
echo "   DialoGPT-small:  ~2GB RAM, ~1GB VRAM"
echo "   DialoGPT-medium: ~4GB RAM, ~2GB VRAM"  
echo "   Llama 2 7B:      ~16GB RAM, ~8GB VRAM"
echo "   Llama 2 13B:     ~28GB RAM, ~14GB VRAM"
echo ""

echo "🎯 Ready to proceed with real AWQ quantization!"
echo "Check the installation status above and proceed if all packages are installed."