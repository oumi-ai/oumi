#!/bin/bash

# Qwen 7B AWQ Quantization Example Script
# This script runs AWQ quantization on Qwen-7B with 1024 calibration samples

echo "🚀 Starting Qwen-7B AWQ quantization with 1024 calibration samples..."
echo ""

# Check if oumi is installed
if ! command -v oumi &> /dev/null; then
    echo "❌ oumi command not found. Please install oumi first:"
    echo "   pip install -e ."
    exit 1
fi

# Check available disk space (AWQ quantization requires temporary space)
echo "📊 Checking available disk space..."
df -h .

echo ""
echo "⚙️  Configuration:"
echo "   Model: Qwen/Qwen-7B"
echo "   Method: AWQ 4-bit quantization"
echo "   Calibration samples: 1024"
echo "   Output: qwen-7b-awq-q4.gguf"
echo ""

# Run quantization
echo "🔥 Starting quantization (this may take 10-20 minutes)..."
oumi quantize examples/quantization/qwen_7b_awq_example.yaml

# Check if quantization succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Quantization completed successfully!"
    echo "📁 Output file: qwen-7b-awq-q4.gguf"
    
    # Show file size
    if [ -f "qwen-7b-awq-q4.gguf" ]; then
        echo "📊 File size: $(du -h qwen-7b-awq-q4.gguf | cut -f1)"
    elif [ -d "qwen-7b-awq-q4.pytorch" ]; then
        echo "📊 Directory size: $(du -sh qwen-7b-awq-q4.pytorch | cut -f1)"
        echo "ℹ️  Note: Saved as PyTorch format (install llama-cpp-python for GGUF)"
    fi
    
    echo ""
    echo "🎉 Your quantized Qwen-7B model is ready for inference!"
else
    echo ""
    echo "❌ Quantization failed. Check the logs above for details."
    exit 1
fi