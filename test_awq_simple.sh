#!/bin/bash

# Simple AWQ test script for Llama models
# Tests the CLI interface with different models

set -e  # Exit on any error

echo "🚀 Simple AWQ Quantization Test"
echo "================================"

cd /home/yuzhang/oumi

# Test 1: TinyLlama (small model for quick testing)
echo ""
echo "🔍 Test 1: TinyLlama (1.1B parameters)"
echo "Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo "Method: awq_q4_0"
echo "Output: tinyllama-awq-test.gguf"

python -m oumi.cli quantize \
    --method awq_q4_0 \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --output "tinyllama-awq-test.gguf" \
    --level INFO

# Check if file was created
if [ -f "tinyllama-awq-test.gguf" ]; then
    file_size=$(du -h "tinyllama-awq-test.gguf" | cut -f1)
    echo "✅ Output file created: tinyllama-awq-test.gguf ($file_size)"
else
    echo "❌ Output file not found"
fi

echo ""
echo "🔍 Test 2: Using configuration file"
echo "Config: examples/quantization/basic_awq_test.yaml"

python -m oumi.cli quantize \
    --config examples/quantization/basic_awq_test.yaml \
    --level INFO

echo ""
echo "🏁 Test completed!"
echo ""
echo "💡 Notes:"
echo "   - If you see 'SIMULATION MODE', install AWQ: pip install autoawq"
echo "   - For real quantization, ensure you have sufficient GPU memory"
echo "   - Check output files to verify they were created correctly"