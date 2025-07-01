#!/bin/bash
# AWQ Quantization Testing Script
# Run various AWQ quantization tests with different configurations

# Don't exit on error - we want to continue testing even if some fail
# set -e  

echo "🧪 AWQ Quantization Testing Suite"
echo "=================================="
echo "⚠️  Note: If autoawq is not installed, tests will run in simulation mode"
echo ""

# Create output directories
mkdir -p test_outputs models/production models/high_quality models/fast models/edge

# Function to run test and handle errors
run_test() {
    local test_name="$1"
    local command="$2"
    
    echo "Running: $test_name"
    echo "Command: $command"
    
    if eval "$command"; then
        echo "✅ $test_name completed successfully"
    else
        echo "❌ $test_name failed (exit code: $?)"
        echo "   This may be expected if dependencies are missing"
    fi
    echo ""
}

echo "1️⃣ Basic AWQ Test (Small Model)"
run_test "Basic AWQ Test" "oumi quantize --config examples/quantization/basic_awq_test.yaml"

echo "2️⃣ Production AWQ Test (CLI Override)"
run_test "Production AWQ Test" "oumi quantize --config examples/quantization/production_awq_llama2.yaml --method awq_q4_1 --output 'test_outputs/llama2_cli_override.gguf'"

echo "3️⃣ High Quality AWQ Q8 Test"
run_test "High Quality AWQ Q8 Test" "oumi quantize --config examples/quantization/high_quality_awq_q8.yaml"

echo "4️⃣ Fast AWQ F16 Test"
run_test "Fast AWQ F16 Test" "oumi quantize --config examples/quantization/fast_awq_f16.yaml"

echo "5️⃣ Edge Deployment Test"
run_test "Edge Deployment Test" "oumi quantize --config examples/quantization/edge_deployment_awq.yaml"

echo "6️⃣ CLI-Only Test (No Config File)"
run_test "CLI-Only Test" "oumi quantize --method awq_q4_0 --model 'microsoft/DialoGPT-small' --output 'test_outputs/cli_only_test.gguf'"

echo "7️⃣ Comparison Test (AWQ vs Direct GGUF)"
run_test "AWQ Method Test" "oumi quantize --method awq_q4_0 --model 'microsoft/DialoGPT-small' --output 'test_outputs/awq_method.gguf'"
run_test "Direct GGUF Method Test" "oumi quantize --method q4_0 --model 'microsoft/DialoGPT-small' --output 'test_outputs/direct_method.gguf'"

echo ""
echo "✅ All AWQ tests completed!"
echo ""
echo "📊 Output files created:"
echo "  test_outputs/"
echo "  models/production/"
echo "  models/high_quality/" 
echo "  models/fast/"
echo "  models/edge/"
echo ""
echo "🔍 Check verbose logs for detailed AWQ processing information."