#!/bin/bash
# Llama 2 7B AWQ Quantization Example Script
# Demonstrates complete AWQ workflow from configuration to testing

echo "🦙 Llama 2 7B AWQ Quantization Example"
echo "====================================="
echo ""

# Check current mode
echo "🔍 Checking AWQ dependencies..."
python3 -c "
try:
    import autoawq
    print('✅ AutoAWQ installed - will perform REAL quantization')
    print(f'   Version: {autoawq.__version__}')
    
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name()}')
        print('   Expected time: 20-45 minutes')
    else:
        print('⚠️  CUDA not available - will use CPU (very slow)')
        print('   Expected time: 2-4 hours')
except ImportError:
    print('ℹ️  AutoAWQ not installed - will run in SIMULATION mode')
    print('   Install with: pip install autoawq')
    print('   Expected time: 30 seconds')
"
echo ""

# Create output directory
echo "📁 Creating output directories..."
mkdir -p models test_outputs
echo "   Created: models/ and test_outputs/"
echo ""

# Run the quantization
echo "🚀 Starting Llama 2 7B AWQ quantization..."
echo "Command: oumi quantize --config examples/quantization/llama2_7b_awq_example.yaml"
echo ""

# Record start time
start_time=$(date +%s)

# Run quantization
if oumi quantize --config examples/quantization/llama2_7b_awq_example.yaml; then
    echo ""
    echo "✅ Quantization completed successfully!"
    
    # Record end time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "⏱️  Total time: ${duration} seconds"
    echo ""
    
    # Check output file
    output_file="models/llama2-7b-awq-q4.gguf"
    if [ -f "$output_file" ]; then
        file_size=$(du -h "$output_file" | cut -f1)
        echo "📊 Output file information:"
        echo "   Path: $output_file"
        echo "   Size: $file_size"
        echo "   Type: $(file "$output_file" | cut -d: -f2-)"
        
        # Check if it's a real GGUF file
        if head -c 4 "$output_file" | grep -q "GGUF"; then
            echo "   Format: ✅ Valid GGUF file"
        else
            echo "   Format: ℹ️  Mock file (simulation mode)"
        fi
        echo ""
        
        # Provide usage examples
        echo "🔧 Usage Examples:"
        echo ""
        echo "1. Test with llama.cpp (if real quantization):"
        echo "   ./llama.cpp/main -m $output_file -p \"Hello, I am Llama 2\""
        echo ""
        echo "2. Test with llama-cpp-python:"
        echo "   python3 -c \""
        echo "   from llama_cpp import Llama"
        echo "   llm = Llama(model_path='$output_file')"
        echo "   response = llm('Hello, I am', max_tokens=50)"
        echo "   print(response['choices'][0]['text'])"
        echo "   \""
        echo ""
        echo "3. Use with Oumi inference:"
        echo "   oumi infer --model $output_file --engine LLAMACPP"
        echo ""
        
    else
        echo "❌ Output file not found: $output_file"
    fi
    
else
    echo ""
    echo "❌ Quantization failed!"
    echo ""
    echo "Common solutions:"
    echo "1. For missing dependencies:"
    echo "   pip install autoawq torch transformers accelerate"
    echo ""
    echo "2. For memory issues:"
    echo "   - Use smaller batch_size in config (try batch_size: 4 or 2)"
    echo "   - Ensure 16+ GB RAM available"
    echo "   - Close other applications"
    echo ""
    echo "3. For access issues to Llama 2:"
    echo "   - Accept Llama 2 license on HuggingFace"
    echo "   - Login with: huggingface-cli login"
    echo ""
fi

echo "📚 For more information:"
echo "   - Configuration: examples/quantization/llama2_7b_awq_example.yaml"
echo "   - Documentation: examples/quantization/README.md"
echo "   - Install dependencies: examples/quantization/install_awq_deps.sh"