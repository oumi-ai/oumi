#!/bin/bash

echo "🚀 Quick Quantize Command Tests"
echo "================================"

# Test 1: Basic CLI (should work)
echo "Test 1: Basic CLI usage"
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output quick_test.gguf
echo ""

# Test 2: Config file (should work)
echo "Test 2: Config file usage"
oumi quantize --config quantize_config.yaml
echo ""

# Test 3: Missing model (should fail)
echo "Test 3: Missing model (should fail)"
oumi quantize --method q4_0 --output error_test.gguf 2>&1 || echo "✅ Correctly failed"
echo ""

# Test 4: Invalid method (should fail)
echo "Test 4: Invalid method (should fail)"
oumi quantize --method invalid --model microsoft/DialoGPT-small --output error_test.gguf 2>&1 || echo "✅ Correctly failed"
echo ""

# Test 5: Help command
echo "Test 5: Help command"
oumi quantize --help | head -10
echo ""

echo "🎉 Quick tests complete!"
