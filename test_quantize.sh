#!/bin/bash

echo "🧪 Testing Oumi Quantize Command Interface"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
test_count=0
pass_count=0
fail_count=0

run_test() {
    test_count=$((test_count + 1))
    echo -e "\n${BLUE}📋 Test $test_count: $1${NC}"
    echo "Command: $2"
    echo "----------------------------------------"

    if eval "$2"; then
        echo -e "${GREEN}✅ Test $test_count PASSED${NC}"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}❌ Test $test_count FAILED${NC}"
        fail_count=$((fail_count + 1))
    fi
}

run_error_test() {
    test_count=$((test_count + 1))
    echo -e "\n${BLUE}📋 Test $test_count: $1${NC}"
    echo "Command: $2"
    echo "Expected: Should fail gracefully"
    echo "----------------------------------------"

    if eval "$2" 2>/dev/null; then
        echo -e "${RED}❌ Test $test_count FAILED (should have failed)${NC}"
        fail_count=$((fail_count + 1))
    else
        echo -e "${GREEN}✅ Test $test_count PASSED (correctly failed)${NC}"
        pass_count=$((pass_count + 1))
    fi
}

# Test 1: Basic CLI usage
run_test "Basic CLI Usage" \
    "oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output test1.gguf"

# Test 2: Config file usage
run_test "Config File Usage" \
    "oumi quantize --config quantize_config.yaml"

# Test 3: Config override
run_test "Config Override" \
    "oumi quantize --config quantize_config.yaml --method q8_0"

# Test 4: Different methods
echo -e "\n${YELLOW}🔄 Testing Different Quantization Methods${NC}"
for method in q4_0 q4_1 q5_0 q8_0 f16; do
    run_test "Method $method" \
        "oumi quantize --method $method --model microsoft/DialoGPT-small --output test_${method}.gguf"
done

# Test 5: Different models
echo -e "\n${YELLOW}🔄 Testing Different Models${NC}"
run_test "GPT-2 Model" \
    "oumi quantize --method q4_0 --model gpt2 --output gpt2_test.gguf"

run_test "Large Model" \
    "oumi quantize --method q4_0 --model meta-llama/Llama-2-7b-hf --output llama_test.gguf"

# Test 6: Different output formats
echo -e "\n${YELLOW}🔄 Testing Different Output Formats${NC}"
run_test "GGUF Output" \
    "oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output test.gguf"

run_test "Safetensors Output" \
    "oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output test.safetensors"

run_test "PyTorch Output" \
    "oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output test.pt"

# Test 7: Log levels
echo -e "\n${YELLOW}🔄 Testing Log Levels${NC}"
run_test "Debug Logging" \
    "oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output debug.gguf --log-level DEBUG"

run_test "Warning Logging" \
    "oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output warn.gguf --log-level WARNING"

# Test 8: Error scenarios
echo -e "\n${YELLOW}🔄 Testing Error Scenarios${NC}"
run_error_test "Missing Model" \
    "oumi quantize --method q4_0 --output error_test.gguf"

run_error_test "Invalid Method" \
    "oumi quantize --method invalid_method --model microsoft/DialoGPT-small --output error_test.gguf"

run_error_test "Nonexistent Config" \
    "oumi quantize --config nonexistent_config.yaml"

run_error_test "Invalid Model" \
    "oumi quantize --method q4_0 --model invalid/model-name --output error_test.gguf"

# Test 9: Help commands
echo -e "\n${YELLOW}🔄 Testing Help Commands${NC}"
run_test "Main Help" \
    "oumi --help > /dev/null"

run_test "Quantize Help" \
    "oumi quantize --help > /dev/null"

run_test "Quantize Help Short" \
    "oumi quantize -h > /dev/null"

# Test 10: Multiple parameter combinations
echo -e "\n${YELLOW}🔄 Testing Parameter Combinations${NC}"
run_test "All Parameters" \
    "oumi quantize --method q8_0 --model microsoft/DialoGPT-small --output combo_test.gguf --log-level INFO"

run_test "Config + Multiple Overrides" \
    "oumi quantize --config quantize_config.yaml --method q8_0 --model gpt2 --output multi_override.gguf"

# Summary
echo -e "\n${BLUE}===========================================${NC}"
echo -e "${BLUE}🎯 TEST SUMMARY${NC}"
echo -e "${BLUE}===========================================${NC}"
echo -e "Total Tests: $test_count"
echo -e "${GREEN}Passed: $pass_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"

if [ $fail_count -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    echo -e "${GREEN}The quantize command interface is working correctly!${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please review the output above.${NC}"
    exit 1
fi
