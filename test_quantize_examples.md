# Quantize Command Testing Examples

🚧 **Status**: All commands run in simulation mode for interface testing

## Basic CLI Testing

### Test 1: Simple CLI Usage
```bash
# Basic quantization with minimal parameters
oumi quantize --method q4_0 --model meta-llama/Llama-2-7b-hf --output test1.gguf
```

### Test 2: Different Methods
```bash
# Test 4-bit quantization
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output dialogpt-q4.gguf

# Test 8-bit quantization
oumi quantize --method q8_0 --model microsoft/DialoGPT-small --output dialogpt-q8.gguf

# Test 16-bit float
oumi quantize --method f16 --model microsoft/DialoGPT-small --output dialogpt-f16.gguf
```

### Test 3: Different Model Sources
```bash
# HuggingFace model
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output hf-model.gguf

# Local model (if you have one)
oumi quantize --method q4_0 --model ./my_model --output local-model.gguf

# Larger model (just for interface testing)
oumi quantize --method q4_0 --model meta-llama/Llama-2-13b-hf --output llama13b.gguf
```

## Configuration File Testing

### Test 4: Basic Config File
```bash
# Use the provided config file
oumi quantize --config quantize_config.yaml
```

### Test 5: Config Override Testing
```bash
# Override method
oumi quantize --config quantize_config.yaml --method q8_0

# Override output path
oumi quantize --config quantize_config.yaml --output new-output.gguf

# Override model
oumi quantize --config quantize_config.yaml --model microsoft/DialoGPT-small

# Override multiple parameters
oumi quantize --config quantize_config.yaml --method q8_0 --model microsoft/DialoGPT-small --output multi-override.gguf
```

## Error Testing

### Test 6: Error Scenarios
```bash
# Missing model (should fail)
oumi quantize --method q4_0 --output test.gguf

# Invalid method (should fail)
oumi quantize --method invalid_method --model microsoft/DialoGPT-small --output test.gguf

# Invalid model (should fail)
oumi quantize --method q4_0 --model non/existent-model --output test.gguf

# Missing config file (should fail)
oumi quantize --config non_existent_config.yaml
```

## Help and Information Testing

### Test 7: Help Commands
```bash
# Main help
oumi --help

# Quantize help
oumi quantize --help

# Quantize help (short form)
oumi quantize -h
```

## Logging Level Testing

### Test 8: Different Log Levels
```bash
# Debug logging
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output debug.gguf --log-level DEBUG

# Warning logging
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output warn.gguf --log-level WARNING

# Error logging only
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output error.gguf --log-level ERROR
```

## Output Format Testing

### Test 9: Different Output Extensions
```bash
# GGUF format (default)
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output model.gguf

# Test with different extensions (interface testing)
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output model.safetensors
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output model.pt
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output model.pth
```

## Comprehensive Testing Script

### Test 10: Automated Testing
Save this as `test_quantize.sh`:

```bash
#!/bin/bash

echo "🧪 Testing Oumi Quantize Command Interface"
echo "==========================================="

# Test 1: Basic CLI
echo -e "\n📋 Test 1: Basic CLI Usage"
oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output test1.gguf

# Test 2: Config file
echo -e "\n📋 Test 2: Config File Usage"
oumi quantize --config quantize_config.yaml

# Test 3: Config override
echo -e "\n📋 Test 3: Config Override"
oumi quantize --config quantize_config.yaml --method q8_0

# Test 4: Different methods
echo -e "\n📋 Test 4: Different Methods"
for method in q4_0 q4_1 q5_0 q8_0 f16; do
    echo "Testing method: $method"
    oumi quantize --method $method --model microsoft/DialoGPT-small --output test_$method.gguf
done

# Test 5: Error scenarios
echo -e "\n📋 Test 5: Error Scenarios"
echo "Testing missing model (should fail):"
oumi quantize --method q4_0 --output error_test.gguf || echo "✅ Correctly failed"

echo "Testing invalid method (should fail):"
oumi quantize --method invalid --model microsoft/DialoGPT-small --output error_test.gguf || echo "✅ Correctly failed"

echo -e "\n🎉 Testing Complete!"
```

Make it executable and run:
```bash
chmod +x test_quantize.sh
./test_quantize.sh
```

## Expected Output Examples

### Successful Command Output:
```
Starting quantization of model: microsoft/DialoGPT-small
Quantization method: q4_0
Output path: test.gguf
WARNING: Quantization feature is currently in development.
INFO: Simulating quantization process for validation...
✅ Model quantized successfully!
📁 Output saved to: test.gguf
📊 Status: simulated
```

### Error Output Examples:
```
# Missing model
Error: Either --config must be provided or --model must be specified

# Invalid method
ValueError: Unsupported quantization method: invalid_method

# Invalid model
ValueError: Model not found: non/existent-model
```

## Performance Testing

### Test 11: Stress Testing
```bash
# Test with various model sizes (interface only)
oumi quantize --method q4_0 --model gpt2 --output gpt2.gguf
oumi quantize --method q4_0 --model microsoft/DialoGPT-medium --output dialogpt-medium.gguf
oumi quantize --method q4_0 --model microsoft/DialoGPT-large --output dialogpt-large.gguf

# Test rapid commands
for i in {1..5}; do
    oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output rapid_test_$i.gguf
done
```

## Configuration Variations Testing

### Test 12: Custom Config Files
Create different config files for testing:

**test_configs/minimal_config.yaml:**
```yaml
model:
  model_name: "gpt2"
method: "q4_0"
output_path: "minimal_test.gguf"
```

**test_configs/verbose_config.yaml:**
```yaml
model:
  model_name: "microsoft/DialoGPT-small"
  tokenizer_name: "microsoft/DialoGPT-small"
  trust_remote_code: false
method: "q8_0"
output_path: "verbose_test.gguf"
output_format: "gguf"
verbose: true
```

**test_configs/safetensors_config.yaml:**
```yaml
model:
  model_name: "microsoft/DialoGPT-small"
method: "q8_0"
output_path: "safetensors_test/"
output_format: "safetensors"
verbose: true
```

Test these configs:
```bash
mkdir -p test_configs

# Test minimal config
oumi quantize --config test_configs/minimal_config.yaml

# Test verbose config
oumi quantize --config test_configs/verbose_config.yaml

# Test safetensors config
oumi quantize --config test_configs/safetensors_config.yaml
```

## Integration Testing

### Test 13: CLI Integration
```bash
# Test with other oumi commands for comparison
echo "Testing oumi command integration:"

# List all commands
oumi --help

# Compare with other commands
oumi infer --help
oumi train --help
oumi quantize --help

# Test environment
oumi env
```

## Documentation Testing

### Test 14: Documentation Validation
```bash
# Test that examples from documentation work
echo "Testing documentation examples:"

# Example 1 from docs
oumi quantize --method q4_0 --model meta-llama/Llama-2-7b-hf --output llama2-q4.gguf

# Example 2 from docs
oumi quantize --config quantize_config.yaml

# Example 3 from docs
oumi quantize --config quantize_config.yaml --method q8_0 --output different_output.gguf
```

## What to Observe

When running these tests, check for:

✅ **Success Indicators:**
- Commands complete without crashes
- Clear warning about development status
- Simulation mode messages
- Input validation working
- Configuration loading successful
- CLI arguments properly parsed

❌ **Failure Indicators:**
- Unexpected crashes or exceptions
- Missing error messages for invalid inputs
- Config files not loading properly
- CLI arguments not being recognized

🔍 **Areas to Focus On:**
1. **Input Validation**: Do invalid inputs get caught properly?
2. **Configuration Loading**: Do YAML files parse correctly?
3. **CLI Override Logic**: Do CLI args properly override config values?
4. **Error Messages**: Are error messages clear and helpful?
5. **Development Status**: Is it clear this is in simulation mode?

This comprehensive testing suite will help validate the entire quantize command interface before implementing the core quantization logic!
