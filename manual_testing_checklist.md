# Manual Testing Checklist for Quantize Command

## Pre-Testing Setup
- [ ] Ensure you're in the oumi conda environment: `conda activate oumi`
- [ ] Verify quantize command is available: `oumi --help` (should show quantize)
- [ ] Check current directory has config files: `ls *.yaml`

## Core Functionality Tests

### ✅ CLI-Only Usage
- [ ] **Test 1**: `oumi quantize --method q4_0 --model microsoft/DialoGPT-small --output test1.gguf`
  - Should show development warnings
  - Should validate model identifier
  - Should complete with simulation message

- [ ] **Test 2**: `oumi quantize --method q8_0 --model gpt2 --output test2.gguf`
  - Should work with different method
  - Should work with different model

- [ ] **Test 3**: `oumi quantize --method f16 --model meta-llama/Llama-2-7b-hf --output test3.gguf`
  - Should work with larger model
  - Should validate HuggingFace model ID

### ✅ Config File Usage
- [ ] **Test 4**: `oumi quantize --config quantize_config.yaml`
  - Should load config successfully
  - Should show config values in output

- [ ] **Test 5**: `oumi quantize --config test_configs/minimal_config.yaml`
  - Should work with minimal config
  - Should use default values for missing fields

- [ ] **Test 6**: `oumi quantize --config test_configs/verbose_config.yaml`
  - Should work with detailed config
  - Should show verbose logging if enabled

### ✅ Config Override Tests
- [ ] **Test 7**: `oumi quantize --config quantize_config.yaml --method q8_0`
  - Should override method from config
  - Should keep other config values

- [ ] **Test 8**: `oumi quantize --config quantize_config.yaml --output override.gguf`
  - Should override output path
  - Should keep other config values

- [ ] **Test 9**: `oumi quantize --config quantize_config.yaml --model gpt2 --method q8_0 --output multi.gguf`
  - Should override multiple parameters
  - Should show final combined configuration

### ✅ Error Handling Tests
- [ ] **Test 10**: `oumi quantize --method q4_0 --output error.gguf` (no model)
  - Should fail with clear error message
  - Should suggest providing model or config

- [ ] **Test 11**: `oumi quantize --method invalid_method --model gpt2 --output error.gguf`
  - Should fail with unsupported method error
  - Should list supported methods

- [ ] **Test 12**: `oumi quantize --config nonexistent.yaml`
  - Should fail with file not found error
  - Should show clear error message

- [ ] **Test 13**: `oumi quantize --method q4_0 --model invalid/model --output error.gguf`
  - Should fail with model not found error
  - Should validate model identifier

### ✅ Help and Documentation
- [ ] **Test 14**: `oumi --help`
  - Should show quantize in command list
  - Should show development status indicator

- [ ] **Test 15**: `oumi quantize --help`
  - Should show development warning
  - Should show all available options
  - Should show examples

- [ ] **Test 16**: `oumi quantize -h`
  - Should work as shorthand for help
  - Should show same information

### ✅ Different Parameters
- [ ] **Test 17**: All quantization methods
  ```bash
  oumi quantize --method q4_0 --model gpt2 --output q4_0.gguf
  oumi quantize --method q4_1 --model gpt2 --output q4_1.gguf
  oumi quantize --method q5_0 --model gpt2 --output q5_0.gguf
  oumi quantize --method q5_1 --model gpt2 --output q5_1.gguf
  oumi quantize --method q8_0 --model gpt2 --output q8_0.gguf
  oumi quantize --method f16 --model gpt2 --output f16.gguf
  oumi quantize --method f32 --model gpt2 --output f32.gguf
  ```

- [ ] **Test 18**: Different log levels
  ```bash
  oumi quantize --method q4_0 --model gpt2 --output debug.gguf --log-level DEBUG
  oumi quantize --method q4_0 --model gpt2 --output info.gguf --log-level INFO
  oumi quantize --method q4_0 --model gpt2 --output warning.gguf --log-level WARNING
  ```

- [ ] **Test 19**: Different output formats
  ```bash
  oumi quantize --method q4_0 --model gpt2 --output model.gguf
  oumi quantize --method q4_0 --model gpt2 --output model.safetensors
  oumi quantize --method q4_0 --model gpt2 --output model.pt
  ```

## Expected Behavior Checklist

### ✅ What Should Work
- [ ] All commands complete without crashes
- [ ] Clear warning about development status shown
- [ ] Input validation catches invalid parameters
- [ ] Config files load and parse correctly
- [ ] CLI arguments override config values
- [ ] Model identifiers are validated (may take time for HF models)
- [ ] Help commands show useful information
- [ ] Error messages are clear and actionable

### ✅ What Should Be Shown
- [ ] Development warning: "Quantization feature is currently in development"
- [ ] Simulation message: "Simulating quantization process for validation"
- [ ] Success message: "✅ Model quantized successfully!"
- [ ] Status: "📊 Status: simulated"
- [ ] Configuration values being used
- [ ] Clear progress indication

### ✅ What Should Fail Gracefully
- [ ] Missing required parameters (model when no config)
- [ ] Invalid quantization methods
- [ ] Non-existent config files
- [ ] Invalid model identifiers
- [ ] Unsupported parameter combinations

## Performance Observations

### ✅ Speed Tests
- [ ] Commands complete quickly (< 5 seconds for small models)
- [ ] Large model validation may take longer (expected)
- [ ] Config file loading is fast
- [ ] Help commands are instant

### ✅ Resource Usage
- [ ] No excessive memory usage during simulation
- [ ] No hanging processes
- [ ] Clean exit after completion
- [ ] No temporary files left behind

## Output Quality Check

### ✅ Log Messages
- [ ] Timestamps are present and accurate
- [ ] Log levels are appropriate
- [ ] Messages are informative and clear
- [ ] No confusing or misleading messages

### ✅ User Experience
- [ ] Interface is intuitive
- [ ] Error messages help user fix issues
- [ ] Progress is clearly communicated
- [ ] Development status is transparent

## Automated Testing

### ✅ Run Test Scripts
- [ ] `./quick_test.sh` - Basic functionality
- [ ] `./test_quantize.sh` - Comprehensive testing
- [ ] Check exit codes and output

### ✅ Integration Testing
- [ ] Test with other oumi commands for comparison
- [ ] Verify no conflicts with existing functionality
- [ ] Check that quantize appears in main help

## Final Validation

### ✅ Ready for Development
- [ ] All interface tests pass
- [ ] Configuration system works correctly
- [ ] Error handling is comprehensive
- [ ] Documentation matches behavior
- [ ] CLI follows oumi patterns consistently

### ✅ Development Notes
- [ ] Record any unexpected behavior
- [ ] Note any confusing error messages
- [ ] Identify areas for improvement
- [ ] Test edge cases thoroughly

---

## Quick Command Reference

```bash
# Basic testing
oumi quantize --method q4_0 --model gpt2 --output test.gguf
oumi quantize --config quantize_config.yaml
oumi quantize --help

# Error testing
oumi quantize --method q4_0 --output test.gguf  # Should fail
oumi quantize --method invalid --model gpt2 --output test.gguf  # Should fail

# Automated testing
./quick_test.sh
./test_quantize.sh
```

This checklist ensures comprehensive testing of the quantize command interface before implementing the core quantization logic!
