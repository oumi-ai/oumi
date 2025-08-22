# Inference Test Data

This directory contains test data used by the inference engine integration tests.

## File Structure

- **`model_configs.json`** - Model configurations for testing different model types
  - `gemma_270m` - Standard Gemma 270M model configuration
  - `smollm_135m` - SmolLM 135M instruction-tuned model
  - `gemma_270m_gguf` - GGUF quantized Gemma model for LlamaCpp tests

- **`test_conversations.json`** - Standard test conversations used across tests
  - Simple greeting with keyword validation
  - Multi-turn math conversation
  - Story prompt with natural language requirements

- **`generation_params.json`** - Standard generation parameters for consistent testing
  - Optimized for fast, deterministic test execution
  - Short token limits and reproducible settings

## Usage

These files are automatically loaded by the utility functions in `tests/integration/infer/inference_test_utils.py`:

- `get_test_models()` → loads `model_configs.json`
- `create_test_conversations()` → loads `test_conversations.json` 
- `get_test_generation_params()` → loads `generation_params.json`

## Benefits

- **Maintainability**: Test data can be modified without touching Python code
- **Reusability**: Same test data used across multiple test files
- **Visibility**: Changes to test cases are clear in version control
- **Separation of Concerns**: Test data separated from test logic

## Adding New Test Data

To add new models, conversations, or parameters:

1. Edit the appropriate JSON file
2. Follow the existing structure and naming conventions
3. The utility functions will automatically load the new data
4. No Python code changes required