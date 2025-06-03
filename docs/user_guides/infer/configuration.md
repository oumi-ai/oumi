# Inference Configuration

## Introduction

This guide covers the configuration options available for inference in Oumi. The configuration system is designed to be:

- **Modular**: Each aspect of inference (model, generation, remote settings) is configured separately
- **Type-safe**: All configuration options are validated at runtime
- **Flexible**: Supports various inference scenarios from local to remote inference
- **Extensible**: Easy to add new configuration options and validate them

The configuration system is built on the {py:obj}`~oumi.core.configs.inference_config.InferenceConfig` class, which contains all inference settings. This class is composed of several parameter classes:

- [Model Configuration](#model-configuration): Model architecture and loading settings via {py:obj}`~oumi.core.configs.params.model_params.ModelParams`
- [Generation Configuration](#generation-configuration): Text generation parameters via {py:obj}`~oumi.core.configs.params.generation_params.GenerationParams`
- [Remote Configuration](#remote-configuration): Remote API settings via {py:obj}`~oumi.core.configs.params.remote_params.RemoteParams`

All configuration files in Oumi are YAML files, which provide a human-readable format for specifying inference settings. The configuration system automatically validates these files and converts them to the appropriate Python objects.

## Basic Structure

A typical configuration file has this structure:

```yaml
model:  # Model settings
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  trust_remote_code: true
  model_kwargs:
    device_map: "auto"
    torch_dtype: "float16"

generation:  # Generation parameters
  max_new_tokens: 100
  temperature: 0.7
  top_p: 0.9
  batch_size: 1

engine: "VLLM"  # VLLM, LLAMACPP, NATIVE, REMOTE_VLLM, etc.

remote_params:  # Optional remote settings
  api_url: "https://api.example.com/v1"
  api_key: "${API_KEY}"
  connection_timeout: 20.0
```

## Configuration Components

### Model Configuration

Configure the model architecture and loading using the {py:obj}`~oumi.core.configs.params.model_params.ModelParams` class:

```yaml
model:
  # Required
  model_name: "meta-llama/Llama-3.1-8B-Instruct"    # Model ID or path (REQUIRED)

  # Model loading
  adapter_model: null                                # Path to adapter model (auto-detected if model_name is adapter)
  tokenizer_name: null                               # Custom tokenizer name/path (defaults to model_name)
  tokenizer_pad_token: null                          # Override pad token
  tokenizer_kwargs: {}                               # Additional tokenizer args
  model_max_length: null                             # Max sequence length (positive int or null)
  load_pretrained_weights: true                      # Load pretrained weights
  trust_remote_code: false                           # Allow remote code execution (use with trusted models only)

  # Model precision and hardware
  torch_dtype_str: "float32"                         # Model precision (float32/float16/bfloat16/float64)
  device_map: "auto"                                 # Device placement strategy (auto/null)
  compile: false                                     # JIT compile model

  # Attention and optimization
  attn_implementation: null                          # Attention impl (null/sdpa/flash_attention_2/eager)
  enable_liger_kernel: false                         # Enable Liger CUDA kernel for potential speedup

  # Model behavior
  chat_template: null                                # Chat formatting template
  freeze_layers: []                                  # Layer names to freeze during training

  # Additional settings
  model_kwargs: {}                                   # Additional model constructor args
```

### Generation Configuration

Configure text generation parameters using the {py:obj}`~oumi.core.configs.params.generation_params.GenerationParams` class:

```yaml
generation:
  max_new_tokens: 256                # Maximum number of new tokens to generate (default: 256)
  batch_size: 1                      # Number of sequences to generate in parallel (default: 1)
  exclude_prompt_from_response: true # Whether to remove the prompt from the response (default: true)
  seed: null                        # Seed for random number determinism (default: null)
  temperature: 0.0                  # Controls randomness in output (0.0 = deterministic) (default: 0.0)
  top_p: 1.0                       # Nucleus sampling probability threshold (default: 1.0)
  frequency_penalty: 0.0           # Penalize repeated tokens (default: 0.0)
  presence_penalty: 0.0            # Penalize tokens based on presence in text (default: 0.0)
  stop_strings: null               # List of sequences to stop generation (default: null)
  stop_token_ids: null            # List of token IDs to stop generation (default: null)
  logit_bias: {}                  # Token-level biases for generation (default: {})
  min_p: 0.0                      # Minimum probability threshold for tokens (default: 0.0)
  use_cache: false                # Whether to use model's internal cache (default: false)
  num_beams: 1                    # Number of beams for beam search (default: 1)
  use_sampling: false             # Whether to use sampling vs greedy decoding (default: false)
  guided_decoding: null           # Parameters for guided decoding (default: null)
```

```{note}
Not all inference engines support all generation parameters. Each engine has its own set of supported parameters which can be checked via the `get_supported_params` attribute of the engine class. For example:
- {py:obj}`NativeTextInferenceEngine <oumi.inference.NativeTextInferenceEngine.get_supported_params>`
- {py:obj}`VLLMInferenceEngine <oumi.inference.VLLMInferenceEngine.get_supported_params>`
- {py:obj}`RemoteInferenceEngine <oumi.inference.RemoteInferenceEngine.get_supported_params>`

Please refer to the specific engine's documentation for details on supported parameters.
```

### Remote Configuration

Configure remote API settings using the {py:obj}`~oumi.core.configs.params.remote_params.RemoteParams` class:

```yaml
remote_params:
  api_url: "https://api.example.com/v1"   # Required: URL of the API endpoint
  api_key: "your-api-key"                 # API key for authentication
  api_key_env_varname: null               # Environment variable for API key
  max_retries: 3                          # Maximum number of retries
  connection_timeout: 20.0                # Request timeout in seconds
  num_workers: 1                          # Number of parallel workers
  politeness_policy: 0.0                  # Sleep time between requests
  batch_completion_window: "24h"          # Time window for batch completion
```

### Engine Selection

The `engine` parameter specifies which inference engine to use. Available options from {py:obj}`~oumi.core.configs.inference_engine_type.InferenceEngineType`:

- `ANTHROPIC`: Use Anthropic's API via {py:obj}`~oumi.inference.AnthropicInferenceEngine`
- `DEEPSEEK`: Use DeepSeek Platform API via {py:obj}`~oumi.inference.DeepSeekInferenceEngine`
- `GOOGLE_GEMINI`: Use Google Gemini via {py:obj}`~oumi.inference.GoogleGeminiInferenceEngine`
- `GOOGLE_VERTEX`: Use Google Vertex AI via {py:obj}`~oumi.inference.GoogleVertexInferenceEngine`
- `LAMBDA`: Use Lambda AI API via {py:obj}`~oumi.inference.LambdaInferenceEngine`
- `LLAMACPP`: Use llama.cpp for CPU inference via {py:obj}`~oumi.inference.LlamaCppInferenceEngine`
- `NATIVE`: Use native PyTorch inference via {py:obj}`~oumi.inference.NativeTextInferenceEngine`
- `OPENAI`: Use OpenAI API via {py:obj}`~oumi.inference.OpenAIInferenceEngine`
- `PARASAIL`: Use Parasail API via {py:obj}`~oumi.inference.ParasailInferenceEngine`
- `REMOTE_VLLM`: Use external vLLM server via {py:obj}`~oumi.inference.RemoteVLLMInferenceEngine`
- `REMOTE`: Use any OpenAI-compatible API via {py:obj}`~oumi.inference.RemoteInferenceEngine`
- `SAMBANOVA`: Use SambaNova API via {py:obj}`~oumi.inference.SambanovaInferenceEngine`
- `SGLANG`: Use SGLang inference engine via {py:obj}`~oumi.inference.SGLangInferenceEngine`
- `TOGETHER`: Use Together API via {py:obj}`~oumi.inference.TogetherInferenceEngine`
- `VLLM`: Use vLLM for optimized local inference via {py:obj}`~oumi.inference.VLLMInferenceEngine`

### Additional Configuration

The following top-level parameters are also available in the configuration:

```yaml
# Input/Output paths
input_path: null    # Path to input file containing prompts (JSONL format)
output_path: null   # Path to save generated outputs
```

The `input_path` should contain prompts in JSONL format, where each line is a JSON representation of an Oumi `Conversation` object.

## See Also

- {doc}`/user_guides/infer/inference_engines` for local and remote inference engines usage
- {doc}`/user_guides/infer/common_workflows` for common workflows
- {doc}`/user_guides/infer/configuration` for detailed parameter documentation
