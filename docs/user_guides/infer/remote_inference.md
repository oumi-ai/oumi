# Remote Inference

This guide covers setting up and using remote inference options.

Remote inference is particularly useful for for deploying your own fine-tuned models in distributed and cloud environments, or accessing frontier models from Anthropic, Vertex AI, and OpenAI for benchmarking and evaluation.

## Quick Setup

1. Make sure to install any extra dependencies:

```bash
# For remote VLLM
pip install vllm

# For Google Vertex AI
pip install "oumi[gcp]"
```

2. Basic usage:

```python
from oumi.inference import AnthropicInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = AnthropicInferenceEngine(
    model_params=ModelParams(
        model_name="claude-3-5-sonnet-20240620",
    ),
    remote_params=RemoteParams(
        api_url="https://api.anthropic.com/v1/messages",
        api_key_env_varname="ANTHROPIC_API_KEY",
    )
)
```


## Cloud APIs

### Anthropic (Claude)

[Claude](https://www.anthropic.com/claude) is Anthropic's advanced language model, available through their API:

```python
from oumi.inference import AnthropicInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

# Basic configuration
engine = AnthropicInferenceEngine(
    model_params=ModelParams(
        model_name="claude-3-5-sonnet-20240620"  # Latest model as of March 2024
    ),
    remote_params=RemoteParams(
        api_url="https://api.anthropic.com/v1/messages",
        api_key_env_varname="ANTHROPIC_API_KEY",
    )
)
```

### Google Vertex AI

[Gemini](https://ai.google/gemini/) can be accessed either through Google Cloud's [Vertex AI](https://cloud.google.com/vertex-ai) or directly through the Gemini API:

```python
# Direct Gemini API access
engine = RemoteInferenceEngine(
    model_params=ModelParams(
        model_name="gemini-1.5-flash"
    ),
    remote_params=RemoteParams(
        api_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        api_key_env_varname="GEMINI_API_KEY",
    )
)

# Through Vertex AI
engine = GoogleVertexInferenceEngine(
    model_params=ModelParams(
        model_name="google/gemini-1.5-pro"
    ),
    remote_params=RemoteParams(
        api_url="https://<your-region>-aiplatform.googleapis.com/v1beta1/projects/<your-project>/locations/<your-region>/endpoints/openapi/chat/completions",
    )
)
```

### OpenAI

[OpenAI](https://platform.openai.com/) provides access to powerful models like GPT-4:

```python
from oumi.inference import RemoteInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

# Basic configuration
engine = RemoteInferenceEngine(
    model_params=ModelParams(
        model_name="gpt-4o-mini"  # or gpt-4o for the larger model
    ),
    remote_params=RemoteParams(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key_env_varname="OPENAI_API_KEY",
        timeout=60
    )
)

# Production configuration with retries
engine = RemoteInferenceEngine(
    model_params=ModelParams(
        model_name="gpt-4o"
    ),
    remote_params=RemoteParams(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key_env_varname="OPENAI_API_KEY",
        timeout=60,
        retry_count=3,
        retry_delay=1
    )
)
```


## Remote VLLM

[vLLM](https://github.com/vllm-project/vllm) can be deployed as a server, providing high-performance inference capabilities over HTTP. This section covers different deployment scenarios and configurations.

### Server Setup

1. **Basic Server** - Suitable for development and testing:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000
```

2. **Multi-GPU Server** - For large models requiring multiple GPUs:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000
```


### Client Configuration

The client can be configured with different reliability and performance options:

```python
# Basic client with timeout and retry settings
engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
    ),
    remote_params=RemoteParams(
        api_url="http://localhost:8000",
    )
)
```

## See Also

- {doc}`configuration` for detailed configuration options
- {doc}`common_workflows` for practical usage examples
- {doc}`local_inference` for running models locally
- [vLLM Documentation](https://vllm.readthedocs.io/) for advanced vLLM configurations
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api) for Claude API details
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs) for Google Cloud AI services
- [OpenAI API Documentation](https://platform.openai.com/docs) for OpenAI API details
- [Gemini API Documentation](https://ai.google.dev/docs) for Gemini API details
