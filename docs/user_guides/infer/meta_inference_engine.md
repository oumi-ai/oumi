# MetaInferenceEngine - Simplified Model Inference

The `MetaInferenceEngine` provides a unified interface for running inference with different language models. It simplifies the process of working with multiple model providers by automatically selecting the appropriate inference engine based on the model name.

## Basic Usage

Here's how to run inference with various models using a single engine:

```python
from oumi.inference import MetaInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role

# Create a simple conversation
conversation = Conversation(messages=[
    Message(role=Role.USER, content="Explain quantum computing in simple terms.")
])

# Initialize the engine with generation parameters
engine = MetaInferenceEngine(temperature=0.7, max_tokens=1000)

# Run inference with different models
for model_name in ["gpt-4o", "claude-3-sonnet", "gemini-pro"]:
    response = engine.infer([conversation], model_name=model_name)
    print(f"\n=== {model_name} response ===")
    print(response[0].messages[-1].content)
```

## Benefits

- **Simplified Interface**: Consistent API across different model providers
- **Automatic Engine Selection**: Chooses the right engine based on model name pattern
- **Parameter Normalization**: Handles common API parameter differences (e.g., `max_tokens` → `max_new_tokens`)
- **Engine Caching**: Reuses engines for the same model to improve performance

## Supported Models

The `MetaInferenceEngine` automatically selects the appropriate engine based on model name pattern:

| Model Pattern | Engine |
|---------------|--------|
| `gpt-*`, `text-*`, `o1-*` | OpenAI |
| `claude-*` | Anthropic |
| `gemini-*` | Google Gemini |
| `llama-*`, `meta-llama/*` | VLLM (falls back to Native) |
| `mistral-*`, `mixtral-*`, etc. | VLLM (falls back to Native) |
| Other models | Native |

## Authentication

For models requiring API keys:

```python
# For OpenAI
response = engine.infer(
    [conversation], 
    model_name="gpt-4", 
    remote_params={"api_key": "your-openai-key"}
)

# For Anthropic
response = engine.infer(
    [conversation], 
    model_name="claude-3-opus", 
    remote_params={"api_key": "your-anthropic-key"}
)
```

## Advanced Usage

### Using with Custom InferenceConfig

You can also use the `MetaInferenceEngine` with a custom `InferenceConfig`:

```python
from oumi.core.configs import InferenceConfig, GenerationParams, ModelParams

# Create a custom inference config
config = InferenceConfig(
    model=ModelParams(model_name="dummy-will-be-replaced"),  # Will be replaced with model_name
    generation=GenerationParams(
        temperature=0.8,
        max_new_tokens=2000,
        top_p=0.95
    )
)

# Use with MetaInferenceEngine
engine = MetaInferenceEngine()
response = engine.infer(
    [conversation],
    model_name="gpt-4o",  # This overrides config.model.model_name
    inference_config=config
)
```

### Using with Local Models

For local models, the engine will automatically use VLLM if available:

```python
engine = MetaInferenceEngine(temperature=0.7, max_tokens=1000)

# Local Llama model
response = engine.infer(
    [conversation],
    model_name="meta-llama/Llama-3-8b"
)
```