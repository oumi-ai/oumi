# Inference FAQ

Common questions and solutions for running inference with Oumi.

## Engine Selection

### Which inference engine should I use?

| Scenario | Recommended Engine |
|----------|-------------------|
| Production GPU deployment | vLLM |
| CPU-only or edge devices | LlamaCPP |
| Quick prototyping | Native |
| Remote server deployment | Remote vLLM or SGLang |
| Cloud API access | Anthropic, OpenAI, etc. |

See {doc}`/user_guides/infer/engine_comparison` for a detailed comparison.

### How do I switch between engines?

Simply change the engine class:

```python
from oumi.inference import (
    VLLMInferenceEngine,
    NativeTextInferenceEngine,
    LlamaCppInferenceEngine,
)
from oumi.core.configs import ModelParams

model_params = ModelParams(model_name="meta-llama/Llama-3.2-1B-Instruct")

# Using vLLM
engine = VLLMInferenceEngine(model_params)

# Using Native
engine = NativeTextInferenceEngine(model_params)
```

Or via CLI:

```bash
oumi infer --engine VLLM --model.model_name meta-llama/Llama-3.2-1B-Instruct
oumi infer --engine NATIVE --model.model_name meta-llama/Llama-3.2-1B-Instruct
```

## Memory Issues

### I'm running out of GPU memory during inference

1. **Use quantization**:

    ```python
    model_params = ModelParams(
        model_name="model-name",
        model_kwargs={"load_in_4bit": True}
    )
    ```

2. **Reduce batch size**:

    ```python
    generation_params = GenerationParams(batch_size=1)
    ```

3. **Use vLLM's memory management**:

    ```python
    model_params = ModelParams(
        model_name="model-name",
        model_kwargs={"gpu_memory_utilization": 0.8}
    )
    ```

### How do I check GPU memory usage?

```bash
nvidia-smi
# Or for continuous monitoring:
watch -n 1 nvidia-smi
```

## Model Loading

### My model isn't loading

Common issues:

1. **Wrong model name**: Verify the exact HuggingFace model ID
2. **Authentication required**: Log in to HuggingFace for gated models
3. **Insufficient disk space**: Models need to be downloaded first

```bash
# Log in to HuggingFace
huggingface-cli login

# Check disk space
df -h
```

### How do I use a local model?

Point to the local directory:

```python
model_params = ModelParams(
    model_name="./path/to/local/model"
)
```

### How do I load a LoRA adapter?

Specify the `adapter_model` parameter:

```python
model_params = ModelParams(
    model_name="meta-llama/Llama-3.1-8B-Instruct",  # Base model
    adapter_model="./path/to/lora/adapter"          # LoRA adapter
)
```

## Generation Parameters

### How do I control output length?

Use `GenerationParams`:

```python
from oumi.core.configs import GenerationParams

generation_params = GenerationParams(
    max_new_tokens=512,      # Maximum tokens to generate
    min_new_tokens=10,       # Minimum tokens to generate
)
```

### How do I adjust creativity/randomness?

Control sampling parameters:

```python
generation_params = GenerationParams(
    temperature=0.7,         # Higher = more creative (0.0-2.0)
    top_p=0.9,              # Nucleus sampling threshold
    top_k=50,               # Top-k sampling
)
```

| Goal | Temperature | top_p |
|------|-------------|-------|
| Deterministic | 0.0 | 1.0 |
| Balanced | 0.7 | 0.9 |
| Creative | 1.0+ | 0.95 |

### How do I use greedy decoding?

Set temperature to 0:

```python
generation_params = GenerationParams(
    temperature=0.0,
    do_sample=False
)
```

## Input/Output Formatting

### How do I format chat messages?

Use the standard conversation format:

```python
from oumi.core.types import Conversation, Message, Role

conversation = Conversation(
    messages=[
        Message(role=Role.SYSTEM, content="You are helpful."),
        Message(role=Role.USER, content="Hello!"),
    ]
)

response = engine.infer([conversation])
```

### How do I process multiple inputs in batch?

Pass a list of conversations:

```python
conversations = [
    Conversation(messages=[Message(role=Role.USER, content="Question 1")]),
    Conversation(messages=[Message(role=Role.USER, content="Question 2")]),
    Conversation(messages=[Message(role=Role.USER, content="Question 3")]),
]

responses = engine.infer(conversations)
```

### How do I stream responses?

Use async streaming:

```python
async for token in engine.infer_stream(conversation):
    print(token, end="", flush=True)
```

## Remote Inference

### How do I connect to a remote vLLM server?

```python
from oumi.inference import RemoteVLLMInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(model_name="model-name"),
    remote_params=RemoteParams(
        api_url="http://your-server:8000",
        max_retries=3,
    )
)
```

### How do I handle API rate limits?

Configure rate limiting in `RemoteParams`:

```python
remote_params = RemoteParams(
    api_url="http://api-endpoint",
    requests_per_minute=100,
    input_tokens_per_minute=100000,
    output_tokens_per_minute=50000,
)
```

### My API calls are timing out

Increase timeout and retry settings:

```python
remote_params = RemoteParams(
    api_url="http://api-endpoint",
    max_retries=5,
    connection_timeout=60.0,
)
```

## Vision Language Models

### How do I run inference with images?

Use a VLM-capable engine with image inputs:

```python
from oumi.core.types import ContentItem, Type

message = Message(
    role=Role.USER,
    content=[
        ContentItem(type=Type.IMAGE_URL, content="https://example.com/image.jpg"),
        ContentItem(type=Type.TEXT, content="What's in this image?"),
    ]
)
```

See {doc}`/user_guides/infer/common_workflows` for VLM examples.

## Performance

### How do I improve inference throughput?

1. **Use vLLM** for GPU inference (continuous batching)
2. **Enable prefix caching**:

    ```python
    model_kwargs={"enable_prefix_caching": True}
    ```

3. **Use tensor parallelism** for large models:

    ```python
    model_kwargs={"tensor_parallel_size": 2}
    ```

4. **Batch requests** instead of single queries

### How do I reduce latency?

1. **Use a smaller model** (e.g., 1B instead of 7B)
2. **Use quantization** (4-bit or 8-bit)
3. **Reduce `max_new_tokens`**
4. **Use speculative decoding** if supported

## See Also

- {doc}`/user_guides/infer/inference_engines` - Detailed engine documentation
- {doc}`/user_guides/infer/engine_comparison` - Engine comparison guide
- {doc}`/user_guides/infer/configuration` - Configuration options
- {doc}`/user_guides/infer/common_workflows` - Common inference patterns
