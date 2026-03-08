# Common Workflows

This guide provides practical examples for common inference tasks in Oumi, focusing on real-world use cases and patterns.

The examples below can be used with multiple different inference engines. Check out the {doc}`inference_engines` page to learn about the available engines and how to choose between them based on your latency, cost, and privacy requirements.

## Chat Completion

The most common use case is interactive chat using foundation models. You can set up a basic chat system, using the `VLLMInferenceEngine` engine. You can easily switch between different inference engines by changing the engine in the `engine` variable, and customize the generation by updating the `GenerationParams` config:

```{code-block} python
:emphasize-lines:  1, 6,7, 8, 9, 10, 11, 21, 22, 23, 24, 25

from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import InferenceConfig, ModelParams, GenerationParams
from oumi.core.types.conversation import Conversation, Message, Role

# Initialize engine
engine = VLLMInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype="bfloat16"
    )
)

# Create a conversation with system and user messages
conversation = Conversation(messages=[
    Message(role=Role.SYSTEM, content="You are a helpful coding assistant."),
    Message(role=Role.USER, content="How do I read a file in Python?")
])

# Configure generation parameters
config = InferenceConfig(
    generation=GenerationParams(
        max_new_tokens=512,  # Maximum response length
        temperature=0.7,     # Control randomness
        top_p=0.9            # Nucleus sampling threshold
    )
)

# Get model response
result = engine.infer([conversation], config)
print(result[0].messages[-1].content)
```

## Structured Outputs

Multiple scenarios require parsing model outputs into structured data. Instead of manually parsing the output via a regex, which can be error prone and brittle, you can use structured decoding via the `GuidedDecodingParams` class.

Structured decoding helps ensure the model outputs data in a consistent, parseable format that matches your schema. This reduces errors from malformed outputs and eliminates the need for post-processing. As a bonus, since the model is constrained to only generate valid JSON following your schema, it can sometimes generate responses more quickly than with free-form text generation.

Here's an example of how to use guided decoding to generate data in a structured format:

```{code-block} python
:emphasize-lines: 1, 7, 11, 12, 13, 14, 15, 16, 25, 49

from pydantic import BaseModel
from oumi.inference import OpenAIInferenceEngine
from oumi.core.configs import (
    ModelParams,
    RemoteParams,
    GenerationParams,
    GuidedDecodingParams,
)


# Define output schema using Pydantic
class ProductInfo(BaseModel):
    name: str
    price: float
    features: list[str]
    color: str


config = InferenceConfig(
    model=ModelParams(model_name="gpt-4o-mini"),

    generation=GenerationParams(
        max_new_tokens=512,
        temperature=0,  # Use deterministic output for structured data
        guided_decoding=GuidedDecodingParams(json=ProductInfo.model_json_schema()),
    ),
)

# Configure engine for JSON output
engine = OpenAIInferenceEngine(
    model_params=config.model,
    remote_params=config.remote_params,
)

# Extract and validate structured data
text = (
    "I'd like to tell you about the new iPhone 15. It's priced at $999 and comes with "
    "some amazing features including the A17 Pro chip, USB-C connectivity, and a premium "
    "Titanium design."
)
conversation = Conversation(
    messages=[
        Message(
            role=Role.USER, content=f"Extract product information as JSON from: {text}"
        )
    ]
)
result = engine.infer([conversation], inference_config=config)
product = ProductInfo.model_validate_json(result[0].messages[-1].content)
```

## Parallel Processing with Multiple Workers

For high-throughput scenarios where you need to process many requests concurrently, you can leverage multiple workers:

```{code-block} python
:emphasize-lines: 8, 9, 10

from oumi.inference import OpenAIInferenceEngine
from oumi.core.configs import InferenceConfig, RemoteParams, ModelParams

# Configure engine with multiple workers
config = InferenceConfig(
    model=ModelParams(model_name="gpt-4"),
    remote_params=RemoteParams(
        max_retries=3,  # Number of retry attempts on failure
        num_workers=4,  # Process 4 requests concurrently
        politeness_policy=1.  # Sleep duration in seconds after an error
    )
)

engine = OpenAIInferenceEngine(
    model_params=config.model,
    remote_params=config.remote_params
)
```

The remote inference engine includes built-in error handling and rate limiting features:

- `max_retries`: Number of times to retry a failed request before giving up (default: 3)
- `politeness_policy`: Time in seconds to wait after encountering an error before retrying (default: 0)
- `num_workers`: Number of concurrent workers processing requests

Most inference providers have a limit on the number of requests per minute you can make. You can configure the `num_workers` to match your provider's limit. For example, if your provider allows 100 requests per minute, you can set `num_workers=100` and `politeness_policy=60.0` to ensure you don't exceed the limit.

## Async Batch Inference

Async Batch Inference allows you to upload up to 50,000 requests at a time, which will be processed in bulk by the inference engine. You can check-in the status of the batch job, and retrieve the results once the batch is complete.

This is useful for non-interactive workloads, or when you need to process a large number of requests in a short period of time.

Batch processing offers several key advantages over real-time inference with multiple workers: all the error handling, rate limiting, and throughput batching are handled for you, and it's typically cheaper than manually managing concurrent real-time requests.

Here's an OpenAI example of how to use Oumi's async batch inference:

```{code-block} python
:emphasize-lines: 8,24

from oumi.inference import OpenAIInferenceEngine
from oumi.core.configs import InferenceConfig, RemoteParams, ModelParams

# Initialize engine with batch settings
config = InferenceConfig(
    model=ModelParams(model_name="gpt-4"),
    remote_params=RemoteParams(
        batch_completion_window="24h"  # Time window for processing
    )
)

engine = OpenAIInferenceEngine(
    model_params=config.model,
    remote_params=config.remote_params
)

# Prepare batch of conversations
conversations = [
    Conversation(messages=[Message(content="What is 2+2?", role=Role.USER)]),
    Conversation(messages=[Message(content="What is the capital of France?", role=Role.USER)])
]

# The only difference with online inference is using `infer_batch` instead of `infer`
batch_id = engine.infer_batch(conversations, config)
```

### Checking Batch Status

You can check the status of your batch job:

```python
status = engine.get_batch_status(batch_id, config)
print(f"Progress: {status.completed_requests}/{status.total_requests}")
print(f"Status: {status.status}")
```

### Processing Results

Once the batch is complete, you can process the results:

```python
# Get results for completed batch
results = engine.get_batch_results(batch_id, conversations, config)

# Process results
for conv in results:
    print(f"Question: {conv.messages[-2].content}")  # User message
    print(f"Answer: {conv.messages[-1].content}")    # Assistant response
    print()
```

### Batch Processing Details

```{list-table}
:header-rows: 1

* - Status Code
  - Description
* - `VALIDATING`
  - Input validation in progress
* - `IN_PROGRESS`
  - Processing requests
* - `COMPLETED`
  - Results available
* - `FAILED`
  - Processing failed
* - `EXPIRED`
  - Time window exceeded
* - `CANCELLED`
  - Manually cancelled
```

For more examples and detailed API documentation, see the [OpenAI Batch API documentation](https://platform.openai.com/docs/api-reference/batch).

## Serving LoRA Fine-Tuned Models

If you've fine-tuned a model using LoRA (Low-Rank Adaptation), you can serve the adapted model efficiently without loading the full fine-tuned weights. This is particularly useful for deploying multiple variants of a base model.

### Local LoRA Serving with vLLM

```{code-block} python
:emphasize-lines: 6, 7

from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import ModelParams, GenerationParams, InferenceConfig
from oumi.core.types.conversation import Conversation, Message, Role

# Initialize engine with LoRA adapter
engine = VLLMInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",  # Base model
        adapter_model="path/to/lora/adapter",           # Your LoRA adapter
        torch_dtype_str="bfloat16"
    )
)

# Create conversation
conversation = Conversation(messages=[
    Message(role=Role.USER, content="What is machine learning?")
])

# Run inference with the adapted model
config = InferenceConfig(
    generation=GenerationParams(
        max_new_tokens=256,
        temperature=0.7
    )
)

result = engine.infer([conversation], config)
print(result[0].messages[-1].content)
```

### Using LoRA with Configuration Files

You can also specify the LoRA adapter in a YAML configuration file:

```yaml
model:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  adapter_model: "path/to/lora/adapter"
  torch_dtype_str: "bfloat16"

generation:
  max_new_tokens: 256
  temperature: 0.7

engine: VLLM
```

Then run inference using the CLI:

```bash
oumi infer -i -c config.yaml
```

### Remote LoRA Serving

For production deployments, you can serve LoRA adapters using a remote vLLM server:

**1. Start the vLLM server with LoRA support:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 6864 \
    --enable-lora \
    --lora-modules my-adapter=path/to/lora/adapter
```

**2. Connect from the client:**

```python
from oumi.inference import RemoteVLLMInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(
        model_name="my-adapter"  # Use the adapter name from server
    ),
    remote_params=RemoteParams(
        api_url="http://localhost:6864"
    )
)
```

### Serving Multiple LoRA Adapters

You can serve multiple LoRA adapters from a single vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 6864 \
    --enable-lora \
    --lora-modules \
        adapter-v1=path/to/adapter-v1 \
        adapter-v2=path/to/adapter-v2 \
        adapter-v3=path/to/adapter-v3
```

Then switch between adapters by changing the `model_name`:

```python
# Use adapter-v1
engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(model_name="adapter-v1"),
    remote_params=RemoteParams(api_url="http://localhost:6864")
)

# Or use adapter-v2
engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(model_name="adapter-v2"),
    remote_params=RemoteParams(api_url="http://localhost:6864")
)
```

### Important Notes

- The base model specified in `model_name` must match the base model used during LoRA fine-tuning
- Not all model architectures support LoRA adapters in vLLM - check the [vLLM documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for compatibility
- LoRA adapters can be stored locally or on HuggingFace Hub (e.g., `"username/model-lora-adapter"`)

## See Also

- {doc}`configuration` for configuration options
- {doc}`inference_engines` for local and remote inference engines
