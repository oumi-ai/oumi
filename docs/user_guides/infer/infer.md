# Inference

```{toctree}
:maxdepth: 2
:caption: Inference
:hidden:

inference_engines
common_workflows
configuration
inference_cli
```

Oumi Infer provides a unified interface for running models, whether you're deploying models locally or calling external APIs. It handles the complexity of different backends and providers while maintaining a consistent interface for both batch and interactive workflows.

## Why Use Oumi Infer?

Running models in production environments presents several challenges that Oumi helps address:

- **Universal Model Support**: Run models locally (vLLM, LlamaCPP, Transformers) or connect to hosted APIs (Anthropic, Gemini, OpenAI, Together, Parasail, Vertex AI, SambaNova) through a single, consistent interface
- **Production-Ready**: Support for batching, retries, error-handling, structured outputs, and high-performance inference via multi-threading to hit a target throughput.
- **Scalable Architecture**: Deploy anywhere from a single GPU to distributed systems without code changes
- **Unified Configuration**: Control all aspects of model execution through a single config file

## Quick Start

Let's jump right in with a simple example. Here's how to run interactive inference using the CLI:

```bash
oumi infer -i -c configs/recipes/smollm/inference/135m_infer.yaml
```

Or use the Python API for a basic chat interaction:

```{testcode} python
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role

# Initialize with a small, free model
engine = VLLMInferenceEngine(
    ModelParams(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        model_kwargs={"device_map": "auto"}
    )
)

# Create a conversation
conversation = Conversation(
    messages=[Message(role=Role.USER, content="What is Oumi?")]
)

# Get response
result = engine.infer_online([conversation], InferenceConfig())
print(result[0].messages[-1].content)
```

## Core Concepts

### System Architecture

The inference system is built around three main components:

1. **Inference Engines**: Handle model execution and generation
2. **Conversation Format**: Structure inputs and outputs
3. **Configuration System**: Manage model and runtime settings

Here's how these components work together:

```python
# 1. Initialize engine
engine = VLLMInferenceEngine(model_params)

# 2. Prepare input
conversation = Conversation(messages=[...])

# 3. Configure inference
config = InferenceConfig(...)

# 4. Run inference
result = engine.infer_online([conversation], config)

# 5. Process output
response = result[0].messages[-1].content
```

### Inference Engines

Inference Engines are simple tools for running inference on models in Oumi. This includes newly trained models, downloaded pretrained models, and even remote APIs such as Anthropic, Gemini, and OpenAI.

#### Choosing an Engine

Our engines are broken into two categories: local inference vs remote inference. But how do you decide between the two?

Generally, the answer is simple: if you have sufficient resources to run the model locally without OOMing, then use a local engine like {py:obj}`~oumi.inference.VLLMInferenceEngine`, {py:obj}`~oumi.inference.NativeTextInferenceEngine`, or {py:obj}`~oumi.inference.LlamaCppInferenceEngine`.

If you don't have enough local compute resources, then the model must be hosted elsewhere. Our remote inference engines assume that your model is hosted behind a remote API. You can use {py:obj}`~oumi.inference.AnthropicInferenceEngine`, {py:obj}`~oumi.inference.GoogleGeminiInferenceEngine`, or {py:obj}`~oumi.inference.GoogleVertexInferenceEngine`, {py:obj}`~oumi.inference.SambanovaInferenceEngine`..., to call their respective APIs. You can also use {py:obj}`~oumi.inference.RemoteInferenceEngine` to call any API implementing the OpenAI Chat API format (including OpenAI's native API), or use {py:obj}`~oumi.inference.SGLangInferenceEngine` or {py:obj}`~oumi.inference.RemoteVLLMInferenceEngine` to call external SGLang or vLLM servers started remotely or locally outside of Oumi.

For a comprehensive list of engines, see the [Supported Engines](#supported-engines) section below.

```{note}
Still unsure which engine to use? Try {py:obj}`~oumi.inference.VLLMInferenceEngine` to get started locally.
```

#### Loading an Engine

Now that you've decided on the engine you'd like to use, you'll need to create a small config to instantiate your engine.

All engines require a model, specified via {py:obj}`~oumi.core.configs.ModelParams`. Any engine calling an external API / service (such as Anthropic, Gemini, OpenAI, or a self-hosted server) will also require {py:obj}`~oumi.core.configs.RemoteParams`.

See {py:obj}`~oumi.inference.NativeTextInferenceEngine` for an example of a local inference engine.

See {py:obj}`~oumi.inference.AnthropicInferenceEngine` for an example of an inference engine that requires a remote API.

See {py:obj}`~oumi.inference.SambanovaInferenceEngine` for an example of an inference engine that requires a remote API.
```python
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role

model_params = ModelParams(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
engine = VLLMInferenceEngine(model_params)
conversation = Conversation(
    messages=[Message(role=Role.USER, content="What is Oumi?")]
)

inference_config = InferenceConfig()
output_conversations = engine.infer_online(
    input=[conversation], inference_config=inference_config
)
print(output_conversations)
```

#### Input Data

Oumi supports several input formats for inference:

1. JSONL files
    - Prepare a JSONL file with your inputs, where each line is a JSON object containing your input data.
    - See {doc}`/resources/datasets/data_formats` for more details.
2. Interactive console input
    - To run inference interactively, use the `oumi infer` command with the `-i` flag.

    ```{code-block} bash
    oumi infer -c infer_config.yaml -i
    ```

## Supported Engines

```{include} /api/summary/inference_engines.md
```

## Advanced Topics

### Inference with Quantized Models

```{code-block} yaml
model:
  model_name: "model.gguf"

engine: LLAMACPP

generation:
  temperature: 0.7
  batch_size: 1
```

```{warning}
Ensure the selected inference engine supports the specific quantization method used in your model.
```

(multi-modal-inference)=
### Multi-modal Inference

For models that support multi-modal inputs (e.g., text and images):

```python
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import InferenceConfig, InferenceEngineType, ModelParams, GenerationParams
from oumi.core.types.conversation import Conversation, ContentItem, Message, Role, Type

model_params = ModelParams(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_max_length=1024,
    chat_template="llava",
)

engine = VLLMInferenceEngine(model_params)
```

```python
input_conversation = Conversation(
    messages=[
        Message(
            role=Role.USER,
            content=[
                ContentItem(
                    content="https://oumi.ai/the_great_wave_off_kanagawa.jpg",
                    type=Type.IMAGE_URL,
                ),
                ContentItem(content="Describe this image", type=Type.TEXT),
            ],
        )
    ]
)
inference_config = InferenceConfig(
    model=model_params,
    generation=GenerationParams(max_new_tokens=64),
    engine=InferenceEngineType.VLLM,
)
output_conversations = engine.infer_online(
    input=[input_conversation], inference_config=inference_config
)
print(output_conversations)
```

To run multimodal inference interactively, use the `oumi infer` command with the `-i` and `--image` flags.

```{code-block} bash
oumi infer -c infer_config.yaml -i --image="https://oumi.ai/the_great_wave_off_kanagawa.jpg"
```

### Distributed Inference

For large-scale inference across multiple GPUs or machines, see the following tutorial
for inference with Llama 3.3 70B on {gh}`<GitHub> notebooks/Oumi - Using vLLM Engine for Inference.ipynb`.

## Next Steps

- Learn about the supported {doc}`inference_engines`
- Review {doc}`common_workflows` for practical examples
- See the {doc}`configuration` section for detailed configuration options.
