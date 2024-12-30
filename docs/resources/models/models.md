# Models

```{toctree}
:maxdepth: 2
:caption: Models
:hidden:

supported_models
custom_models
```


Oumi aims to provide a unified interface for working with foundation models from multiple providers (`HuggingFace`, `Meta`, `NanoGPT`, etc.), as well as your own custom models for inference, fine-tuning, pre-training, evaluation, and more.

Out-of-the-box, you can use most popular causal language models, and multimodal vision-language models from the HuggingFace `transformers` library.

In this guide, we will briefly go over the available models, and how to work with them in oumi.

## Available Models

Every model shares a common interface, simplifying development across different architectures:

```python
# Example using the unified interface
from oumi.builders import build_model
from oumi.core.configs import ModelParams

# Configure and build any supported model
model_params = ModelParams(model_name="meta-llama/Llama-3.2-3B-Instruct")

model = build_model(model_params)

# Use the same interface regardless of model type
outputs = model.generate(input_ids)
```

### HuggingFace Hub Integration

Oumi integrates with the HuggingFace Hub and HuggingFace `transformers` library, allowing you to use any model available on the platform:

```python
from oumi.builders import build_model, build_tokenizer
from oumi.core.configs import ModelParams

# Configure model parameters
model_params = ModelParams(model_name="meta-llama/Llama-3.2-3B-Instruct")

# Build model and tokenizer
model = build_model(model_params)
tokenizer = build_tokenizer(model_params)
```

The {py:func}`oumi.builders.build_model` and {py:func}`oumi.builders.build_tokenizer` functions provide a unified interface for model creation. Configure your model using {py:class}`oumi.core.configs.ModelParams`.

### Supported Models

The framework includes optimized implementations of popular models. See our {doc}`/resources/recipes` for detailed configuration examples and best practices.

### Custom Models

You can easily create custom models by extending our base classes. See the {doc}`/resources/models/custom_models` guide for details:

```python
from oumi.core.models import BaseModel

class MyCustomModel(BaseModel):
    """Create your own model architecture."""
    def __init__(self, config):
        super().__init__(config)
        # Define your architecture
```

For detailed implementation guidance, see the {doc}`/resources/models/custom_models` documentation.

## Advanced Topics
### Tokenizer Integration

The framework provides consistent tokenizer handling through the {py:mod}`core.tokenizers` module. Tokenizers can be configured independently of models while maintaining compatibility:

```python
from builders import build_tokenizer
from core.configs import ModelParams

# Configure tokenizer with model
model_params = ModelParams(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",  # Optional: use different tokenizer
    model_max_length=4096,                               # Set custom max length
    chat_template="llama3-instruct"                      # Specify chat template
)

# Build tokenizer with settings
tokenizer = build_tokenizer(model_params)
```

See {py:func}`core.tokenizers.get_default_special_tokens` for information about special token handling.

For advanced model configuration, see {py:class}`oumi.core.configs.ModelParams` and {py:class}`oumi.core.configs.PeftParams` for PEFT/LoRA support.

### Model Adapters and Quantization

Oumi supports loading models with PEFT adapters and quantization for efficient inference. You can configure these through `ModelParams`:

```python
from oumi.core.configs import ModelParams, PeftParams

# Load a model with a PEFT adapter
model_params = ModelParams(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    adapter_model="path/to/adapter",  # Load PEFT adapter
)

# Load a model with 4-bit quantization
model_params = ModelParams(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    peft_params=PeftParams(
        q_lora=True,  # Enable quantization
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
)

# Build the model with adapter/quantization
model = build_model(model_params)
```

The framework supports:
- **PEFT Adapters**: Load trained LoRA or other PEFT adapters using the `adapter_model` parameter
- **Quantization**: Enable 4-bit quantization through `PeftParams` with `q_lora=True`
- **Mixed Precision**: Control model precision using `torch_dtype` parameter

For more details on training with adapters and quantization, see {doc}`/user_guides/train/configuration`.

### Chat Templates
Oumi uses Jinja2 templates to format conversations for different model architectures. These templates ensure that messages are formatted correctly for each model's expected input format.

Available templates include:
- `default` - Basic template without special tokens
- `llama3-instruct` - For Llama 3 instruction models
- `llava` - For LLaVA multimodal models
- `phi3-instruct` - For Phi-3 instruction models
- `qwen2-vl-instruct` - For Qwen2-VL instruction models
- `zephyr` - For Zephyr models

All the templates expect a `messages` list, where each message is a dictionary with `role` and `content` keys in `oumi` format.

Here's an example of the Llama3 template:

````{dropdown} src/oumi/datasets/chat_templates/llama3-instruct.jinja
```{literalinclude} /../src/oumi/datasets/chat_templates/llama3-instruct.jinja
:language: jinja
```
````

You can find all supported templates in the {file}`src/oumi/datasets/chat_templates` directory. Each template is designed to match the training format of its corresponding model architecture.

## Next Steps

For more detailed information about working with models, see:
- {doc}`/resources/recipes` - Detailed configuration examples
- {doc}`/user_guides/train/train` - Model fine-tuning guide
- {doc}`/user_guides/evaluate/evaluate` - Model evaluation and benchmarking
- {doc}`/user_guides/infer/infer` - Inference
