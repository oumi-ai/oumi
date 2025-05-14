# Models

```{toctree}
:maxdepth: 2
:caption: Models
:hidden:

supported_models
custom_models
```

Oumi provides a _unified_ interface for working with foundation models from multiple providers, including `HuggingFace`, `Meta`, `NanoGPT`, or your own custom models. Whether you’re performing inference, fine-tuning, pre-training, or evaluation, Oumi simplifies the process with _seamless_ integrations.

Out-of-the-box, Oumi supports popular causal LLMs and large vision-language models, with optimized implementations available for efficient use. For a comprehensive list of supported models, configuration examples, and best practices, see the {doc}`/resources/recipes` page.

This guide provides a quick overview of Oumi’s unified interface, demonstrating how to instantiate models, customize their parameters, configure underlying tokenizers, and more, enabling seamless integration with your applications.

## Main Model Interface

Using the functions {py:func}`oumi.builders.build_model` and {py:func}`oumi.builders.build_tokenizer`, you can instantiate models and tokenizers, regardless of their architecture. To further configure and customize a model, you can use the {py:class}`oumi.core.configs.ModelParams` class.

```python
# Example using Oumi's main model interface
import torch
from oumi.builders import build_model, build_tokenizer
from oumi.core.configs import ModelParams

# Specify parameters to customize your model
model_params = ModelParams(model_name="HuggingFaceTB/SmolLM-135M", tokenizer_kwargs={'pad_token': '<|endoftext|>'})

# Build the model
device = torch.device("cpu") # or gpu, or mps, etc.
model = build_model(model_params).to(device)

# Build a corresponding tokenizer
tokenizer = build_tokenizer(model_params)
input_data = tokenizer("What are the benefits of open source coding?", return_tensors="pt")

# Use the same interface regardless of model type for generation
outputs = model.generate(input_data['input_ids'].to(device), attention_mask=input_data['attention_mask'].to(device), max_length=64)
print(tokenizer.decode(outputs[0]))
```

### Hugging Face Hub Integration

Oumi integrates directly with the Hugging Face Hub and Hugging Face `transformers` library, allowing you to use any model available on Hugging Face Hub:

```python
from oumi.builders import build_model, build_tokenizer
from oumi.core.configs import ModelParams

# Configure model parameters
model_params = ModelParams(model_name="meta-llama/Llama-3.2-3B-Instruct")

# Build model and tokenizer
model = build_model(model_params)
tokenizer = build_tokenizer(model_params)
```

### Custom Models

You can also easily create custom models by extending our base classes:

```python
from oumi.core.models import BaseModel

class MyCustomModel(BaseModel):
    """Create your own model architecture."""
    def __init__(self, config):
        super().__init__(config)
        # Define your architecture
```

For detailed implementation guidance on this subject, see the {doc}`/resources/models/custom_models` documentation.

## Advanced Topics

### Tokenizer Integration

Oumi ensures consistent tokenizer handling through the {py:mod}`core.tokenizers` module. Tokenizers can be configured independently of models while maintaining full compatibility.

```python
from builders import build_tokenizer
from core.configs import ModelParams

# Configure tokenizer with model
model_params = ModelParams(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",   # Optional: use different tokenizer
    model_max_length=4096,                               # Set custom max length
    chat_template="llama3-instruct"                      # Specify chat template
)

# Build tokenizer with settings
tokenizer = build_tokenizer(model_params)
```

For details on handling special tokens, refer to {py:func}`core.tokenizers.get_default_special_tokens`.

### Parameter Adapters and Quantization

Oumi supports loading models with [PEFT adapters](https://arxiv.org/pdf/2403.14608){target="_blank"} and quantization for efficiency purposes. You can configure these through `ModelParams`:

```python
from oumi.core.configs import ModelParams, PeftParams

# Load a model with a PEFT adapter
model_params = ModelParams(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    adapter_model="path/to/adapter",  # Load PEFT adapter
)

# Load a model with 8-bit quantization
model_params = ModelParams(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    peft_params=PeftParams(
        q_lora=True,  # Enable quantization
        q_lora_bits=8,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
)

# Build the model with adapter/quantization
model = build_model(model_params)
```

The framework supports:

- **PEFT Adapters**: Load trained LoRA or other PEFT adapters using the `adapter_model` parameter
- **Quantization**: Enable 8-bit (or 4-bit) quantization through `PeftParams` with `q_lora` and `q_lora_bits`
- **Mixed Precision**: Control model precision using `torch_dtype` parameter

For more details on training with adapters and quantization, see {doc}`/user_guides/train/configuration`.

### Chat Templates

Oumi uses Jinja2 templates to format conversations for different model architectures. Oumi's default templates ensure that messages are formatted correctly for each model's expected input format.

Available templates include:

- `default` - Basic template without special tokens
- `llama3-instruct` - For Llama 3 instruction models
- `llava` - For LLaVA multimodal models
- `phi3-instruct` - For Phi-3 instruction models
- `qwen2-vl-instruct` - For Qwen2-VL instruction models
- `zephyr` - For Zephyr models

All the templates expect a `messages` list, where each message is a dictionary with `role` and `content` keys in {doc}`oumi format </resources/datasets/data_formats>`.

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
- {doc}`/user_guides/infer/infer` - Inference guide
