# Custom Models

This guide explains how to create custom models in Oumi. We'll use the `MLPEncoder` as a concrete example to demonstrate best practices and requirements.

## Core Concepts

Before diving into the implementation, let's understand the key concepts and components:

1. **Base Model Interface**
   - All Oumi models inherit from {py:class}`oumi.core.models.BaseModel`
   - This provides a consistent interface and enforces implementation of required methods
   - The base class extends {external:class}`torch.nn.Module` for PyTorch compatibility

2. **Model Registry**
   - Models are registered using {py:class}`oumi.core.registry.Registry`
   - This makes models discoverable and instantiatable by name
   - Registration is done via the {py:func}`oumi.core.registry.register` decorator

3. **Model Outputs**
   - Models return dictionaries containing {external:class}`torch.Tensor`
   - Required outputs depend on the task (e.g., logits for classification)
   - Loss is included in outputs during training

4. **Loss Functions**
   - Models define their training criterion via the `criterion` property
   - Common losses are available in {external:mod}`torch.nn.functional`
   - Custom losses can be implemented as needed

5. **Integration Points**
   - Models work with {external:class}`torch.utils.data.DataLoader` for data loading
   - Compatible with {external:class}`transformers.Trainer` for training
   - Support distributed training via PyTorch's mechanisms


## Configuration

The configuration is part of the overall {py:obj}`~oumi.core.configs.training_config.TrainingConfig` and is defined under the `model` section:

```yaml
model:
  # Required
  model_name: "my_custom_model"    # Model ID or path
  model_kwargs:                    # Parameters passed to model constructor
    input_dim: 768
    hidden_dim: 128
    output_dim: 10

  # Optional settings
  trust_remote_code: false         # Allow remote code execution
  torch_dtype_str: "float32"       # Model precision
  device_map: "auto"              # Device placement strategy
```

Key points about configuration:
- Model parameters are defined in the `model` section of training config
- `model_kwargs` contains parameters passed to the model's constructor
- Configuration can be loaded from YAML files or created programmatically

## Implementing a Custom Model
### Overview

At a high level, an Oumi model:

1. Inherits from {py:class}`oumi.core.models.BaseModel` (which extends {external:class}`torch.nn.Module`)
2. Implements a forward pass that returns a dictionary of tensors
3. Defines a loss criterion for training
4. Follows PyTorch and (optionally) Hugging Face conventions

Here's the complete implementation of the {py:class}`oumi.models.mlp.MLPEncoder`, a simple text encoder model:

````{dropdown} MLPEncoder
```python
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

from oumi.core import registry
from oumi.core.models.base_model import BaseModel

@registry.register("MLPEncoder", registry.RegistryType.MODEL)
class MLPEncoder(BaseModel):
    def __init__(
        self, input_dim: int = 768, hidden_dim: int = 128, output_dim: int = 10
    ):
        """Initialize the MLPEncoder.

        Args:
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            output_dim (int): The output dimension.
        """
        super().__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the MLP model.

        Args:
            input_ids (torch.LongTensor): The input tensor of shape
                (batch_size, sequence_length).
            labels (torch.LongTensor, optional): The target labels tensor
                of shape (batch_size,).
            **kwargs: Additional keyword arguments provided by the tokenizer.
                Not used in this model.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the model outputs.
                The dictionary has the following keys:
                - "logits" (torch.Tensor): The output logits tensor of
                  shape (batch_size, num_classes).
                - "loss" (torch.Tensor, optional): The computed loss tensor
                  if labels is not None.
        """
        x = self.embedding(input_ids)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        outputs = {"logits": logits}

        if labels is not None:
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1
            )
            outputs["loss"] = loss

        return outputs

    @property
    def criterion(self) -> Callable:
        """Returns the criterion function for the MLP model.

        The criterion function is used to compute the loss during training.

        Returns:
            torch.nn.CrossEntropyLoss: The cross-entropy loss function.
        """
        return F.cross_entropy
```
````

### Implementation Breakdown

Let's break down each component of the implementation:

#### 1. Model Registration and Base Class

```python
@registry.register("MLPEncoder", registry.RegistryType.MODEL)
class MLPEncoder(BaseModel):
```

- Uses {py:class}`oumi.core.registry.Registry` for model registration
- Inherits from {py:class}`oumi.core.models.BaseModel`
- Registration enables model discovery and instantiation by name

#### 2. Model Architecture

```python
def __init__(self, input_dim: int = 768, hidden_dim: int = 128, output_dim: int = 10):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, hidden_dim)
    self.fc1 = nn.Linear(hidden_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_dim, output_dim)
```

- Uses standard PyTorch layers from {external:mod}`torch.nn`:
  - {external:class}`torch.nn.Embedding` for input embedding
  - {external:class}`torch.nn.Linear` for fully connected layers
  - {external:class}`torch.nn.ReLU` for activation
  - {external:class}`torch.nn.Module` for module composition
- Follows PyTorch's module composition pattern

#### 3. Forward Pass

```python
def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, **kwargs) -> dict[str, torch.Tensor]:
```

- Implements the required {external:meth}`torch.nn.Module.forward` method
- Takes input tensors and optional labels
- Returns a dictionary with model outputs
- Computes loss during training if labels are provided

#### 4. Loss Function

```python
@property
def criterion(self) -> Callable:
    return F.cross_entropy
```

- Uses {external:func}`torch.nn.functional.cross_entropy` for training
- Implemented as a property following {py:class}`oumi.core.models.BaseModel` interface
- Can be customized for different training objectives

## Using Custom Models via the CLI

See {doc}`/user_guides/customization` to quickly enable your model when using the CLI.

## Testing Models

Oumi uses pytest for testing models. Here's an example test for the MLPEncoder:

```python
import pytest
import torch
from oumi.models import MLPEncoder

def test_mlp_encoder_forward():
    """Test forward pass of MLPEncoder."""
    # Initialize model
    model = MLPEncoder(input_dim=100, hidden_dim=32, output_dim=10)

    # Create dummy inputs
    batch_size, seq_len = 16, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    labels = torch.randint(0, 10, (batch_size,))

    # Test forward pass without labels
    outputs = model(input_ids=input_ids)
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, 10)

    # Test forward pass with labels
    outputs = model(input_ids=input_ids, labels=labels)
    assert "loss" in outputs
    assert outputs["loss"].shape == ()
```

## Using the Model

There are two main ways to use custom models in Oumi:

### 1. Using `build_model`

You can create model instances programmatically using the `build_model` function:

```python
from oumi.builders import build_model
from oumi.core.configs import ModelParams

# Create model parameters
model_params = ModelParams(
    model_name="MLPEncoder",  # Name used in @registry.register
    model_kwargs={
        "input_dim": 1000,
        "hidden_dim": 256,
        "output_dim": 10,
    },
    torch_dtype_str="float32",
    device_map="auto",
)

# Build the model
model = build_model(model_params=model_params)
```

### 2. Using Training Configuration

More commonly, you'll define the model as part of a training configuration:

```yaml
# train_config.yaml
model:
  model_name: "MLPEncoder"
  model_kwargs:
    input_dim: 1000
    hidden_dim: 256
    output_dim: 10
  torch_dtype_str: "float32"
  device_map: "auto"

data:
  train:
    datasets:
      - dataset_name: "text_classification"
        dataset_path: "path/to/data"
        split: "train"

training:
  output_dir: "outputs/mlp_run"
  num_train_epochs: 3
  learning_rate: 1e-4
  per_device_train_batch_size: 32
```

Then use it in your training script:

```python
from oumi.core.configs import TrainingConfig
from oumi.train import train

# Load and run training
config = TrainingConfig.from_yaml("train_config.yaml")
train(config)
```

The key points about using models in Oumi:
- Models are instantiated through the {py:func}`oumi.builders.build_model` function
- All constructor parameters go in `model_kwargs`
- Models can be configured through YAML for training
- The training system handles device placement, distributed training, etc.

### 3. Standard PyTorch Training Loop

The model can be used in a standard PyTorch training loop:

```python
# Initialize model and optimizer
model = MLPEncoder()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
```

## See Also

- {py:class}`oumi.core.models.BaseModel` - Base class for all Oumi models
- {py:class}`oumi.core.registry.Registry` - Model registration system
- {py:class}`oumi.core.configs.params.model_params.ModelParams` - Base parameters class for models
- {gh}`➿ Training CNN on Custom Dataset <notebooks/Oumi - Training CNN on Custom Dataset.ipynb>` - Sample Jupyter notebook using {py:class}`oumi.models.CNNClassifier` and [Custom Numpy Dataset](sample-custom-numpy-dataset).
