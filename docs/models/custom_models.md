# Custom Models

```{attention}
Section under construction. Contributions welcome!
```

At a high level, this is the pseudo code for a Pytorch training loop:

```python
# The Oumi config defines everything required to train a model.
# The config is immutable and serializable.
config = load_config()

# The first major component: OumiDataset
# An Oumi dataset object implements the pytorch dataset spec.
# A dataset can be either a map-style dataset (`torch.utils.data.Dataset`, or an iterable-style dataset (`torch.utils.data.IterableDataset`), or both.
# It can be either streamed, or fully loaded in memory.
dataset = OumiDataset(config.data)  # Pytorch Dataset object

sample = dataset[0] # Load and preprocess an individual training sample
# Each sample contains both the model inputs, and optionally any labels required to compute the loss and/or metrics.
# E.g. {"input_ids": ..., "attention_mask": ..., "labels": ...} or {"image": ..., "labels": ...}

# A dataset can be used with PyTorch-compatible dataloaders directly (`torch.utils.data.DataLoader`)
# Here we pass the arguments directly, but usually they are defined in config.data.dataloader_kwargs
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)  # Pytorch DataLoader object

# The data loader object handles batching, shuffling, sharding, pre-fetching, distributing data over multiple nodes, etc.
# The data loader returns a batch of samples, which can be directly consumed by the model
for batch in dataloader:
    optimizer.zero_grad()

    # In this case, we manually move the batch to the device.
    # With FSDP, the model wrapper will handle this during the forward function.
    batch = batch.to(device)

    # The model returns a dict-like object, which contains the model outputs, and optionally the loss. Internally calls model.forward
    outputs = model(**batch)

    # The loss is computed in the forward via the `model.criterion` attribute, and returned as part of the outputs. E.g. loss = model.criterion(outputs, labels)
    loss = outputs.loss

    ## Backward pass
    loss.backward()

    # Everything so-far is process-local, and only involves computations performed locally on the current GPU.
    # The optimizer is responsible for updating the model weights, and performs any necessary communication between nodes.
    optimizer.step()
```

Digging deeper into the model itself, here is a simple model definition. In this example, we define a simple model that consists of two convolutional layers, followed by a fully connected layer.

Each lines is annotated with the relevant information:

- [Oumi] means this is an Oumi-specific decision, which we can revisit as we see fit.
- [PT] means this is a Pytorch-specific requirement. We need a strong reason to deviate from this.
- [HF] means this is a Huggingface-specific requirement. We can revisit if the tradeoff is worth it.

```python
# [Oumi] We will use Pytorch as the base library for our models
# [Oumi] We will aim to keep maximal compatibility with Native Pytorch
# [Oumi] We will aim to keep moderate compatibility with Huggingface Transformers
class SimpleModel(nn.Module):  # [PT] Needs to inherit from nn.Module
    def __init__(self, *args, **kwargs):
        # [PT] *args and **kwargs contain all the argments needed to build the model scaffold
        # [PT] weights should not be loaded, or moved to devices at this point
        super(SimpleModel, self).__init__()

        # [Oumi]: Keep free-form args and kwargs at this time.
        # [Oumi] Downstream (more opinionated models) can use structured config file that inherits from a dict.

        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(16*5*5, 120)

    def forward(self, model_inputs, labels: Optional[torch.Tensor]=None, **kwargs) -> Dict[str, torch.Tensor]:
        # [PT] forward function is required for all Pytorch models
        # [PT] It needs to be able to consume all the outputs from the dataloader
        # [PT] It needs to be able to consume either a batch or individual samples.
        # [Oumi] For simplicity we exclusively use batched inputs

        # [HF] To be compatible with HF trainers,
        # [HF] labels are optional, and are only used during training
        # [HF] if labels are not None, the model is expected to return a loss
        hidden = F.relu(self.conv1(model_inputs))
        hidden = F.relu(self.conv2(hidden))
        outputs = F.relu(self.conv2(x))

        if labels is not None:
            loss = self.criterion(outputs, labels)
        else:
            loss = None

        # Can technically be a Tuple or a Dict
        # [Oumi]: Keep free-form args and kwargs at this time.
        # Downstream (more opinionated models) can use structured config file that inherits from a dict.
        return {"outputs": outputs, "loss": loss}

    @property
    def criterion(self):
        # [Oumi] Keep loss function as an attribute
        return nn.CrossEntropyLoss()

    #
    # Everything else is optional, convenience stuff!
    # E.g. from_pretrained, save_pretrained, etc.
    #
```

This same model can then be used by the `Huggingface Trainer` class, which is a high-level API that abstracts away the training loop, and provides a simple interface to train models. In the future, we can consider building our own Trainer class, or Megatron which follows similar conventions.

```python
from transformers import Trainer, TrainingArguments
model = SimpleModel()

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Instantiate the trainer
trainer = Trainer(
    model=model,                     # The instantiated model to be trained
    args=training_args,              # HF Training arguments
    train_dataset=dataset,           # Training dataset
    eval_dataset=dataset             # Evaluation dataset
)

# Train the model
trainer.train()
```
