## Model Definition

At a high level, this is the data flow in a Pytorch training loop:
Dataset[idx] -> DataLoader.collate_fn -> Model.forward -> Criterion(model_out, labels) -> loss.backward() -> Optimizer.step()

1. Dataset: A dataset object that contains the data.
    - Can either be a MapDataset or an IterableDataset
    - Only responsible for the business logic of loading, preprocessing the data
2. DataLoader: A dataloader object that loads the data in batches
    - Collates the data into batches
    - Responsible for distributed training support, shuffling, performance, async, resuming, etc.
3. Model: A model object that takes the data and returns the output
    - Should be able to directly consume the outputs from the dataloader, with no intermediate steps
4. Loss: A loss object that takes the output and the labels and returns the loss
    - Should be able to directly consume the outputs from the model AND the dataloader, with no intermediate steps
5. Optimizer: An optimizer object that takes the loss and updates the model weights
6. Training loop: A loop that takes the data, model, loss, optimizer, and trains the model


```python
# [LeMa] We will use Pytorch as the base library for our models
# [LeMa] We will aim to keep maximal compatibility with Native Pytorch
# [LeMa] We will aim to keep moderate compatibility with Huggingface Transformers
class SimpleModel(nn.Module):  # [PT] Needs to inherit from nn.Module
    def __init__(self, *args, **kwargs):
        # [PT] *args and **kwargs contain all the argments needed to build the model scaffold
        # [PT] weights should not be loaded, or moved to devices at this point
        super(LeMaModel, self).__init__()

        # [LeMa]: Keep free-form args and kwargs at this time.
        # [LeMa] Downstream (more opinionated models) can use structured config file that inherits from a dict.

        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(16*5*5, 120)

    def forward(self, model_inputs, labels: Optional[torch.Tensor]=None, **kwargs) -> Dict[str, torch.Tensor]:
        # [PT] forward function is required for all Pytorch models
        # [PT] It needs to be able to consume all the outputs from the dataloader
        # [PT] It needs to be able to consume either a batch or individual samples.
        # [Lema] For simplicity we exclusively use batched inputs

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
        # [LeMa Decison #4]: Keep free-form args and kwargs at this time.
        # Downstream (more opinionated models) can use structured config file that inherits from a dict.
        return {"outputs": outputs, "loss": loss}

    @attribute
    def criterion(self):
        # [LeMa] Keep loss function as an attribute
        return nn.CrossEntropyLoss()

    #
    # Everything else is optional, convenience stuff!
    # E.g. from_pretrained, save_pretrained, etc.
    #
```
