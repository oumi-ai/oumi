import torch
import torch.nn as nn
from torch.nn import functional as F

from lema.core import registry
from lema.core.types.base_model import BaseModel


@registry.register("MlpEncoder", registry.RegistryType.MODEL_CONFIG)
class MLPEncoder(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize the MLPEncoder.

        Args:
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            output_dim (int): The output dimension.
        """
        super(MLPEncoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids: torch.LongTensor, labels=None):
        """Forward pass of the MLP model.

        Args:
            input_ids (torch.LongTensor): The input tensor of shape
                (batch_size, sequence_length).
            labels (torch.LongTensor, optional): The target labels tensor
                of shape (batch_size,).

        Returns:
            dict: A dictionary containing the model outputs.
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
    def criterion(self):
        """Returns the criterion function for the MLP model.

        The criterion function is used to compute the loss during training.

        Returns:
            torch.nn.CrossEntropyLoss: The cross-entropy loss function.
        """
        return F.cross_entropy
