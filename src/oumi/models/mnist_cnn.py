"""The SampleMnistCNN model provides a basic example how to use ConvNets in Oumi."""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from oumi.core import registry
from oumi.core.models.base_model import BaseModel


@registry.register("SampleMnistCNN", registry.RegistryType.MODEL)
class SampleMnistCNN(BaseModel):
    """Sample ConvNet for MNIST handwritten digits classification."""

    def __init__(
        self,
    ):
        """Initialize the ConvNet for MNIST digits classification."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model."""
        training = labels is not None

        x = F.relu(self.conv1(images))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=training)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=training)
        logits = self.fc2(x)
        outputs = {"logits": logits}
        if training:
            targets = F.log_softmax(logits, dim=1)
            loss = self.criterion(targets, labels)
            outputs["loss"] = loss
        return outputs

    @property
    def criterion(self) -> Callable:
        """Returns the criterion function to compute loss."""
        return F.nll_loss
