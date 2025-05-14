# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The CNNClassifier model provides a basic example how to use ConvNets in Oumi."""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from oumi.core import registry
from oumi.core.models.base_model import BaseModel


@registry.register("CnnClassifier", registry.RegistryType.MODEL)
class CNNClassifier(BaseModel):
    """A simple ConvNet for classification of small fixed-size images."""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        *,
        in_channels: int = 3,
        output_dim: int = 10,
        kernel_size: int = 5,
    ):
        """Initialize the ConvNet for image classification.

        Args:
            image_width: Width of input images in pixels.
            image_height: Height of input images in pixels.
            in_channels: The number of input channels e.g., 3 for RGB, 1 for greyscale.
            output_dim: The output dimension i.e., the number of classes.
            kernel_size: Convolutional kernel size.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=kernel_size)
        w, h = self._compute_next_level_image_size(
            image_width, image_height, kernel_size=kernel_size, halve=False
        )
        self.conv2 = nn.Conv2d(32, 32, kernel_size=kernel_size)
        w, h = self._compute_next_level_image_size(
            w, h, kernel_size=kernel_size, halve=True
        )
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size)
        w, h = self._compute_next_level_image_size(
            w, h, kernel_size=kernel_size, halve=True
        )
        self._final_image_width = w
        self._final_image_height = h
        self.fc1 = nn.Linear(
            self._final_image_width * self._final_image_height * 64, 256
        )
        self.fc2 = nn.Linear(256, output_dim)

    @staticmethod
    def _compute_next_level_image_size(
        w: int, h: int, kernel_size: int, halve: bool
    ) -> tuple[int, int]:
        w, h = (w - (kernel_size - 1)), (h - (kernel_size - 1))
        if halve:
            w, h = (w // 2), (h // 2)
        if w <= 0 or h <= 0:
            raise ValueError(f"Image is too small for kernel_size={kernel_size}")
        return (w, h)

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model."""
        # Whether to apply dropout. `False` corresponds to inference mode.
        training_mode = labels is not None

        x = F.relu(self.conv1(images))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=training_mode)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=training_mode)
        x = x.view(-1, self._final_image_width * self._final_image_height * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=training_mode)
        logits = self.fc2(x)
        outputs = {"logits": logits}
        if training_mode:
            targets = F.log_softmax(logits, dim=1)
            loss = self.criterion(targets, labels)
            outputs["loss"] = loss
        return outputs

    @property
    def criterion(self) -> Callable:
        """Returns the criterion function to compute loss."""
        return F.nll_loss
