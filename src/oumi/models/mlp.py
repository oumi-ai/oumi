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

"""This module defines the MLPEncoder class, which is a simple text encoder."""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from oumi.core import registry
from oumi.core.models.base_model import BaseModel


@registry.register("MlpEncoder", registry.RegistryType.MODEL)
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
    def criterion(self) -> Callable:
        """Returns the criterion function for the MLP model.

        The criterion function is used to compute the loss during training.

        Returns:
            torch.nn.CrossEntropyLoss: The cross-entropy loss function.
        """
        return F.cross_entropy
