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

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self, **kwargs):
        """Initializes a new instance of the model class, and builds the model scaffold.

        Note:
            - All model layers should be registered in this method.
            - The weights should not be loaded or moved to devices at this point.

        Args:
            **kwargs: should contain all the parameters needed
                to build the model scaffold.
        """
        super().__init__()

    @abstractmethod
    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Performs the forward pass of the model.

        Optionally computes the loss if the necessary keyword arguments are provided.

        Args:
            **kwargs: should contain all the parameters needed to perform the forward
                pass, and compute the loss if needed.

        Returns:
            A dictionary containing the output tensors.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def criterion(self) -> Callable:
        """Returns the criterion function used for model training.

        Returns:
            A callable object representing the criterion function.
        """
        raise NotImplementedError
