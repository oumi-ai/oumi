from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def forward(
        self, model_inputs: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            model_inputs (torch.Tensor): The input tensor to the model.
            labels (Optional[torch.Tensor]): The labels tensor (optional).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the
                output tensors of the model.
        """
        raise NotImplementedError

    @property
    def criterion(self) -> Callable:
        """Returns the criterion function used for model training.

        This method should be implemented by subclasses.

        Returns:
            A callable object representing the criterion function.
        """
        raise NotImplementedError
