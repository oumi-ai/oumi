"""Based on MFU from PaLM paper: https://arxiv.org/pdf/2204.02311."""

from typing import Optional, Union

import transformers
import numpy as np
import torch

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams
from oumi.utils.logging import logger

_MODEL_KWARG = "model"


class AutoClipGradNormCallback(BaseTrainerCallback):
    """Trainer callback to calculate the MFU of the model during training.

    Should be compatible with all trainers that inherit from transformers.Trainer.
    """

    def __init__(self, clip_percentile: int = 10):
        """Initialize the MfuTrainerCallback.

        Args:
            metric_name: Name of the metric to aggregate
        """
        self._grad_norm_history = []
        self._clip_percentile = clip_percentile
    
    def _get_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm 

    def on_pre_optimizer_step(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        model = kwargs[_MODEL_KWARG]

        grad_norm = self._get_grad_norm(model)
        self._grad_norm_history.append(grad_norm)
        clip_value = np.percentile(self._grad_norm_history, self._clip_percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        
