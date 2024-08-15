"""Calls `profiler.step()`  at the end of each training step."""

from typing import Optional, Union

import transformers

from lema.core.types import TrainingParams


class ProfilerStepCallback(transformers.TrainerCallback):
    """Trainer callback to calculate the MFU of the model during training.

    Should be compatible with all trainers that inherit from transformers.Trainer.
    """

    def __init__(self, profiler):
        """Initialize the ProfilerStepCallback.

        Args:
            profiler: PyTorch profiler object.
        """
        self._profiler = profiler

    def on_step_end(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the end of each train step.

        Note that this will be called after all gradient accumulation substeps.
        """
        self._profiler.step()
