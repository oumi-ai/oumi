"""Core trainers module for the Oumi (Open Universal Machine Intelligence) library.

This module provides various trainer implementations for use in the Oumi framework.
These trainers are designed to facilitate the training process for different
types of models and tasks.

Example:
    >>> from oumi.core.trainers import Trainer
    >>> trainer = Trainer(model=my_model, dataset=my_dataset) # doctest: +SKIP
    >>> trainer.train() # doctest: +SKIP

Note:
    For detailed information on each trainer, please refer to their respective
        class documentation.
"""

from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.core.trainers.hf_trainer import HuggingFaceTrainer
from oumi.core.trainers.oumi_trainer import Trainer

__all__ = [
    "BaseTrainer",
    "HuggingFaceTrainer",
    "Trainer",
]
