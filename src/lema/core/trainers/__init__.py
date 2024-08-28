"""Core trainers module for the LeMa (Learning Machines) library.

This module provides various trainer implementations for use in the LeMa framework.
These trainers are designed to facilitate the training process for different
types of models and tasks.

Example:
    >>> from lema.core.trainers import Trainer
    >>> trainer = Trainer(model=my_model, dataset=my_dataset)
    >>> trainer.train()

Note:
    For detailed information on each trainer, please refer to their respective
        class documentation.
"""

from lema.core.trainers.base_trainer import BaseTrainer
from lema.core.trainers.hf_trainer import HuggingFaceTrainer
from lema.core.trainers.lema_trainer import Trainer

__all__ = [
    "BaseTrainer",
    "HuggingFaceTrainer",
    "Trainer",
]
