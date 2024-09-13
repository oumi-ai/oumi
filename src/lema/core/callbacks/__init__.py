"""Trainer callbacks module for the LeMa (Learning Machines) library.

This module provides trainer callbacks, which can be used to customize
the behavior of the training loop in the LeMa Trainer
that can inspect the training loop state for progress reporting, logging,
early stopping, etc.
"""

from lema.core.callbacks.base_trainer_callback import BaseTrainerCallback

__all__ = [
    "BaseTrainerCallback",
]
