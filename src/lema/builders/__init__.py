"""Builders module for the LeMa (Learning Machines) library.

This module provides builder functions to construct and configure
different components of the LeMa framework, including datasets, models,
optimizers, and trainers.

The builder functions encapsulate the complexity of creating these components,
allowing for easier setup and configuration of machine learning experiments.
"""

from lema.builders.callbacks import build_training_callbacks
from lema.builders.data import build_dataset
from lema.builders.metrics import build_metrics_function
from lema.builders.models import build_model, build_peft_model, build_tokenizer
from lema.builders.optimizers import build_optimizer
from lema.builders.training import build_trainer

__all__ = [
    "build_dataset",
    "build_metrics_function",
    "build_model",
    "build_optimizer",
    "build_peft_model",
    "build_tokenizer",
    "build_trainer",
    "build_training_callbacks",
]
