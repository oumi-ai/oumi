"""Builders module for the Oumi (Open Unified Machine Intelligence) library.

This module provides builder functions to construct and configure
different components of the Oumi framework, including datasets, models,
optimizers, and trainers.

The builder functions encapsulate the complexity of creating these components,
allowing for easier setup and configuration of machine learning experiments.
"""

from oumi.builders.callbacks import build_training_callbacks
from oumi.builders.data import (
    build_dataset,
    build_dataset_from_params,
    build_dataset_mixture,
)
from oumi.builders.metrics import build_metrics_function
from oumi.builders.models import (
    build_chat_template,
    build_model,
    build_peft_model,
    build_tokenizer,
)
from oumi.builders.optimizers import build_optimizer
from oumi.builders.training import build_trainer

__all__ = [
    "build_chat_template",
    "build_dataset_from_params",
    "build_dataset_mixture",
    "build_dataset",
    "build_metrics_function",
    "build_model",
    "build_optimizer",
    "build_peft_model",
    "build_tokenizer",
    "build_trainer",
    "build_training_callbacks",
]
