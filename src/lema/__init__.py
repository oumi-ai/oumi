"""Lema (Learning Machines) library.

This library provides tools and utilities for training, evaluating, and
inferring with machine learning models, particularly focused on language tasks.

Modules:
    - :mod:`~lema.models`: Contains model architectures and related utilities.
    - :mod:`~lema.evaluate` : Functions for evaluating models using LeMa and LM Harness.
    - :mod:`~lema.evaluate_async`: Asynchronous evaluation functionality.
    - :mod:`~lema.infer`: Functions for model inference, including interactive mode.
    - :mod:`~lema.train`: Training utilities for machine learning models.
    - :mod:`~lema.utils`: Utility functions, including logging configuration.

Functions:
    - :func:`~lema.train.train`: Train a machine learning model.
    - :func:`~lema.evaluate_async.evaluate_async`: Asynchronously evaluate a model.
    - :func:`~lema.evaluate.evaluate_lema`: Evaluate a model using LeMa benchmarks.
    - :func:`~lema.evaluate.evaluate_lm_harness`: Evaluate a model using Language
        Model Harness.
    - :func:`~lema.infer.infer`: Perform inference with a trained model.
    - :func:`~lema.infer.infer_interactive`: Run interactive inference with a model.

Examples:
    Training a model::

        from lema import train
        from lema.core.configs import TrainingConfig

        config = TrainingConfig(...)
        train(config)

    Evaluating a model::

        from lema import evaluate_lema
        from lema.core.configs import EvaluationConfig

        config = EvaluationConfig(...)
        results = evaluate_lema(config)

    Performing inference::

        from lema import infer
        from lema.core.configs import InferenceConfig

        config = InferenceConfig(...)
        outputs = infer(config)

See Also:
    - :mod:`lema.core.configs`: For configuration classes used in LeMa
"""

from lema import models
from lema.evaluate import evaluate_lema, evaluate_lm_harness
from lema.evaluate_async import evaluate_async
from lema.infer import infer, infer_interactive
from lema.train import train
from lema.utils import logging

logging.configure_dependency_warnings()


__all__ = [
    "train",
    "evaluate_async",
    "evaluate_lema",
    "evaluate_lm_harness",
    "infer",
    "infer_interactive",
    "models",
]
