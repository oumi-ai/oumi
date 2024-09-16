"""Lema (Learning Machines) library.

This library provides tools and utilities for training, evaluating, and
inferring with machine learning models, particularly focused on language tasks.

Modules:
    - :mod:`~oumi.models`: Contains model architectures and related utilities.
    - :mod:`~oumi.evaluate` : Functions for evaluating models using LeMa and LM Harness.
    - :mod:`~oumi.evaluate_async`: Asynchronous evaluation functionality.
    - :mod:`~oumi.infer`: Functions for model inference, including interactive mode.
    - :mod:`~oumi.train`: Training utilities for machine learning models.
    - :mod:`~oumi.utils`: Utility functions, including logging configuration.

Functions:
    - :func:`~oumi.train.train`: Train a machine learning model.
    - :func:`~oumi.evaluate_async.evaluate_async`: Asynchronously evaluate a model.
    - :func:`~oumi.evaluate.evaluate_lema`: Evaluate a model using LeMa benchmarks.
    - :func:`~oumi.evaluate.evaluate_lm_harness`: Evaluate a model using Language
        Model Harness.
    - :func:`~oumi.infer.infer`: Perform inference with a trained model.
    - :func:`~oumi.infer.infer_interactive`: Run interactive inference with a model.

Examples:
    Training a model::

        from lema import train
        from oumi.core.configs import TrainingConfig

        config = TrainingConfig(...)
        train(config)

    Evaluating a model::

        from lema import evaluate_lema
        from oumi.core.configs import EvaluationConfig

        config = EvaluationConfig(...)
        results = evaluate_lema(config)

    Performing inference::

        from lema import infer
        from oumi.core.configs import InferenceConfig

        config = InferenceConfig(...)
        outputs = infer(config)

See Also:
    - :mod:`oumi.core.configs`: For configuration classes used in LeMa
"""

from oumi import models
from oumi.evaluate import evaluate_lema, evaluate_lm_harness
from oumi.evaluate_async import evaluate_async
from oumi.infer import infer, infer_interactive
from oumi.train import train
from oumi.utils import logging

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
