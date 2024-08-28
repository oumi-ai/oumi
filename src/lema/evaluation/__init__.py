"""Evaluation module for the LeMa (Learning Machines) library.

This module provides various evaluation metrics and utility functions for assessing
the performance of machine learning models in the LeMa framework.

Example:
    >>> from lema.evaluation import compute_multiple_choice_accuracy
    >>> accuracy = compute_multiple_choice_accuracy(predictions, labels)
    >>> print(f"Multiple choice accuracy: {accuracy}")

Note:
    This module is part of the LeMa framework and is designed to work seamlessly
    with other components of the library for comprehensive model evaluation.
"""

from lema.evaluation.compute_metrics import compute_multiple_choice_accuracy
from lema.evaluation.infer_prob import infer_prob, most_probable_tokens

__all__ = [
    "infer_prob",
    "most_probable_tokens",
    "compute_multiple_choice_accuracy",
]
