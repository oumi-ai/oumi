"""Evaluation module for the OUMI (Open Unified Machine Intelligence) library.

This module provides various evaluation metrics and utility functions for assessing
the performance of machine learning models in the OUMI framework.

Example:
    >>> from oumi.evaluation import compute_multiple_choice_accuracy
    >>> accuracy = compute_multiple_choice_accuracy(predictions, labels)
    >>> print(f"Multiple choice accuracy: {accuracy}")

Note:
    This module is part of the OUMI framework and is designed to work seamlessly
    with other components of the library for comprehensive model evaluation.
"""

from oumi.evaluation.compute_metrics import compute_multiple_choice_accuracy
from oumi.evaluation.infer_prob import infer_prob, most_probable_tokens

__all__ = [
    "infer_prob",
    "most_probable_tokens",
    "compute_multiple_choice_accuracy",
]
