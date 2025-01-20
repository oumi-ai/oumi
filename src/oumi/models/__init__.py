"""Models module for the Oumi (Open Universal Machine Intelligence) library.

This module provides various model implementations for use in the Oumi framework.
These models are designed for different machine learning tasks and can be used
with the datasets and training pipelines provided by Oumi.

Available models:
    - :py:class:`~oumi.models.mlp.MLPEncoder`: A Multi-Layer Perceptron (MLP)
        encoder model.
    - :py:class:`~oumi.models.cnn_classifier.CNNClassifier`: A simple ConvNet for
        image classification e.g., can be used for MNIST digits classification.

Each model is implemented as a separate class, inheriting from appropriate base classes
in the Oumi framework.

Example:
    >>> from oumi.models import MLPEncoder
    >>> encoder = MLPEncoder(input_dim=784, hidden_dim=256, output_dim=10)
    >>> output = encoder(input_data) # doctest: +SKIP

Note:
    For detailed information on each model, please refer to their respective
    class documentation.
"""

from oumi.models.cnn_classifier import CNNClassifier
from oumi.models.mlp import MLPEncoder

__all__ = ["MLPEncoder", "CNNClassifier"]
