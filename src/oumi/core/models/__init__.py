"""Core models module for the LeMa (Learning Machines) library.

This module provides base classes for different types of models used in the
LeMa framework.


See Also:
    - :mod:`oumi.models`: Module containing specific model implementations.
    - :class:`oumi.models.mlp.MLPEncoder`: An example of a concrete model
        implementation.

Example:
    To create a custom model, inherit from :class:`BaseModel`:

    >>> from oumi.core.models import BaseModel
    >>> class CustomModel(BaseModel):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...
    ...     def forward(self, x):
    ...         # Implement the forward pass
    ...         pass
"""

from oumi.core.models.base_model import BaseModel

__all__ = [
    "BaseModel",
]
