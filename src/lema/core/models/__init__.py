"""Core models module for the LeMa (Learning Machines) library.

This module provides base classes for different types of models used in the
LeMa framework.

Classes:
    BaseModel (:class:`lema.core.models.base_model.BaseModel`):
        Base class for all models in the LeMa framework.
        This class defines the common interface and functionality that all
        models in the LeMa framework should implement.

See Also:
    - :mod:`lema.models`: Module containing specific model implementations.
    - :class:`lema.models.mlp.MLPEncoder`: An example of a concrete model
    implementation.

Example:
    To create a custom model, inherit from :class:`BaseModel`:

    >>> from lema.core.models import BaseModel
    >>> class CustomModel(BaseModel):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...
    ...     def forward(self, x):
    ...         # Implement the forward pass
    ...         pass
"""

from lema.core.models.base_model import BaseModel

__all__ = [
    "BaseModel",
]
