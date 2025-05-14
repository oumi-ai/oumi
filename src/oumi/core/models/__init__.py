# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core models module for the Oumi (Open Universal Machine Intelligence) library.

This module provides base classes for different types of models used in the
Oumi framework.

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
