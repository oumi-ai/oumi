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

"""Types module for the Oumi (Open Universal Machine Intelligence) library.

This module provides custom types and exceptions used throughout the Oumi framework.

Exceptions:
    :class:`OumiError`: Base class for user-facing errors (clean display).
    :class:`ConfigurationError`: Configuration-related errors.
    :class:`MissingParameterError`: Required parameter not provided.
    :class:`InvalidParameterValueError`: Parameter has invalid value.
    :class:`MissingDependencyError`: Required dependency not installed.
    :class:`DatasetError`: Dataset-related errors.
    :class:`DatasetNotFoundError`: Dataset file/resource not found.
    :class:`DatasetFormatError`: Data in unexpected format.
    :class:`RemoteServiceError`: External service communication errors.
    :class:`APIResponseError`: API returned an error.
    :class:`RegistryLookupError`: Registry item not found.
    :class:`HardwareException`: Hardware-related errors.
    :class:`ConfigNotFoundError`: Config file not found.

Note:
    This module is part of the core Oumi framework and is used across various
    components to ensure consistent error handling and type definitions.
"""

from oumi.core.types.conversation import (
    ContentItem,
    ContentItemCounts,
    Conversation,
    Message,
    Role,
    TemplatedMessage,
    Type,
)
from oumi.core.types.exceptions import (
    APIResponseError,
    ConfigNotFoundError,
    ConfigurationError,
    DatasetError,
    DatasetFormatError,
    DatasetNotFoundError,
    HardwareException,
    InvalidParameterValueError,
    MissingDependencyError,
    MissingParameterError,
    OumiError,
    RegistryLookupError,
    RemoteServiceError,
)

__all__ = [
    # Exceptions
    "OumiError",
    "ConfigurationError",
    "MissingParameterError",
    "InvalidParameterValueError",
    "MissingDependencyError",
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetFormatError",
    "RemoteServiceError",
    "APIResponseError",
    "RegistryLookupError",
    "HardwareException",
    "ConfigNotFoundError",
    # Conversation types
    "ContentItem",
    "ContentItemCounts",
    "Conversation",
    "Message",
    "Role",
    "Type",
    "TemplatedMessage",
]
