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
    :class:`HardwareException`: Exception raised for hardware-related errors.

Example:
    >>> from oumi.core.types import HardwareException
    >>> try:
    ...     # Some hardware-related operation
    ...     pass
    ... except HardwareException as e:
    ...     print(f"Hardware error occurred: {e}")

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
from oumi.core.types.exceptions import HardwareException

__all__ = [
    "HardwareException",
    "ContentItem",
    "ContentItemCounts",
    "Conversation",
    "Message",
    "Role",
    "Type",
    "TemplatedMessage",
]
