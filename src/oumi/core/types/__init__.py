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
