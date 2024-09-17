"""Types module for the OUMI (Open Unified Machine Intelligence) library.

This module provides custom types and exceptions used throughout the OUMI framework.

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
    This module is part of the core OUMI framework and is used across various
    components to ensure consistent error handling and type definitions.
"""

from oumi.core.types.exceptions import HardwareException

__all__ = [
    "HardwareException",
]
