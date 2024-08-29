"""Types module for the LeMa (Learning Machines) library.

This module provides custom types and exceptions used throughout the LeMa framework.

Exceptions:
    :class:`HardwareException`: Exception raised for hardware-related errors.

Example:
    >>> from lema.core.types import HardwareException
    >>> try:
    ...     # Some hardware-related operation
    ...     pass
    ... except HardwareException as e:
    ...     print(f"Hardware error occurred: {e}")

Note:
    This module is part of the core LeMa framework and is used across various
    components to ensure consistent error handling and type definitions.
"""

from lema.core.types.exceptions import HardwareException

__all__ = [
    "HardwareException",
]
