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

"""Oumi exception hierarchy.

This module is intentionally free of heavy dependencies (torch, transformers, etc.)
so that it can be imported cheaply in lightweight entry-points such as the CLI.

Exception Hierarchy:
    OumiError (base)
    ├── OumiConfigError (alias: ConfigurationError) - Invalid configuration settings
    │   ├── OumiConfigTypeError - Wrong config class loaded
    │   └── OumiConfigParsingError - OmegaConf parsing wrapper
    ├── ModelError - Model loading or initialization errors
    │   └── ModelNotFoundError - Model could not be found
    ├── DatasetError - Dataset loading or processing errors
    ├── InferenceError - Errors during inference
    ├── TrainingError - Errors during training
    └── HardwareException - Hardware-related errors
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf.errors import OmegaConfBaseException


class OumiError(Exception):
    """Base exception for all Oumi errors.

    All Oumi-specific exceptions inherit from this class, making it easy
    to catch all Oumi-related errors.

    Args:
        message: The error message.
        fix: Optional suggestion for how to fix the error.
    """

    def __init__(self, message: str, *, fix: str | None = None):
        self.message = message
        self.fix = fix
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with optional fix suggestion."""
        if self.fix:
            return f"{self.message}\n\nHow to fix: {self.fix}"
        return self.message


class OumiConfigError(OumiError):
    """Raised for invalid or inconsistent configuration (paths, values, structure)."""


# Alias for the structured exception hierarchy. ConfigurationError and
# OumiConfigError refer to the same class so that existing call sites and
# isinstance checks remain consistent.
ConfigurationError = OumiConfigError


class OumiConfigTypeError(OumiConfigError):
    """Raised when a loaded config is not an instance of the expected class."""

    def __init__(self, config_type: type, config_value: Any):
        """Record the expected config class and the actual loaded value."""
        self.config_type = config_type
        self.config_value = config_value
        super().__init__(
            f"Expected config of type {config_type.__name__}, "
            f"got {type(config_value).__name__}"
        )


class OumiConfigParsingError(OumiConfigError):
    """Wraps an OmegaConf exception into a user-friendly config error.

    The original exception is preserved via ``__cause__`` (``raise ... from e``);
    the CLI displays only this wrapper's message, keeping the OmegaConf traceback
    out of the user-facing output.
    """

    def __init__(self, cause: "OmegaConfBaseException"):
        """Build a user-facing message from an OmegaConf exception's key and msg."""
        key = getattr(cause, "full_key", None) or getattr(cause, "key", None)
        self.config_key: str | None = str(key) if key is not None else None
        msg = getattr(cause, "msg", None) or str(cause)
        if self.config_key:
            super().__init__(f"Config error at '{self.config_key}': {msg}")
        else:
            super().__init__(f"Config error: {msg}")


class ModelError(OumiError):
    """Exception raised for model-related errors.

    This exception is raised when there are issues loading, initializing,
    or using a model.
    """


class ModelNotFoundError(ModelError):
    """Exception raised when a model cannot be found.

    This exception is raised when a specified model cannot be located,
    either locally or on a remote hub.
    """


class DatasetError(OumiError):
    """Exception raised for dataset-related errors.

    This exception is raised when there are issues loading, processing,
    or validating datasets.
    """


class InferenceError(OumiError):
    """Exception raised for errors during inference.

    This exception is raised when there are issues during model inference,
    such as invalid inputs or engine failures.
    """


class TrainingError(OumiError):
    """Exception raised for errors during training.

    This exception is raised when there are issues during the training
    process, such as gradient issues or checkpoint failures.
    """


class HardwareException(OumiError):
    """Exception raised for hardware-related errors.

    This exception is raised when there are issues with hardware
    configuration or availability, such as missing GPUs or
    incompatible CUDA versions.
    """
