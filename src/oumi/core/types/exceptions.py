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

"""Domain-specific exceptions for the Oumi library.

This module provides a hierarchy of exceptions that allow for more precise
error handling and better error messages throughout the Oumi framework.

Exception Hierarchy:
    OumiError (base)
    ├── ConfigurationError - Invalid configuration settings
    ├── ModelError - Model loading or initialization errors
    │   └── ModelNotFoundError - Model could not be found
    ├── DatasetError - Dataset loading or processing errors
    ├── InferenceError - Errors during inference
    ├── TrainingError - Errors during training
    └── HardwareException - Hardware-related errors

Example:
    >>> from oumi.core.types.exceptions import ConfigurationError
    >>> raise ConfigurationError(
    ...     "model_name is required",
    ...     fix="Set model.model_name to a HuggingFace model ID"
    ... )
"""


class OumiError(Exception):
    """Base exception for all Oumi errors.

    This is the root of the Oumi exception hierarchy. All Oumi-specific
    exceptions inherit from this class, making it easy to catch all
    Oumi-related errors.

    Args:
        message: The error message.
        fix: Optional suggestion for how to fix the error.

    Example:
        >>> try:
        ...     # Some Oumi operation
        ...     pass
        ... except OumiError as e:
        ...     print(f"Oumi error: {e}")
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


class ConfigurationError(OumiError):
    """Exception raised for invalid configuration settings.

    This exception is raised when configuration parameters are invalid,
    missing, or inconsistent.

    Example:
        >>> raise ConfigurationError(
        ...     "output_dir is not specified",
        ...     fix="Set training.output_dir in your config file"
        ... )
    """

    pass


class ModelError(OumiError):
    """Exception raised for model-related errors.

    This exception is raised when there are issues loading, initializing,
    or using a model.
    """

    pass


class ModelNotFoundError(ModelError):
    """Exception raised when a model cannot be found.

    This exception is raised when a specified model cannot be located,
    either locally or on a remote hub.

    Example:
        >>> raise ModelNotFoundError(
        ...     "Model 'nonexistent-model' not found",
        ...     fix="Check the model name or provide a valid HuggingFace model ID"
        ... )
    """

    pass


class DatasetError(OumiError):
    """Exception raised for dataset-related errors.

    This exception is raised when there are issues loading, processing,
    or validating datasets.
    """

    pass


class InferenceError(OumiError):
    """Exception raised for errors during inference.

    This exception is raised when there are issues during model inference,
    such as invalid inputs or engine failures.
    """

    pass


class TrainingError(OumiError):
    """Exception raised for errors during training.

    This exception is raised when there are issues during the training
    process, such as gradient issues or checkpoint failures.
    """

    pass


class HardwareException(OumiError):
    """Exception raised for hardware-related errors.

    This exception is raised when there are issues with hardware
    configuration or availability, such as missing GPUs or
    incompatible CUDA versions.

    Example:
        >>> raise HardwareException(
        ...     "Flash attention 2 is not supported on this hardware",
        ...     fix="Install flash-attn or use a different attention implementation"
        ... )
    """

    def __init__(self, message: str, *, fix: str | None = None):
        # For backward compatibility, HardwareException can be raised
        # with just a message string
        super().__init__(message, fix=fix)
