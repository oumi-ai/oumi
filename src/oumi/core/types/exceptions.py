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

"""Custom exceptions for the Oumi framework.

This module defines a hierarchy of exceptions for user-facing errors that should
be displayed cleanly without stack traces, as well as internal exceptions for
hardware and configuration issues.

Exception Hierarchy:
    OumiError (base for all user-facing errors)
    ├── ConfigurationError (invalid or missing configuration)
    │   ├── MissingParameterError (required parameter not provided)
    │   └── InvalidParameterValueError (parameter has invalid value)
    ├── MissingDependencyError (required package not installed)
    ├── DatasetError (dataset-related issues)
    │   ├── DatasetNotFoundError (dataset file/resource missing)
    │   └── DatasetFormatError (data in unexpected format)
    ├── RemoteServiceError (external service communication)
    │   └── APIResponseError (API returned an error)
    └── RegistryLookupError (registry item not found)

    HardwareException (hardware configuration issues)
    ConfigNotFoundError (config file not found - inherits FileNotFoundError)
"""


class OumiError(Exception):
    """Base class for user-facing errors that should display cleanly.

    Exceptions inheriting from OumiError are caught by the CLI and displayed
    with a clean error message instead of a full stack trace. Use this for
    errors that users can act on (configuration issues, missing files, etc.).
    """

    pass


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(OumiError):
    """Base class for configuration-related errors.

    Raised when there are issues with configuration values, whether from
    config files, environment variables, or programmatic configuration.
    """

    pass


class MissingParameterError(ConfigurationError):
    """A required parameter was not provided.

    Raised when a mandatory configuration parameter is missing or empty.

    Example:
        raise MissingParameterError(
            "model_name is required for training. "
            "Set it in your config file or via --model.model_name"
        )
    """

    pass


class InvalidParameterValueError(ConfigurationError):
    """A parameter has an invalid value.

    Raised when a configuration parameter is provided but has an invalid
    value (out of range, wrong type, incompatible with other settings, etc.).

    Example:
        raise InvalidParameterValueError(
            f"learning_rate must be positive, got {lr}. "
            "Valid range: 0 < learning_rate <= 1"
        )
    """

    pass


# =============================================================================
# Dependency Errors
# =============================================================================


class MissingDependencyError(OumiError):
    """A required dependency is not installed.

    Raised when an optional dependency is required for a specific feature
    but is not installed in the environment.

    Example:
        raise MissingDependencyError(
            "vLLM is required for VLLMInferenceEngine. "
            "Install it with: pip install oumi[vllm]"
        )
    """

    pass


# =============================================================================
# Dataset Errors
# =============================================================================


class DatasetError(OumiError):
    """Base class for dataset-related errors.

    Raised when there are issues loading, parsing, or processing datasets.
    """

    pass


class DatasetNotFoundError(DatasetError):
    """Dataset file or resource was not found.

    Raised when a specified dataset path doesn't exist or a remote
    dataset cannot be accessed.

    Example:
        raise DatasetNotFoundError(
            f"Dataset not found at '{path}'. "
            "Verify the path exists or check your Hugging Face credentials."
        )
    """

    pass


class DatasetFormatError(DatasetError):
    """Dataset is in an unexpected or invalid format.

    Raised when dataset contents don't match the expected schema or format.

    Example:
        raise DatasetFormatError(
            f"Expected 'messages' field in conversation data, got: {fields}. "
            "See docs for the expected conversation format."
        )
    """

    pass


# =============================================================================
# Remote Service Errors
# =============================================================================


class RemoteServiceError(OumiError):
    """Base class for remote service communication errors.

    Raised when there are issues communicating with external services
    like inference APIs, cloud providers, or job schedulers.
    """

    pass


class APIResponseError(RemoteServiceError):
    """An API returned an error response.

    Raised when an external API returns an error status or malformed response.

    Example:
        raise APIResponseError(
            f"OpenAI API error: {error_message}. "
            "Check your API key and request parameters."
        )
    """

    pass


# =============================================================================
# Registry Errors
# =============================================================================


class RegistryLookupError(OumiError):
    """An item was not found in a registry.

    Raised when looking up a dataset, model, or other registered item
    by name fails because the name is not recognized.

    Example:
        raise RegistryLookupError(
            f"Unknown dataset '{name}'. "
            "Use `oumi registry list datasets` to see available datasets."
        )
    """

    pass


# =============================================================================
# Legacy/Hardware Exceptions (kept for backwards compatibility)
# =============================================================================


class HardwareException(Exception):
    """An exception thrown for invalid hardware configurations."""

    pass


class ConfigNotFoundError(FileNotFoundError):
    """An exception thrown when a config file cannot be found."""

    pass
