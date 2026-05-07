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
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf.errors import OmegaConfBaseException


class OumiConfigError(Exception):
    """Raised for invalid or inconsistent configuration (paths, values, structure)."""


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


class HardwareException(Exception):
    """An exception thrown for invalid hardware configurations."""
