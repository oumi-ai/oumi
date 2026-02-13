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


class HardwareException(Exception):
    """An exception thrown for invalid hardware configurations."""


class ConfigNotFoundError(FileNotFoundError):
    """Raised when a config file is not found, with a helpful error message."""

    def __init__(self, config_path: str):
        """Initialize the error with the missing config path.

        Args:
            config_path: The path to the config file that was not found.
        """
        self.config_path = config_path

        message = (
            f"Config file not found: '{config_path}'\n\n"
            "Tip: Use 'oumi train --list' to see available configs, "
            "or check that the file path is correct."
        )

        super().__init__(message)
