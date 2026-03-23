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


class LoraTargetModulesError(ValueError):
    """Raised when LoRA target modules don't exist in the model."""

    def __init__(
        self,
        requested_modules: list[str],
        available_modules: list[str],
    ):
        """Initialize the error with requested and available modules.

        Args:
            requested_modules: The modules specified in the config.
            available_modules: The modules that exist in the model.
        """
        self.requested_modules = requested_modules
        self.available_modules = available_modules

        # Find common LoRA-compatible module patterns
        linear_modules = [
            m
            for m in available_modules
            if any(
                pattern in m.lower()
                for pattern in [
                    "proj",
                    "attn",
                    "mlp",
                    "fc",
                    "linear",
                    "dense",
                    "query",
                    "key",
                    "value",
                ]
            )
        ]

        message = (
            f"LoRA target modules not found in model.\n\n"
            f"You specified:  {requested_modules}\n"
            f"Model contains: {linear_modules[:10]}"
        )
        if len(linear_modules) > 10:
            message += f" ... and {len(linear_modules) - 10} more"

        message += (
            "\n\nSuggested fixes:\n"
            "  1. Update target_modules in your config to use modules from above\n"
            '  2. Use target_modules: ["all-linear"] to automatically target all linear layers'
        )

        super().__init__(message)
