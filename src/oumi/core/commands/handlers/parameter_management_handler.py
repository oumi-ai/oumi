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

"""Parameter management command handler."""

from typing import Any, Optional

from oumi.core.commands.base_handler import BaseCommandHandler, CommandResult
from oumi.core.commands.command_parser import ParsedCommand


class ParameterManagementHandler(BaseCommandHandler):
    """Handles parameter-related commands: set."""

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return ["set"]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a parameter management command."""
        if command.command == "set":
            return self._handle_set(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unsupported command: {command.command}",
                should_continue=False,
            )

    def _handle_set(self, command: ParsedCommand) -> CommandResult:
        """Handle the /set(param=value) command to adjust generation parameters."""
        if not command.kwargs and not command.args:
            return CommandResult(
                success=False,
                message="set command requires parameter=value arguments (e.g., /set(temperature=0.8, top_p=0.9))",
                should_continue=False,
            )

        # Track successful parameter changes
        changes_made = []
        errors = []

        # Process kwargs (param=value format)
        for param, value in command.kwargs.items():
            success, error_msg = self._update_parameter(param, value)
            if success:
                changes_made.append(f"{param}={value}")
            else:
                errors.append(f"{param}: {error_msg}")

        # Process positional args that might contain =
        for arg in command.args:
            if "=" in arg:
                try:
                    param, value = arg.split("=", 1)
                    param = param.strip()
                    value = value.strip()

                    success, error_msg = self._update_parameter(param, value)
                    if success:
                        changes_made.append(f"{param}={value}")
                    else:
                        errors.append(f"{param}: {error_msg}")
                except ValueError:
                    errors.append(f"Invalid parameter format: {arg}")

        # Build response message
        messages = []
        if changes_made:
            messages.append(f"Updated parameters: {', '.join(changes_made)}")
        if errors:
            messages.append(f"Errors: {'; '.join(errors)}")

        if changes_made:
            return CommandResult(
                success=True,
                message=" | ".join(messages) if messages else "Parameters updated",
                should_continue=False,
            )
        else:
            return CommandResult(
                success=False,
                message=" | ".join(messages)
                if messages
                else "No parameters were updated",
                should_continue=False,
            )

    def _update_parameter(self, param: str, value: str) -> tuple[bool, str]:
        """Update a single generation parameter.

        Args:
            param: Parameter name.
            value: Parameter value as string.

        Returns:
            Tuple of (success, error_message).
        """
        param = param.lower().strip()

        # Validate parameter name
        valid_params = {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "max_new_tokens",
            "sampling",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "min_p",
            "num_beams",
        }

        if param not in valid_params:
            return (
                False,
                f"Unknown parameter '{param}'. Valid parameters: {', '.join(sorted(valid_params))}",
            )

        # Parse and validate value
        try:
            parsed_value = self._parse_parameter_value(param, value)
        except ValueError as e:
            return False, str(e)

        # Validate value range/constraints
        validation_error = self._validate_parameter_value(param, parsed_value)
        if validation_error:
            return False, validation_error

        # Update the parameter in generation config
        try:
            if hasattr(self.config, "generation") and self.config.generation:
                setattr(self.config.generation, param, parsed_value)
                return True, ""
            else:
                return False, "Generation config not available"
        except Exception as e:
            return False, f"Failed to update parameter: {str(e)}"

    def _parse_parameter_value(self, param: str, value: str) -> Any:
        """Parse parameter value from string.

        Args:
            param: Parameter name.
            value: String value to parse.

        Returns:
            Parsed value.

        Raises:
            ValueError: If value cannot be parsed.
        """
        value = value.strip()

        # Boolean parameters
        if param in ["sampling"]:
            if value.lower() in ["true", "1", "yes", "on"]:
                return True
            elif value.lower() in ["false", "0", "no", "off"]:
                return False
            else:
                raise ValueError(f"Boolean value expected for {param}, got '{value}'")

        # Integer parameters
        elif param in ["max_tokens", "max_new_tokens", "top_k", "seed", "num_beams"]:
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Integer value expected for {param}, got '{value}'")

        # Float parameters
        elif param in [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "min_p",
        ]:
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Numeric value expected for {param}, got '{value}'")

        else:
            # Default to string
            return value

    def _validate_parameter_value(self, param: str, value: Any) -> Optional[str]:
        """Validate parameter value constraints.

        Args:
            param: Parameter name.
            value: Parsed parameter value.

        Returns:
            Error message if invalid, None if valid.
        """
        if param == "temperature":
            if not (0.0 <= value <= 2.0):
                return "Temperature must be between 0.0 and 2.0"

        elif param == "top_p":
            if not (0.0 <= value <= 1.0):
                return "top_p must be between 0.0 and 1.0"

        elif param == "top_k":
            if value < 1:
                return "top_k must be at least 1"

        elif param in ["max_tokens", "max_new_tokens"]:
            if value < 1:
                return f"{param} must be at least 1"
            elif value > 100000:  # Reasonable upper bound
                return f"{param} must be less than 100,000"

        elif param == "frequency_penalty":
            if not (-2.0 <= value <= 2.0):
                return "frequency_penalty must be between -2.0 and 2.0"

        elif param == "presence_penalty":
            if not (-2.0 <= value <= 2.0):
                return "presence_penalty must be between -2.0 and 2.0"

        elif param == "min_p":
            if not (0.0 <= value <= 1.0):
                return "min_p must be between 0.0 and 1.0"

        elif param == "seed":
            if value < 0:
                return "seed must be non-negative"

        elif param == "num_beams":
            if value < 1:
                return "num_beams must be at least 1"

        return None  # Valid
