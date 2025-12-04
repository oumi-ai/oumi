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

"""Unit tests for parameter management command handlers."""

from unittest.mock import Mock

from oumi.core.configs import GenerationParams
from oumi_chat.commands import ParsedCommand
from oumi_chat.commands.command_context import CommandContext
from oumi_chat.commands.handlers.parameter_management_handler import (
    ParameterManagementHandler,
)
from tests.oumi_chat.utils.chat_test_utils import (
    create_test_inference_config,
    validate_command_result,
)


class TestSetCommand:
    """Test suite for /set() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Ensure generation config exists with default values
        self.test_config.generation = GenerationParams(
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=100,
            seed=42,
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        self.handler = ParameterManagementHandler(context=self.command_context)

    def test_set_single_float_parameter(self):
        """Test setting a single float parameter."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"temperature": "0.8"},
            raw_input="/set(temperature=0.8)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Updated parameters", "temperature=0.8"],
        )
        assert self.test_config.generation.temperature == 0.8

    def test_set_single_integer_parameter(self):
        """Test setting a single integer parameter."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"max_new_tokens": "200"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Updated parameters", "max_new_tokens=200"],
        )
        assert self.test_config.generation.max_new_tokens == 200

    def test_set_single_boolean_parameter(self):
        """Test setting a single boolean parameter."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"sampling": "true"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Updated parameters", "sampling=true"],
        )
        assert self.test_config.generation.use_sampling is True

    def test_set_multiple_parameters(self):
        """Test setting multiple parameters at once."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"temperature": "0.5", "top_p": "0.8", "max_new_tokens": "150"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Updated parameters", "temperature=0.5"],
        )
        assert self.test_config.generation.temperature == 0.5
        assert self.test_config.generation.top_p == 0.8
        assert self.test_config.generation.max_new_tokens == 150

    def test_set_parameters_via_positional_args(self):
        """Test setting parameters using positional arguments with = format."""
        command = ParsedCommand(
            command="set",
            args=["temperature=0.6", "min_p=0.1"],
            kwargs={},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Updated parameters"],
        )
        assert self.test_config.generation.temperature == 0.6
        assert self.test_config.generation.min_p == 0.1

    def test_set_mixed_kwargs_and_args(self):
        """Test setting parameters using both kwargs and positional args."""
        command = ParsedCommand(
            command="set",
            args=["temperature=0.3"],
            kwargs={"top_p": "0.7"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Updated parameters"],
        )
        assert self.test_config.generation.temperature == 0.3
        assert self.test_config.generation.top_p == 0.7

    def test_set_invalid_parameter_name(self):
        """Test setting an invalid parameter name."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"invalid_param": "123"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Errors", "Unknown parameter", "invalid_param"],
        )

    def test_set_invalid_temperature_range(self):
        """Test setting temperature outside valid range."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"temperature": "3.0"},  # Above valid range (0.0-2.0)
            raw_input="/set(temperature=3.0)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Errors", "Temperature must be between"],
        )

    def test_set_invalid_top_p_range(self):
        """Test setting top_p outside valid range."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"top_p": "-0.1"},  # Below valid range (0.0-1.0)
            raw_input="/set(top_p=-0.1)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Errors", "top_p must be between"],
        )

    def test_set_invalid_integer_value(self):
        """Test setting integer parameter with non-integer value."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"max_new_tokens": "not_a_number"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Errors", "Integer value expected"],
        )

    def test_set_invalid_float_value(self):
        """Test setting float parameter with non-numeric value."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"temperature": "invalid"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Errors", "Numeric value expected"],
        )

    def test_set_invalid_boolean_value(self):
        """Test setting boolean parameter with invalid value."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"sampling": "maybe"},  # Invalid boolean
            raw_input="/set(sampling=maybe)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Errors", "Boolean value expected"],
        )

    def test_set_boolean_variations(self):
        """Test various boolean value formats."""
        boolean_values = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ]

        for string_value, expected_bool in boolean_values:
            command = ParsedCommand(
                command="set",
                args=[],
                kwargs={"sampling": string_value},
                raw_input="/set(...)",
            )

            result = self.handler.handle_command(command)

            validate_command_result(result, expect_success=True)
            assert self.test_config.generation.use_sampling == expected_bool

    def test_set_negative_seed_invalid(self):
        """Test that negative seed values are rejected."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"seed": "-1"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Errors", "seed must be non-negative"],
        )

    def test_set_zero_seed_valid(self):
        """Test that zero seed is valid."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"seed": "0"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Updated parameters"],
        )
        assert self.test_config.generation.seed == 0

    def test_set_max_tokens_too_large(self):
        """Test that excessively large max_tokens is rejected."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"max_new_tokens": "150000"},  # Above 100,000 limit
            raw_input="/set(max_new_tokens=150000)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Errors", "max_new_tokens must be less than"],
        )

    def test_set_frequency_penalty_range(self):
        """Test frequency_penalty range validation."""
        # Valid values
        for value in ["0.0", "1.5", "-1.5", "2.0", "-2.0"]:
            command = ParsedCommand(
                command="set",
                args=[],
                kwargs={"frequency_penalty": value},
                raw_input="/set(...)",
            )
            result = self.handler.handle_command(command)
            validate_command_result(result, expect_success=True)

        # Invalid values
        for value in ["2.1", "-2.1"]:
            command = ParsedCommand(
                command="set",
                args=[],
                kwargs={"frequency_penalty": value},
                raw_input="/set(...)",
            )
            result = self.handler.handle_command(command)
            validate_command_result(
                result,
                expect_success=False,
                expected_message_parts=["frequency_penalty must be between"],
            )

    def test_set_no_parameters_provided(self):
        """Test set command with no parameters."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["set command requires parameter=value arguments"],
        )

    def test_set_malformed_positional_arg(self):
        """Test set command with malformed positional argument."""
        command = ParsedCommand(
            command="set",
            args=["temperature"],  # Missing = and value
            kwargs={},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        # Should succeed but ignore the malformed arg (only process args with =)
        # Since no valid parameters were set, should fail
        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["No parameters were updated"],
        )

    def test_set_partial_success_with_errors(self):
        """Test set command with mix of valid and invalid parameters."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={
                "temperature": "0.8",  # Valid
                "invalid_param": "123",  # Invalid
                "top_p": "1.5",  # Invalid range
                "max_new_tokens": "200",  # Valid
            },
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        # Should be successful because some parameters were updated
        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Updated parameters", "Errors"],
        )
        assert self.test_config.generation.temperature == 0.8
        assert self.test_config.generation.max_new_tokens == 200

    def test_set_whitespace_handling(self):
        """Test that whitespace in parameter names and values is handled correctly."""
        command = ParsedCommand(
            command="set",
            args=["  temperature = 0.9  "],  # Extra whitespace
            kwargs={"  top_p  ": "  0.8  "},  # Extra whitespace
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(result, expect_success=True)
        assert self.test_config.generation.temperature == 0.9
        assert self.test_config.generation.top_p == 0.8

    def test_set_case_insensitive_parameters(self):
        """Test that parameter names are case-insensitive."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"TEMPERATURE": "0.6", "Top_P": "0.7"},
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(result, expect_success=True)
        assert self.test_config.generation.temperature == 0.6
        assert self.test_config.generation.top_p == 0.7

    def test_set_all_supported_parameters(self):
        """Test setting all supported parameters with valid values."""
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={
                "temperature": "0.8",
                "top_p": "0.9",
                "max_new_tokens": "300",
                "seed": "123",
                "frequency_penalty": "1.0",
                "presence_penalty": "-0.5",
                "min_p": "0.1",
            },
            raw_input="/set(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(result, expect_success=True)
        # Verify all parameters were set
        gen_config = self.test_config.generation
        assert gen_config.temperature == 0.8
        assert gen_config.top_p == 0.9
        assert gen_config.max_new_tokens == 300
        assert gen_config.seed == 123
        assert gen_config.frequency_penalty == 1.0
        assert gen_config.presence_penalty == -0.5
        assert gen_config.min_p == 0.1


class TestParameterManagementHandler:
    """Test suite for ParameterManagementHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()
        self.test_config.generation = GenerationParams()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        self.handler = ParameterManagementHandler(context=self.command_context)

    def test_get_supported_commands(self):
        """Test that handler returns correct supported commands."""
        supported = self.handler.get_supported_commands()
        assert supported == ["set"]

    def test_unsupported_command(self):
        """Test handler with unsupported command."""
        command = ParsedCommand(
            command="unsupported",
            args=[],
            kwargs={},
            raw_input="/unsupported(...)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Unsupported command", "unsupported"],
        )


class TestParameterValidation:
    """Test suite for parameter validation edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()
        self.test_config.generation = GenerationParams()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        self.handler = ParameterManagementHandler(context=self.command_context)

    def test_boundary_values_temperature(self):
        """Test boundary values for temperature parameter."""
        # Test valid boundary values
        for value in ["0.0", "2.0"]:
            command = ParsedCommand(
                command="set",
                args=[],
                kwargs={"temperature": value},
                raw_input="/set(...)",
            )
            result = self.handler.handle_command(command)
            validate_command_result(result, expect_success=True)

        # Test invalid boundary values
        for value in ["-0.1", "2.1"]:
            command = ParsedCommand(
                command="set",
                args=[],
                kwargs={"temperature": value},
                raw_input="/set(...)",
            )
            result = self.handler.handle_command(command)
            validate_command_result(result, expect_success=False)

    def test_boundary_values_integers(self):
        """Test boundary values for integer parameters."""
        # Test valid minimum values for max_new_tokens
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"max_new_tokens": "1"},
            raw_input="/set(...)",
        )
        result = self.handler.handle_command(command)
        validate_command_result(result, expect_success=True)

        # Test invalid values (0 or negative) for max_new_tokens
        command = ParsedCommand(
            command="set",
            args=[],
            kwargs={"max_new_tokens": "0"},
            raw_input="/set(...)",
        )
        result = self.handler.handle_command(command)
        validate_command_result(result, expect_success=False)

    def test_seed_boundary_values(self):
        """Test seed parameter boundary values."""
        # Valid values
        for value in ["0", "42", "999999"]:
            command = ParsedCommand(
                command="set", args=[], kwargs={"seed": value}, raw_input="/set(...)"
            )
            result = self.handler.handle_command(command)
            validate_command_result(result, expect_success=True)

        # Invalid negative value
        command = ParsedCommand(
            command="set", args=[], kwargs={"seed": "-1"}, raw_input="/set(...)"
        )
        result = self.handler.handle_command(command)
        validate_command_result(result, expect_success=False)
