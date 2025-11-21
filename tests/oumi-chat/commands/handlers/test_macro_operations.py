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

"""Unit tests for macro operations command handlers."""

from unittest.mock import Mock, patch

from oumi.core.commands import ParsedCommand
from oumi.core.commands.command_context import CommandContext
from oumi.core.commands.handlers.macro_operations_handler import MacroOperationsHandler
from tests.utils.chat_test_utils import (
    create_test_inference_config,
    validate_command_result,
)


class MockMacroInfo:
    """Mock macro info class for testing."""

    def __init__(
        self, name="test_macro", description="Test description", turns=1, fields=None
    ):
        self.name = name
        self.description = description
        self.turns = turns
        self.fields = fields or []


class MockMacroField:
    """Mock macro field class for testing."""

    def __init__(
        self,
        name="test_field",
        description="Test field",
        required=False,
        placeholder=None,
    ):
        self.name = name
        self.description = description
        self.required = required
        self.placeholder = placeholder


class TestMacroCommand:
    """Test suite for /macro() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Mock macro manager
        self.mock_macro_manager = Mock()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )
        # Since macro_manager is a property, use object.__setattr__ to bypass
        # the setter restriction
        object.__setattr__(
            self.command_context, "_macro_manager", self.mock_macro_manager
        )

        self.handler = MacroOperationsHandler(context=self.command_context)

        # Mock style attributes
        self.handler._style = Mock()
        self.handler._style.use_emoji = True
        self.handler._style.assistant_border_style = "cyan"

    def test_macro_simple_success_no_fields(self):
        """Test successful macro execution with no fields."""
        # Mock successful macro loading
        macro_info = MockMacroInfo(name="simple_macro", fields=[])
        self.mock_macro_manager.load_macro.return_value = (True, None, macro_info)
        self.mock_macro_manager.render_macro.return_value = "Hello, world!"

        command = ParsedCommand(
            command="macro",
            args=["simple_macro.jinja"],
            kwargs={},
            raw_input="/macro(simple_macro.jinja)",
        )

        with patch.object(self.handler, "_parse_macro_turns") as mock_parse_turns:
            mock_parse_turns.return_value = ["Hello, world!"]

            result = self.handler.handle_command(command)

            validate_command_result(
                result,
                expect_success=True,
                expected_message_parts=["Executed macro", "simple_macro"],
            )

        # Verify macro manager was called correctly
        self.mock_macro_manager.load_macro.assert_called_once_with("simple_macro.jinja")
        self.mock_macro_manager.render_macro.assert_called_once_with(macro_info, {})


class TestMacroParsingLogic:
    """Test suite for macro parsing methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MacroOperationsHandler(context=Mock())

    def test_parse_macro_turns_structured_content(self):
        """Test parsing structured conversation turns."""
        rendered_content = """User: What is the weather like?
Assistant: I can help you with weather information.

User: Tell me about tomorrow's forecast.
Assistant: Tomorrow will be sunny.

User: Thank you!"""

        turns = self.handler._parse_macro_turns(rendered_content)

        expected_turns = [
            "What is the weather like?",
            "Tell me about tomorrow's forecast.",
            "Thank you!",
        ]
        assert turns == expected_turns

    def test_parse_macro_turns_human_format(self):
        """Test parsing with Human: format instead of User:."""
        rendered_content = """Human: Hello there
AI: Hello! How can I help?"""

        turns = self.handler._parse_macro_turns(rendered_content)

        expected_turns = [
            "Hello there",
        ]
        assert turns == expected_turns
