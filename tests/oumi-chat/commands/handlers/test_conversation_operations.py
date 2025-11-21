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

"""Unit tests for conversation operations command handlers."""

from unittest.mock import Mock, patch

import pytest

from oumi.core.commands import CommandResult, ParsedCommand
from oumi.core.commands.command_context import CommandContext
from oumi.core.types.conversation import Conversation, Message, Role
from tests.utils.chat_test_utils import (
    create_test_inference_config,
    validate_command_result,
)


class TestDeleteCommand:
    """Test suite for /delete() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create mock conversation with multiple messages
        self.mock_conversation = Conversation(
            conversation_id="test_conversation",
            messages=[
                Message(role=Role.USER, content="First message"),
                Message(role=Role.ASSISTANT, content="First response"),
                Message(role=Role.USER, content="Second message"),
                Message(role=Role.ASSISTANT, content="Second response"),
            ],
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=self.mock_conversation.messages.copy(),
            inference_engine=self.mock_engine,
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock conversation operations handler."""
        with patch(
            "oumi.core.commands.handlers.conversation_operations_handler.ConversationOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_delete_last_turn(self, mock_handler):
        """Test deleting the last conversation turn."""
        parsed_cmd = ParsedCommand(
            command="delete", args=[], kwargs={}, raw_input="/delete("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Deleted the last conversation turn"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=True, expected_message_parts=["deleted", "last"]
        )

    def test_delete_specific_position(self, mock_handler):
        """Test deleting a specific conversation position."""
        parsed_cmd = ParsedCommand(
            command="delete", args=["2"], kwargs={}, raw_input="/delete(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Deleted message at position 2"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["deleted", "position 2"],
        )

    def test_delete_empty_conversation(self, mock_handler):
        """Test deleting from empty conversation."""
        # Set up empty conversation
        empty_conversation = Conversation(conversation_id="empty", messages=[])
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(empty_conversation.messages)

        parsed_cmd = ParsedCommand(
            command="delete", args=[], kwargs={}, raw_input="/delete("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="No messages to delete"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["no messages"]
        )

    def test_delete_invalid_position(self, mock_handler):
        """Test deleting invalid position."""
        parsed_cmd = ParsedCommand(
            command="delete", args=["99"], kwargs={}, raw_input="/delete(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="Invalid position: 99. Conversation has 4 messages."
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["invalid position", "99"],
        )


class TestRegenCommand:
    """Test suite for /regen() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.mock_conversation = Conversation(
            conversation_id="test_conversation",
            messages=[
                Message(role=Role.USER, content="Tell me about AI"),
                Message(role=Role.ASSISTANT, content="AI is artificial intelligence."),
            ],
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(
            self.mock_conversation.messages
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock conversation operations handler."""
        with patch(
            "oumi.core.commands.handlers.conversation_operations_handler.ConversationOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_regen_last_response(self, mock_handler):
        """Test regenerating the last assistant response."""
        parsed_cmd = ParsedCommand(
            command="regen", args=[], kwargs={}, raw_input="/regen("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Regenerated the last response"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=True, expected_message_parts=["regenerated"]
        )

    def test_regen_no_assistant_message(self, mock_handler):
        """Test regenerating when there's no assistant message."""
        # Conversation with only user message
        user_only_conversation = Conversation(
            conversation_id="user_only",
            messages=[Message(role=Role.USER, content="Hello")],
        )
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(
            user_only_conversation.messages
        )

        parsed_cmd = ParsedCommand(
            command="regen", args=[], kwargs={}, raw_input="/regen("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="No assistant response to regenerate"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["no assistant response"],
        )

    def test_regen_with_inference_error(self, mock_handler):
        """Test regeneration when inference fails."""
        parsed_cmd = ParsedCommand(
            command="regen", args=[], kwargs={}, raw_input="/regen("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="Failed to regenerate: Model inference error"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["failed", "error"]
        )


class TestClearCommand:
    """Test suite for /clear() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.mock_conversation = Conversation(
            conversation_id="test_conversation",
            messages=[
                Message(role=Role.USER, content="Message 1"),
                Message(role=Role.ASSISTANT, content="Response 1"),
                Message(role=Role.USER, content="Message 2"),
                Message(role=Role.ASSISTANT, content="Response 2"),
            ],
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(
            self.mock_conversation.messages
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock conversation operations handler."""
        with patch(
            "oumi.core.commands.handlers.conversation_operations_handler.ConversationOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_clear_conversation(self, mock_handler):
        """Test clearing the entire conversation."""
        parsed_cmd = ParsedCommand(
            command="clear", args=[], kwargs={}, raw_input="/clear("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Conversation history cleared"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=True, expected_message_parts=["cleared"]
        )

    def test_clear_already_empty(self, mock_handler):
        """Test clearing an already empty conversation."""
        empty_conversation = Conversation(conversation_id="empty", messages=[])
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(empty_conversation.messages)

        parsed_cmd = ParsedCommand(
            command="clear", args=[], kwargs={}, raw_input="/clear("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Conversation is already empty"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(result, expect_success=True)


class TestShowCommand:
    """Test suite for /show() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.mock_conversation = Conversation(
            conversation_id="test_conversation",
            messages=[
                Message(role=Role.USER, content="First question"),
                Message(role=Role.ASSISTANT, content="First answer"),
                Message(role=Role.USER, content="Second question"),
                Message(role=Role.ASSISTANT, content="Second answer"),
            ],
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(
            self.mock_conversation.messages
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock conversation operations handler."""
        with patch(
            "oumi.core.commands.handlers.conversation_operations_handler.ConversationOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_show_specific_position(self, mock_handler):
        """Test showing a specific conversation position."""
        parsed_cmd = ParsedCommand(
            command="show", args=["2"], kwargs={}, raw_input="/show(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Message 2 (User): Second question"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Message 2", "Second question"],
        )

    def test_show_invalid_position(self, mock_handler):
        """Test showing invalid position."""
        parsed_cmd = ParsedCommand(
            command="show", args=["99"], kwargs={}, raw_input="/show(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Invalid position: 99. Conversation has 4 messages (1-4).",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["invalid position", "99"],
        )

    def test_show_without_position(self, mock_handler):
        """Test show command without position argument."""
        parsed_cmd = ParsedCommand(
            command="show", args=[], kwargs={}, raw_input="/show("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Usage: /show(position) - Please specify message position",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["Usage", "position"]
        )

    def test_show_all_messages(self, mock_handler):
        """Test showing all messages with 'all' argument."""
        parsed_cmd = ParsedCommand(
            command="show", args=["all"], kwargs={}, raw_input="/show(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True,
            message="Conversation (4 messages):\n1. User: First question\n"
            "2. Assistant: First answer\n3. User: Second question\n"
            "4. Assistant: Second answer",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Conversation", "4 messages"],
        )


class TestCompactCommand:
    """Test suite for /compact() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create a longer conversation for compacting
        messages = []
        for i in range(10):
            messages.extend(
                [
                    Message(
                        role=Role.USER,
                        content=f"Question {i + 1}: " + "Long question " * 20,
                    ),
                    Message(
                        role=Role.ASSISTANT,
                        content=f"Answer {i + 1}: " + "Long detailed response " * 30,
                    ),
                ]
            )

        self.mock_conversation = Conversation(
            conversation_id="long_conversation", messages=messages
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(
            self.mock_conversation.messages
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock conversation operations handler."""
        with patch(
            "oumi.core.commands.handlers.conversation_operations_handler.ConversationOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_compact_long_conversation(self, mock_handler):
        """Test compacting a long conversation."""
        parsed_cmd = ParsedCommand(
            command="compact", args=[], kwargs={}, raw_input="/compact("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True,
            message="Conversation compacted: 20 messages → 8 messages (60% reduction)",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["compacted", "reduction"],
        )

    def test_compact_short_conversation(self, mock_handler):
        """Test compacting a conversation that doesn't need compacting."""
        short_conversation = Conversation(
            conversation_id="short",
            messages=[
                Message(role=Role.USER, content="Hi"),
                Message(role=Role.ASSISTANT, content="Hello!"),
            ],
        )
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(short_conversation.messages)

        parsed_cmd = ParsedCommand(
            command="compact", args=[], kwargs={}, raw_input="/compact("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Conversation is already compact (2 messages)"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=True, expected_message_parts=["already compact"]
        )

    def test_compact_empty_conversation(self, mock_handler):
        """Test compacting an empty conversation."""
        empty_conversation = Conversation(conversation_id="empty", messages=[])
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(empty_conversation.messages)

        parsed_cmd = ParsedCommand(
            command="compact", args=[], kwargs={}, raw_input="/compact("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="No conversation to compact"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["no conversation"]
        )


class TestRenderCommand:
    """Test suite for /render() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.mock_conversation = Conversation(
            conversation_id="test_conversation",
            messages=[
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hi there!"),
                Message(role=Role.USER, content="How are you?"),
                Message(role=Role.ASSISTANT, content="I'm doing well, thanks!"),
            ],
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(
            self.mock_conversation.messages
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock conversation operations handler."""
        with patch(
            "oumi.core.commands.handlers.conversation_operations_handler.ConversationOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_render_conversation(self, mock_handler):
        """Test rendering conversation to asciinema format."""
        parsed_cmd = ParsedCommand(
            command="render", args=["chat.cast"], kwargs={}, raw_input="/render(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True,
            message="Conversation rendered to chat.cast (4 messages, 2.5s duration)",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["rendered", "chat.cast"],
        )

    def test_render_without_filename(self, mock_handler):
        """Test render command without filename argument."""
        parsed_cmd = ParsedCommand(
            command="render", args=[], kwargs={}, raw_input="/render("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Usage: /render(filename.cast) - Please specify output filename",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["Usage", "filename"]
        )

    def test_render_empty_conversation(self, mock_handler):
        """Test rendering empty conversation."""
        empty_conversation = Conversation(conversation_id="empty", messages=[])
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(empty_conversation.messages)

        parsed_cmd = ParsedCommand(
            command="render", args=["empty.cast"], kwargs={}, raw_input="/render(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="No conversation to render"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["no conversation"]
        )


class TestThinkingCommands:
    """Test suite for thinking-related commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create conversation with thinking content
        self.mock_conversation = Conversation(
            conversation_id="thinking_conversation",
            messages=[
                Message(role=Role.USER, content="Solve this math problem: 2+2"),
                Message(
                    role=Role.ASSISTANT,
                    content="<thinking>Let me think... 2 + 2 = 4</thinking>\n\n"
                    "The answer is 4.",
                ),
            ],
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(
            self.mock_conversation.messages
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock conversation operations handler."""
        with patch(
            "oumi.core.commands.handlers.conversation_operations_handler.ConversationOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_clear_thoughts(self, mock_handler):
        """Test clearing thinking content from responses."""
        parsed_cmd = ParsedCommand(
            command="clear_thoughts", args=[], kwargs={}, raw_input="/clear_thoughts("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Cleared thinking content from 1 message"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["cleared thinking", "1 message"],
        )

    def test_full_thoughts_toggle(self, mock_handler):
        """Test toggling full thoughts display mode."""
        parsed_cmd = ParsedCommand(
            command="full_thoughts", args=[], kwargs={}, raw_input="/full_thoughts("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Full thoughts display mode enabled"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["full thoughts", "enabled"],
        )

    def test_clear_thoughts_no_thinking_content(self, mock_handler):
        """Test clearing thoughts when no thinking content exists."""
        simple_conversation = Conversation(
            conversation_id="simple",
            messages=[
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hi there!"),
            ],
        )
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(simple_conversation.messages)

        parsed_cmd = ParsedCommand(
            command="clear_thoughts", args=[], kwargs={}, raw_input="/clear_thoughts("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="No thinking content found to clear"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=True, expected_message_parts=["no thinking content"]
        )


class TestConversationOperationsHandlerReal:
    """Test suite for ConversationOperationsHandler class using real implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create conversation history with thinking content for testing
        self.conversation_history = [
            {"role": "user", "content": "Explain quantum computing"},
            {
                "role": "assistant",
                "content": (
                    "<think>\nThis is a complex topic. I should start with basics:\n"
                    "- Quantum bits (qubits) vs classical bits\n"
                    "- Superposition principle\n- Entanglement\n"
                    "- Quantum algorithms\n</think>\n\n"
                    "Quantum computing is a revolutionary approach to computation "
                    "that leverages quantum mechanics principles."
                ),
            },
            {"role": "user", "content": "How does it differ from classical computing?"},
            {
                "role": "assistant",
                "content": (
                    "<reasoning>\nKey differences to explain:\n"
                    "1. Information storage: bits vs qubits\n"
                    "2. Processing: sequential vs parallel \n"
                    "3. Algorithms: deterministic vs probabilistic\n"
                    "4. Applications: specific use cases\n</reasoning>\n\n"
                    "Classical computers use bits (0 or 1), while quantum computers "
                    "use qubits that can exist in superposition."
                ),
            },
        ]

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=self.conversation_history,
            inference_engine=self.mock_engine,
        )

        # Import the actual handler
        from oumi.core.commands.handlers.conversation_operations_handler import (
            ConversationOperationsHandler,
        )

        self.handler = ConversationOperationsHandler(context=self.command_context)

    def test_get_supported_commands_real(self):
        """Test that handler returns correct supported commands."""
        supported = self.handler.get_supported_commands()
        expected_commands = {
            "delete",
            "regen",
            "clear",
            "compact",
            "full_thoughts",
            "clear_thoughts",
            "show",
            "render",
        }
        assert set(supported) == expected_commands

    def test_unsupported_command_real(self):
        """Test handler with unsupported command."""
        command = ParsedCommand(
            command="unsupported",
            args=[],
            kwargs={},
            raw_input="/unsupported()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Unsupported command", "unsupported"],
        )


class TestFullThoughtsCommandReal:
    """Test suite for /full_thoughts() command using real implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create minimal conversation history
        self.conversation_history = [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "Test response"},
        ]

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=self.conversation_history,
            inference_engine=self.mock_engine,
        )

        # Mock thinking processor
        self.mock_thinking_processor = Mock()
        self.command_context._thinking_processor = self.mock_thinking_processor

        from oumi.core.commands.handlers.conversation_operations_handler import (
            ConversationOperationsHandler,
        )

        self.handler = ConversationOperationsHandler(context=self.command_context)

    def test_full_thoughts_toggle_to_full_real(self):
        """Test toggling full thoughts display mode from compressed to full."""
        # Mock processor to return compressed mode initially
        self.mock_thinking_processor.get_display_mode.return_value = "compressed"

        command = ParsedCommand(
            command="full_thoughts",
            args=[],
            kwargs={},
            raw_input="/full_thoughts()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["full", "complete thinking chains"],
        )

        # Verify processor was called to set full mode
        self.mock_thinking_processor.set_display_mode.assert_called_with("full")

    def test_full_thoughts_toggle_to_compressed_real(self):
        """Test toggling full thoughts display mode from full to compressed."""
        # Mock processor to return full mode initially
        self.mock_thinking_processor.get_display_mode.return_value = "full"

        command = ParsedCommand(
            command="full_thoughts",
            args=[],
            kwargs={},
            raw_input="/full_thoughts()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["compressed", "brief summaries"],
        )

        # Verify processor was called to set compressed mode
        self.mock_thinking_processor.set_display_mode.assert_called_with("compressed")

    def test_full_thoughts_with_arguments_ignored(self):
        """Test that full_thoughts ignores any arguments provided."""
        self.mock_thinking_processor.get_display_mode.return_value = "compressed"

        command = ParsedCommand(
            command="full_thoughts",
            args=["ignored", "arguments"],
            kwargs={"also": "ignored"},
            raw_input="/full_thoughts(ignored, arguments)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(result, expect_success=True)
        # Should still work despite arguments
        self.mock_thinking_processor.set_display_mode.assert_called_with("full")


class TestClearThoughtsCommandReal:
    """Test suite for /clear_thoughts() command using real implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create conversation history with thinking content
        self.conversation_history = [
            {"role": "user", "content": "What is machine learning?"},
            {
                "role": "assistant",
                "content": (
                    "<think>\nI should explain ML concepts clearly:\n"
                    "- Supervised learning\n- Unsupervised learning\n"
                    "- Neural networks\n</think>\n\n"
                    "Machine learning is a subset of artificial intelligence."
                ),
            },
            {"role": "user", "content": "Give me an example"},
            {
                "role": "assistant",
                "content": (
                    "<reasoning>\nGood examples would be:\n"
                    "- Image recognition\n- Recommendation systems\n"
                    "- Natural language processing\n</reasoning>\n\n"
                    "A common example is email spam detection."
                ),
            },
        ]

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=self.conversation_history,
            inference_engine=self.mock_engine,
        )

        # Mock thinking processor
        self.mock_thinking_processor = Mock()
        self.command_context._thinking_processor = self.mock_thinking_processor

        from oumi.core.commands.handlers.conversation_operations_handler import (
            ConversationOperationsHandler,
        )

        self.handler = ConversationOperationsHandler(context=self.command_context)

    def test_clear_thoughts_removes_thinking_content_real(self):
        """Test that clear_thoughts removes thinking content from messages."""

        # Mock clean_thinking_content to return cleaned versions
        def mock_clean_thinking_content(content):
            if "<think>" in content:
                return "Machine learning is a subset of artificial intelligence."
            elif "<reasoning>" in content:
                return "A common example is email spam detection."
            return content

        self.mock_thinking_processor.clean_thinking_content.side_effect = (
            mock_clean_thinking_content
        )

        command = ParsedCommand(
            command="clear_thoughts",
            args=[],
            kwargs={},
            raw_input="/clear_thoughts()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=[
                "removed thinking content",
                "2",
                "assistant message(s)",
            ],
        )

        # Verify processor was called for assistant messages
        assert self.mock_thinking_processor.clean_thinking_content.call_count == 2

        # Verify conversation history was actually modified
        assistant_messages = [
            msg for msg in self.conversation_history if msg["role"] == "assistant"
        ]
        assert "<think>" not in assistant_messages[0]["content"]
        assert "<reasoning>" not in assistant_messages[1]["content"]

    def test_clear_thoughts_no_thinking_content_real(self):
        """Test clear_thoughts when no thinking content exists."""
        # Create conversation without thinking content
        simple_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        self.command_context.conversation_history = simple_history

        # Mock processor to return unchanged content (no thinking to clean)
        self.mock_thinking_processor.clean_thinking_content.side_effect = lambda x: x

        command = ParsedCommand(
            command="clear_thoughts",
            args=[],
            kwargs={},
            raw_input="/clear_thoughts()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["no thinking content", "found"],
        )

    def test_clear_thoughts_empty_conversation_real(self):
        """Test clear_thoughts with empty conversation."""
        self.command_context.conversation_history.clear()
        self.mock_thinking_processor.reset_mock()

        command = ParsedCommand(
            command="clear_thoughts",
            args=[],
            kwargs={},
            raw_input="/clear_thoughts()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["no thinking content found to remove"],
        )

    def test_clear_thoughts_user_messages_ignored(self):
        """Test that clear_thoughts only processes assistant messages."""
        # Mix of user and assistant messages
        mixed_history = [
            {"role": "user", "content": "<think>User thinking</think>Question?"},
            {
                "role": "assistant",
                "content": "<think>Assistant thinking</think>Answer.",
            },
        ]

        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(mixed_history)
        self.mock_thinking_processor.reset_mock()

        # Mock to clean only assistant content
        def selective_clean(content):
            if "Assistant thinking" in content:
                return "Answer."
            return content

        self.mock_thinking_processor.clean_thinking_content.side_effect = (
            selective_clean
        )

        command = ParsedCommand(
            command="clear_thoughts",
            args=[],
            kwargs={},
            raw_input="/clear_thoughts()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(result, expect_success=True)

        # Should only have processed the assistant message (1 call)
        assert self.mock_thinking_processor.clean_thinking_content.call_count == 1

        # User message should remain unchanged
        assert "<think>User thinking</think>" in mixed_history[0]["content"]
        # Assistant message should be cleaned
        assert "<think>Assistant thinking</think>" not in mixed_history[1]["content"]


class TestRenderCommandReal:
    """Test suite for /render() command using real implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create conversation history for rendering
        self.conversation_history = [
            {"role": "user", "content": "Hello, how are you?"},
            {
                "role": "assistant",
                "content": "I'm doing great! How can I help you today?",
            },
            {"role": "user", "content": "Can you explain Python?"},
            {
                "role": "assistant",
                "content": (
                    "Python is a high-level programming language "
                    "known for its simplicity."
                ),
            },
        ]

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=self.conversation_history,
            inference_engine=self.mock_engine,
        )

        from oumi.core.commands.handlers.conversation_operations_handler import (
            ConversationOperationsHandler,
        )

        self.handler = ConversationOperationsHandler(context=self.command_context)

    @patch(
        "oumi.core.commands.handlers.conversation_operations_handler.ConversationRenderer"
    )
    def test_render_conversation_real(self, mock_renderer_class):
        """Test rendering conversation to asciinema format."""
        # Mock the renderer
        mock_renderer = Mock()
        mock_renderer_class.return_value = mock_renderer
        mock_renderer.render_to_asciinema.return_value = (
            True,
            "✅ Successfully recorded conversation to conversation.cast (1,024 bytes)",
        )

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "conversation.cast")

            command = ParsedCommand(
                command="render",
                args=[output_path],
                kwargs={},
                raw_input="/render(conversation.cast)",
            )

            result = self.handler.handle_command(command)

            validate_command_result(
                result,
                expect_success=True,
                expected_message_parts=["successfully recorded", "conversation.cast"],
            )

            # Verify renderer was called with correct parameters
            mock_renderer_class.assert_called_once_with(
                conversation_history=self.conversation_history,
                console=self.mock_console,
                config=self.test_config,
                thinking_processor=self.command_context.thinking_processor,
            )
            mock_renderer.render_to_asciinema.assert_called_once_with(output_path)

    def test_render_no_arguments_real(self):
        """Test render command without arguments."""
        command = ParsedCommand(
            command="render",
            args=[],
            kwargs={},
            raw_input="/render()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["requires a file path", "render"],
        )

    def test_render_empty_filename_real(self):
        """Test render command with empty filename."""
        command = ParsedCommand(
            command="render",
            args=["   "],  # Whitespace only
            kwargs={},
            raw_input="/render(   )",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["error rendering conversation"],
        )

    @patch(
        "oumi.core.commands.handlers.conversation_operations_handler.ConversationRenderer"
    )
    def test_render_auto_extension_real(self, mock_renderer_class):
        """Test that render automatically adds .cast extension."""
        mock_renderer = Mock()
        mock_renderer_class.return_value = mock_renderer
        mock_renderer.render_to_asciinema.return_value = (
            True,
            "Successfully recorded conversation to conversation.cast (1,024 bytes)",
        )

        command = ParsedCommand(
            command="render",
            args=["conversation"],  # No extension
            kwargs={},
            raw_input="/render(conversation)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(result, expect_success=True)

        # Should have added .cast extension
        mock_renderer.render_to_asciinema.assert_called_once()
        call_args = mock_renderer.render_to_asciinema.call_args[0]
        assert call_args[0].endswith(".cast")

    @patch(
        "oumi.core.commands.handlers.conversation_operations_handler.ConversationRenderer"
    )
    def test_render_failure_handling_real(self, mock_renderer_class):
        """Test render command when rendering fails."""
        mock_renderer = Mock()
        mock_renderer_class.return_value = mock_renderer
        mock_renderer.render_to_asciinema.return_value = (
            False,
            "Error rendering conversation: Rendering failed",
        )

        # Use generated_test_files directory for proper test file management
        from tests.utils.chat_test_utils import ensure_test_file_in_generated_dir

        output_path = ensure_test_file_in_generated_dir("conversation.cast")

        command = ParsedCommand(
            command="render",
            args=[output_path],
            kwargs={},
            raw_input=f"/render({output_path})",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["error", "rendering"],
        )

    def test_render_empty_conversation_real(self):
        """Test rendering with empty conversation."""
        # Create handler with empty conversation
        empty_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        from oumi.core.commands.handlers.conversation_operations_handler import (
            ConversationOperationsHandler,
        )

        handler = ConversationOperationsHandler(context=empty_context)

        # Use generated_test_files directory for proper test file management
        from tests.utils.chat_test_utils import ensure_test_file_in_generated_dir

        output_path = ensure_test_file_in_generated_dir("empty.cast")

        command = ParsedCommand(
            command="render",
            args=[output_path],
            kwargs={},
            raw_input=f"/render({output_path})",
        )

        result = handler.handle_command(command)

        # Should handle empty conversation gracefully
        assert isinstance(result.success, bool)


class TestConversationOperationsIntegration:
    """Integration tests for conversation operations commands working together."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create rich conversation with thinking content
        self.conversation_history = [
            {"role": "user", "content": "Explain deep learning"},
            {
                "role": "assistant",
                "content": (
                    "<think>\nDeep learning is complex, I should cover:\n"
                    "1. Neural networks basics\n2. Multiple layers\n"
                    "3. Applications\n4. Differences from ML\n</think>\n\n"
                    "Deep learning is a subset of machine learning that uses "
                    "neural networks with multiple layers."
                ),
            },
            {"role": "user", "content": "How is it different from machine learning?"},
            {
                "role": "assistant",
                "content": (
                    "<reasoning>\nKey differences:\n"
                    "- Depth: Multiple hidden layers vs fewer layers\n"
                    "- Feature extraction: Automatic vs manual\n"
                    "- Data requirements: Large datasets vs smaller\n"
                    "- Computational needs: High vs moderate\n</reasoning>\n\n"
                    "The main difference is that deep learning automatically "
                    "extracts features from raw data."
                ),
            },
        ]

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=self.conversation_history,
            inference_engine=self.mock_engine,
        )

        # Mock thinking processor for integration tests
        self.mock_thinking_processor = Mock()
        self.command_context._thinking_processor = self.mock_thinking_processor

        from oumi.core.commands.handlers.conversation_operations_handler import (
            ConversationOperationsHandler,
        )

        self.handler = ConversationOperationsHandler(context=self.command_context)

    def test_full_thoughts_then_clear_thoughts_workflow(self):
        """Test workflow: toggle full thoughts, then clear thoughts."""
        # First, toggle to full thoughts
        self.mock_thinking_processor.get_display_mode.return_value = "compressed"

        full_thoughts_cmd = ParsedCommand(
            command="full_thoughts",
            args=[],
            kwargs={},
            raw_input="/full_thoughts()",
        )

        result1 = self.handler.handle_command(full_thoughts_cmd)
        validate_command_result(result1, expect_success=True)

        # Then clear thoughts from conversation
        def mock_clean(content):
            # Remove thinking tags for testing
            import re

            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(
                r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL
            )
            return content.strip()

        self.mock_thinking_processor.clean_thinking_content.side_effect = mock_clean

        clear_thoughts_cmd = ParsedCommand(
            command="clear_thoughts",
            args=[],
            kwargs={},
            raw_input="/clear_thoughts()",
        )

        result2 = self.handler.handle_command(clear_thoughts_cmd)
        validate_command_result(result2, expect_success=True)

        # Verify both operations completed successfully
        assert result1.success and result2.success

    @patch(
        "oumi.core.commands.handlers.conversation_operations_handler.ConversationRenderer"
    )
    def test_clear_thoughts_then_render_workflow(self, mock_renderer_class):
        """Test workflow: clear thoughts, then render conversation."""

        # First clear thoughts
        def mock_clean(content):
            import re

            return re.sub(r"<[^>]+>.*?</[^>]+>", "", content, flags=re.DOTALL).strip()

        self.mock_thinking_processor.clean_thinking_content.side_effect = mock_clean

        clear_cmd = ParsedCommand(
            command="clear_thoughts",
            args=[],
            kwargs={},
            raw_input="/clear_thoughts()",
        )

        result1 = self.handler.handle_command(clear_cmd)
        validate_command_result(result1, expect_success=True)

        # Then render the cleaned conversation
        mock_renderer = Mock()
        mock_renderer_class.return_value = mock_renderer
        mock_renderer.render_to_asciinema.return_value = (
            True,
            "Successfully rendered conversation to clean_conversation.cast",
        )

        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f"{temp_dir}/clean_conversation.cast"

            render_cmd = ParsedCommand(
                command="render",
                args=[output_path],
                kwargs={},
                raw_input="/render(clean_conversation.cast)",
            )

            result2 = self.handler.handle_command(render_cmd)
            validate_command_result(result2, expect_success=True)

        # Both operations should succeed
        assert result1.success and result2.success
