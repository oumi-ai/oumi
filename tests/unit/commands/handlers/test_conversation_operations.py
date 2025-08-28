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
        self.command_context.conversation_history.extend(self.mock_conversation.messages)

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
        self.command_context.conversation_history.extend(user_only_conversation.messages)

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
        self.command_context.conversation_history.extend(self.mock_conversation.messages)

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
        self.command_context.conversation_history.extend(self.mock_conversation.messages)

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
        self.command_context.conversation_history.extend(self.mock_conversation.messages)

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
        self.command_context.conversation_history.extend(self.mock_conversation.messages)

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
        self.command_context.conversation_history.extend(self.mock_conversation.messages)

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
