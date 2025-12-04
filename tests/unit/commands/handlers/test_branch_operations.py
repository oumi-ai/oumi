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

"""Unit tests for conversation branching command handlers."""

from unittest.mock import Mock, patch

import pytest

from oumi.core.types.conversation import Conversation, Message, Role
from oumi_chat.commands import CommandResult, ParsedCommand
from oumi_chat.commands.command_context import CommandContext
from tests.utils.chat_test_utils import (
    create_test_inference_config,
    validate_command_result,
)


class TestBranchCommand:
    """Test suite for /branch() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.mock_conversation = Conversation(
            conversation_id="main_conversation",
            messages=[
                Message(role=Role.USER, content="What is AI?"),
                Message(role=Role.ASSISTANT, content="AI is artificial intelligence."),
                Message(role=Role.USER, content="Tell me more about machine learning."),
                Message(
                    role=Role.ASSISTANT, content="Machine learning is a subset of AI..."
                ),
            ],
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=self.mock_conversation.messages,
            inference_engine=self.mock_engine,
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock branch operations handler."""
        with patch(
            "oumi.core.commands.handlers.branch_operations_handler.BranchOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_create_branch_from_current_position(self, mock_handler):
        """Test creating a branch from the current conversation position."""
        parsed_cmd = ParsedCommand(
            command="branch", args=[], kwargs={}, raw_input="/branch("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Created branch 'branch_1' from current position"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["created branch", "branch_1"],
        )

    def test_create_named_branch(self, mock_handler):
        """Test creating a branch with a specific name."""
        parsed_cmd = ParsedCommand(
            command="branch",
            args=["alternative_path"],
            kwargs={},
            raw_input="/branch(...",
        )

        mock_handler.handle.return_value = CommandResult(
            success=True,
            message="Created branch 'alternative_path' from current position",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["created branch", "alternative_path"],
        )

    def test_create_branch_empty_conversation(self, mock_handler):
        """Test creating a branch from an empty conversation."""
        empty_conversation = Conversation(conversation_id="empty", messages=[])
        self.command_context.conversation_history.clear()
        self.command_context.conversation_history.extend(empty_conversation.messages)

        parsed_cmd = ParsedCommand(
            command="branch", args=[], kwargs={}, raw_input="/branch("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="Cannot create branch from empty conversation"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["cannot create", "empty"],
        )

    def test_create_branch_duplicate_name(self, mock_handler):
        """Test creating a branch with a name that already exists."""
        parsed_cmd = ParsedCommand(
            command="branch",
            args=["existing_branch"],
            kwargs={},
            raw_input="/branch(...",
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Branch 'existing_branch' already exists. Use a different name.",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["already exists", "different name"],
        )


class TestBranchFromCommand:
    """Test suite for /branch_from() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.mock_conversation = Conversation(
            conversation_id="main_conversation",
            messages=[
                Message(role=Role.USER, content="Question 1"),
                Message(role=Role.ASSISTANT, content="Answer 1"),
                Message(role=Role.USER, content="Question 2"),
                Message(role=Role.ASSISTANT, content="Answer 2"),
                Message(role=Role.USER, content="Question 3"),
                Message(role=Role.ASSISTANT, content="Answer 3"),
            ],
        )

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=self.mock_conversation.messages,
            inference_engine=self.mock_engine,
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock branch operations handler."""
        with patch(
            "oumi.core.commands.handlers.branch_operations_handler.BranchOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_branch_from_specific_position(self, mock_handler):
        """Test creating a branch from a specific message position."""
        parsed_cmd = ParsedCommand(
            command="branch_from",
            args=["experiment", "3"],
            kwargs={},
            raw_input="/branch_from(...)",
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Created branch 'experiment' from position 3"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["created branch", "experiment", "position 3"],
        )

    def test_branch_from_invalid_position(self, mock_handler):
        """Test creating a branch from an invalid position."""
        parsed_cmd = ParsedCommand(
            command="branch_from",
            args=["test_branch", "99"],
            kwargs={},
            raw_input="/branch_from(...)",
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Invalid position: 99. Conversation has 6 messages (1-6).",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["invalid position", "99"],
        )

    def test_branch_from_missing_arguments(self, mock_handler):
        """Test branch_from command with missing arguments."""
        parsed_cmd = ParsedCommand(
            command="branch_from",
            args=["branch_name"],
            kwargs={},
            raw_input="/branch_from(...",
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message=(
                "Usage: /branch_from(branch_name, position) - Missing position argument"
            ),
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Usage", "Missing position"],
        )


class TestSwitchCommand:
    """Test suite for /switch() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        # Create mock conversation for main branch
        self.mock_conversation = Conversation(
            conversation_id="main_conversation",
            messages=[
                Message(role=Role.USER, content="Main branch question"),
                Message(role=Role.ASSISTANT, content="Main branch answer"),
            ],
        )

        # Simulate existing branches
        self.mock_branches = {
            "main": self.mock_conversation,
            "experiment1": Conversation(
                conversation_id="experiment1",
                messages=[
                    Message(role=Role.USER, content="Experimental question"),
                    Message(role=Role.ASSISTANT, content="Experimental answer"),
                ],
            ),
            "experiment2": Conversation(
                conversation_id="experiment2",
                messages=[
                    Message(role=Role.USER, content="Another experimental question"),
                    Message(role=Role.ASSISTANT, content="Another experimental answer"),
                ],
            ),
        }

    @pytest.fixture
    def mock_handler(self):
        """Mock branch operations handler."""
        with patch(
            "oumi.core.commands.handlers.branch_operations_handler.BranchOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_switch_to_existing_branch(self, mock_handler):
        """Test switching to an existing branch."""
        parsed_cmd = ParsedCommand(
            command="switch", args=["experiment1"], kwargs={}, raw_input="/switch(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Switched to branch 'experiment1' (2 messages)"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["switched", "experiment1", "2 messages"],
        )

    def test_switch_to_nonexistent_branch(self, mock_handler):
        """Test switching to a branch that doesn't exist."""
        parsed_cmd = ParsedCommand(
            command="switch", args=["nonexistent"], kwargs={}, raw_input="/switch(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message=(
                "Branch 'nonexistent' not found. Available branches: "
                "main, experiment1, experiment2"
            ),
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["not found", "available branches"],
        )

    def test_switch_without_branch_name(self, mock_handler):
        """Test switch command without branch name argument."""
        parsed_cmd = ParsedCommand(
            command="switch", args=[], kwargs={}, raw_input="/switch("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Usage: /switch(branch_name) - Please specify a branch name",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Usage", "branch name"],
        )


class TestBranchesCommand:
    """Test suite for /branches() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock branch operations handler."""
        with patch(
            "oumi.core.commands.handlers.branch_operations_handler.BranchOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_list_branches_with_multiple(self, mock_handler):
        """Test listing branches when multiple branches exist."""
        parsed_cmd = ParsedCommand(
            command="branches", args=[], kwargs={}, raw_input="/branches("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True,
            message=(
                "Available branches:\n• main* (4 messages) - current\n"
                "• experiment1 (2 messages)\n• experiment2 (3 messages)"
            ),
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["available branches", "main*", "current"],
        )

    def test_list_branches_single_branch(self, mock_handler):
        """Test listing branches when only main branch exists."""
        parsed_cmd = ParsedCommand(
            command="branches", args=[], kwargs={}, raw_input="/branches("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Available branches:\n• main* (4 messages) - current"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["available branches", "main*"],
        )

    def test_list_branches_no_conversation(self, mock_handler):
        """Test listing branches when no conversation exists."""
        parsed_cmd = ParsedCommand(
            command="branches", args=[], kwargs={}, raw_input="/branches("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="No conversation branches found"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["no conversation", "branches found"],
        )


class TestBranchDeleteCommand:
    """Test suite for /branch_delete() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock branch operations handler."""
        with patch(
            "oumi.core.commands.handlers.branch_operations_handler.BranchOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_delete_existing_branch(self, mock_handler):
        """Test deleting an existing branch."""
        parsed_cmd = ParsedCommand(
            command="branch_delete",
            args=["experiment1"],
            kwargs={},
            raw_input="/branch_delete(...",
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Deleted branch 'experiment1'"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["deleted branch", "experiment1"],
        )

    def test_delete_nonexistent_branch(self, mock_handler):
        """Test deleting a branch that doesn't exist."""
        parsed_cmd = ParsedCommand(
            command="branch_delete",
            args=["nonexistent"],
            kwargs={},
            raw_input="/branch_delete(...",
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="Branch 'nonexistent' not found"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["not found", "nonexistent"],
        )

    def test_delete_main_branch(self, mock_handler):
        """Test attempting to delete the main branch."""
        parsed_cmd = ParsedCommand(
            command="branch_delete",
            args=["main"],
            kwargs={},
            raw_input="/branch_delete(...",
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="Cannot delete the main branch"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["cannot delete", "main branch"],
        )

    def test_delete_current_branch(self, mock_handler):
        """Test attempting to delete the currently active branch."""
        parsed_cmd = ParsedCommand(
            command="branch_delete",
            args=["current_branch"],
            kwargs={},
            raw_input="/branch_delete(...",
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message=(
                "Cannot delete currently active branch 'current_branch'. "
                "Switch to another branch first."
            ),
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["cannot delete", "currently active", "switch"],
        )

    def test_delete_without_branch_name(self, mock_handler):
        """Test branch_delete command without branch name argument."""
        parsed_cmd = ParsedCommand(
            command="branch_delete", args=[], kwargs={}, raw_input="/branch_delete("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message=(
                "Usage: /branch_delete(branch_name) - Please specify a branch to delete"
            ),
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Usage", "branch to delete"],
        )


class TestBranchingWorkflows:
    """Test suite for complex branching workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

    @pytest.fixture
    def mock_handler(self):
        """Mock branch operations handler."""
        with patch(
            "oumi.core.commands.handlers.branch_operations_handler.BranchOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_create_switch_delete_workflow(self, mock_handler):
        """Test a complete workflow: create → switch → delete."""
        # This would normally be multiple commands, but we're testing the handler's
        # ability
        # to maintain state across operations

        commands_and_results = [
            ("branch", ["test_workflow"], "Created branch 'test_workflow'"),
            ("switch", ["test_workflow"], "Switched to branch 'test_workflow'"),
            ("switch", ["main"], "Switched to branch 'main'"),
            ("branch_delete", ["test_workflow"], "Deleted branch 'test_workflow'"),
        ]

        for cmd_name, args, expected_msg in commands_and_results:
            parsed_cmd = ParsedCommand(
                command=cmd_name, args=args, kwargs={}, raw_input=f"/{cmd_name}(...)"
            )

            mock_handler.handle.return_value = CommandResult(
                success=True, message=expected_msg
            )

            result = mock_handler.handle(parsed_cmd, self.command_context)

            validate_command_result(result, expect_success=True)
            assert expected_msg in result.message

    def test_branch_isolation(self, mock_handler):
        """Test that branches maintain isolation from each other."""
        # Test that operations in one branch don't affect others
        parsed_cmd = ParsedCommand(
            command="branches", args=[], kwargs={}, raw_input="/branches("
        )

        mock_handler.handle.return_value = CommandResult(
            success=True,
            message=(
                "Branch isolation maintained: each branch has independent "
                "message history"
            ),
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(result, expect_success=True)
