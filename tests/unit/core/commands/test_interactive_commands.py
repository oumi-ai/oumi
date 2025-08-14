"""Comprehensive tests for all oumi interactive chat commands."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from src.oumi.core.commands.handlers.conversation_operations_handler import ConversationOperationsHandler
from src.oumi.core.commands.handlers.branch_operations_handler import BranchOperationsHandler
from src.oumi.core.commands.handlers.file_operations_handler import FileOperationsHandler
from src.oumi.core.commands.handlers.model_swap_handler import ModelSwapHandler
from src.oumi.core.commands.handlers.parameter_handler import ParameterHandler
from src.oumi.core.commands.command_parser import CommandParser, ParsedCommand
from src.oumi.core.commands.command_context import CommandContext
from src.oumi.core.commands.conversation_branches import ConversationBranchManager
from src.oumi.core.commands.base_handler import CommandResult
from src.oumi.core.attachments.file_handler import FileInfo, AttachmentResult, FileType, ProcessingStrategy
from src.oumi.core.configs.inference_config import InferenceConfig


@pytest.fixture
def mock_config():
    """Mock inference config."""
    config = MagicMock(spec=InferenceConfig)
    config.model.model_max_length = 4096
    config.model.model_name = "test-model"
    config.generation.temperature = 0.7
    config.generation.top_p = 0.9
    config.generation.max_new_tokens = 2048
    return config


@pytest.fixture
def conversation_history():
    """Mock conversation history."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "Can you explain quantum computing?"},
        {"role": "assistant", "content": "Quantum computing is a fascinating field that uses quantum mechanical phenomena..."},
    ]


@pytest.fixture
def command_context(mock_config, conversation_history):
    """Mock command context with all required components."""
    context = MagicMock(spec=CommandContext)
    context.config = mock_config
    context.conversation_history = conversation_history.copy()
    context.system_monitor = MagicMock()
    context.context_window_manager = MagicMock()
    context.file_handler = MagicMock()
    context.compaction_engine = MagicMock()
    
    # Mock branch manager
    branch_manager = MagicMock()
    branch_manager.current_branch_id = "main"
    branch_manager.create_branch.return_value = (True, "Branch 'test' created successfully", MagicMock())
    branch_manager.switch_branch.return_value = (True, "Switched to branch 'test'", MagicMock())
    branch_manager.get_all_branches.return_value = {"main": MagicMock(), "test": MagicMock()}
    branch_manager.delete_branch.return_value = (True, "Branch 'test' deleted")
    context.branch_manager = branch_manager
    
    return context


@pytest.fixture
def command_parser():
    """Command parser instance."""
    return CommandParser()


class TestConversationOperationsHandler:
    """Test suite for conversation operations commands."""

    @pytest.fixture
    def handler(self, command_context):
        """Create conversation operations handler."""
        return ConversationOperationsHandler(command_context)

    def test_delete_command_removes_last_turn(self, handler, command_parser):
        """Test that /delete() removes the last user+assistant turn."""
        command = command_parser.parse_command("/delete()")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "Deleted" in result.message
        assert len(handler.conversation_history) == 2  # Should have 2 messages left

    def test_delete_command_no_history(self, handler, command_parser):
        """Test /delete() with empty conversation history."""
        handler.conversation_history.clear()
        command = command_parser.parse_command("/delete()")
        result = handler.handle_command(command)
        
        assert result.success is False
        assert "No conversation history" in result.message

    def test_regen_command_regenerates_response(self, handler, command_parser):
        """Test that /regen() regenerates the last assistant response."""
        command = command_parser.parse_command("/regen()")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "Regenerating" in result.message
        assert result.should_continue is True
        assert result.user_input_override is not None
        # Should have removed last assistant message
        assert len(handler.conversation_history) == 3
        assert handler.conversation_history[-1]["role"] == "user"

    def test_regen_command_no_user_message(self, handler, command_parser):
        """Test /regen() when no user message exists."""
        # Clear history and add only system message
        handler.conversation_history.clear()
        handler.conversation_history.append({"role": "system", "content": "You are a helpful assistant"})
        
        command = command_parser.parse_command("/regen()")
        result = handler.handle_command(command)
        
        assert result.success is False
        assert "No user message found" in result.message

    def test_clear_command_clears_history(self, handler, command_parser):
        """Test that /clear() clears all conversation history."""
        original_count = len(handler.conversation_history)
        command = command_parser.parse_command("/clear()")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert f"Cleared {original_count}" in result.message
        assert len(handler.conversation_history) == 0

    def test_clear_command_empty_history(self, handler, command_parser):
        """Test /clear() with already empty history."""
        handler.conversation_history.clear()
        command = command_parser.parse_command("/clear()")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "Cleared 0" in result.message

    @patch('src.oumi.core.commands.handlers.conversation_operations_handler.CompactionEngine')
    def test_compact_command_compresses_history(self, mock_compaction, handler, command_parser):
        """Test that /compact() compresses conversation history."""
        # Mock compaction engine
        mock_engine = MagicMock()
        mock_engine.compact_conversation.return_value = [
            {"role": "user", "content": "Summary of previous conversation"},
            {"role": "assistant", "content": "I understand. How can I help?"},
        ]
        handler.context.compaction_engine = mock_engine
        
        command = command_parser.parse_command("/compact()")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "Compacted conversation" in result.message
        mock_engine.compact_conversation.assert_called_once()

    def test_full_thoughts_toggle(self, handler, command_parser):
        """Test /full_thoughts() toggles thinking display mode."""
        command = command_parser.parse_command("/full_thoughts()")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "thinking display" in result.message.lower()

    def test_clear_thoughts_removes_thinking(self, handler, command_parser):
        """Test /clear_thoughts() removes thinking content from history."""
        # Add messages with thinking content
        handler.conversation_history.append({
            "role": "assistant", 
            "content": "<thinking>This is internal reasoning</thinking>\nFinal answer here"
        })
        
        command = command_parser.parse_command("/clear_thoughts()")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "Cleaned thinking content" in result.message


class TestBranchOperationsHandler:
    """Test suite for branch operations commands."""

    @pytest.fixture
    def handler(self, command_context):
        """Create branch operations handler."""
        return BranchOperationsHandler(command_context)

    def test_branch_create_success(self, handler, command_parser):
        """Test successful branch creation."""
        command = command_parser.parse_command("/branch(test_branch)")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "test_branch" in result.message
        handler.context.branch_manager.create_branch.assert_called_once()

    def test_branch_create_no_name(self, handler, command_parser):
        """Test branch creation without name generates default name."""
        command = command_parser.parse_command("/branch()")
        result = handler.handle_command(command)
        
        assert result.success is True
        handler.context.branch_manager.create_branch.assert_called_once()

    def test_branch_create_failure(self, handler, command_parser):
        """Test branch creation failure."""
        handler.context.branch_manager.create_branch.return_value = (
            False, "Maximum branches reached", None
        )
        
        command = command_parser.parse_command("/branch(test_branch)")
        result = handler.handle_command(command)
        
        assert result.success is False
        assert "Maximum branches reached" in result.message

    def test_switch_branch_success(self, handler, command_parser):
        """Test successful branch switching."""
        command = command_parser.parse_command("/switch(test_branch)")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "test_branch" in result.message
        handler.context.branch_manager.switch_branch.assert_called_once_with("test_branch")

    def test_switch_branch_no_name(self, handler, command_parser):
        """Test switch command without branch name."""
        command = command_parser.parse_command("/switch()")
        result = handler.handle_command(command)
        
        assert result.success is False
        assert "requires a branch name" in result.message

    def test_branches_list(self, handler, command_parser):
        """Test /branches() lists all branches."""
        command = command_parser.parse_command("/branches()")
        result = handler.handle_command(command)
        
        assert result.success is True
        handler.context.branch_manager.get_all_branches.assert_called_once()

    def test_branch_delete_success(self, handler, command_parser):
        """Test successful branch deletion."""
        command = command_parser.parse_command("/branch_delete(test_branch)")
        result = handler.handle_command(command)
        
        assert result.success is True
        handler.context.branch_manager.delete_branch.assert_called_once_with("test_branch")


class TestFileOperationsHandler:
    """Test suite for file operations commands."""

    @pytest.fixture
    def handler(self, command_context):
        """Create file operations handler."""
        return FileOperationsHandler(command_context)

    @pytest.fixture
    def mock_file_info(self):
        """Mock file info."""
        return FileInfo(
            path="/test/path/document.pdf",
            name="document.pdf",
            size_bytes=1024000,  # 1MB
            file_type=FileType.PDF,
            mime_type="application/pdf",
            processing_strategy=ProcessingStrategy.FULL_CONTENT
        )

    @pytest.fixture
    def mock_attachment_result(self, mock_file_info):
        """Mock attachment result."""
        return AttachmentResult(
            file_info=mock_file_info,
            text_content="Document content here",
            success=True,
            context_info="PDF attached (1.0 MB)"
        )

    def test_attach_file_success(self, handler, command_parser, mock_attachment_result):
        """Test successful file attachment."""
        handler.context.file_handler.attach_file.return_value = mock_attachment_result
        
        command = command_parser.parse_command('/attach("/path/to/file.pdf")')
        result = handler.handle_command(command)
        
        assert result.success is True
        handler.context.file_handler.attach_file.assert_called_once()

    def test_attach_file_no_path(self, handler, command_parser):
        """Test attach command without file path."""
        command = command_parser.parse_command("/attach()")
        result = handler.handle_command(command)
        
        assert result.success is False
        assert "file path is required" in result.message

    def test_attach_file_not_found(self, handler, command_parser):
        """Test attach command with non-existent file."""
        handler.context.file_handler.attach_file.return_value = AttachmentResult(
            file_info=MagicMock(),
            text_content="",
            success=False,
            warning_message="File not found"
        )
        
        command = command_parser.parse_command('/attach("/nonexistent/file.pdf")')
        result = handler.handle_command(command)
        
        assert result.success is False

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.oumi.core.commands.handlers.file_operations_handler.json.dumps')
    def test_save_command_json(self, mock_json_dumps, mock_file, handler, command_parser):
        """Test /save() command with JSON format."""
        mock_json_dumps.return_value = '{"conversation": "data"}'
        
        command = command_parser.parse_command('/save("conversation.json")')
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "conversation.json" in result.message
        mock_file.assert_called_once_with("conversation.json", "w", encoding="utf-8")

    @patch('builtins.open', new_callable=mock_open)
    def test_save_command_txt(self, mock_file, handler, command_parser):
        """Test /save() command with text format."""
        command = command_parser.parse_command('/save("conversation.txt")')
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "conversation.txt" in result.message
        mock_file.assert_called_once_with("conversation.txt", "w", encoding="utf-8")

    def test_save_command_no_path(self, handler, command_parser):
        """Test /save() command without file path."""
        command = command_parser.parse_command("/save()")
        result = handler.handle_command(command)
        
        assert result.success is False
        assert "file path is required" in result.message

    @patch('builtins.open', new_callable=mock_open, read_data='{"conversation": []}')
    @patch('src.oumi.core.commands.handlers.file_operations_handler.json.loads')
    def test_import_command_json(self, mock_json_loads, mock_file, handler, command_parser):
        """Test /import() command with JSON format."""
        mock_json_loads.return_value = {
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        
        command = command_parser.parse_command('/import("conversation.json")')
        result = handler.handle_command(command)
        
        assert result.success is True
        assert len(handler.conversation_history) > 0


class TestModelSwapHandler:
    """Test suite for model swap commands."""

    @pytest.fixture
    def handler(self, command_context):
        """Create model swap handler."""
        return ModelSwapHandler(command_context)

    def test_swap_model_success(self, handler, command_parser):
        """Test successful model swap."""
        command = command_parser.parse_command('/swap("gpt-4")')
        
        # Mock the model swap process
        with patch.object(handler, '_swap_model', return_value=True):
            result = handler.handle_command(command)
        
        assert result.success is True
        assert "gpt-4" in result.message

    def test_swap_model_no_name(self, handler, command_parser):
        """Test swap command without model name."""
        command = command_parser.parse_command("/swap()")
        result = handler.handle_command(command)
        
        assert result.success is False
        assert "model identifier is required" in result.message

    def test_swap_config_success(self, handler, command_parser):
        """Test config-based model swap."""
        command = command_parser.parse_command('/swap("config:path/to/config.yaml")')
        
        with patch.object(handler, '_swap_config', return_value=True):
            result = handler.handle_command(command)
        
        assert result.success is True

    def test_list_engines(self, handler, command_parser):
        """Test /list_engines() command."""
        command = command_parser.parse_command("/list_engines()")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "engines" in result.message.lower()


class TestParameterHandler:
    """Test suite for parameter adjustment commands."""

    @pytest.fixture
    def handler(self, command_context):
        """Create parameter handler."""
        return ParameterHandler(command_context)

    def test_set_temperature(self, handler, command_parser):
        """Test /set(temperature=0.8) command."""
        command = command_parser.parse_command("/set(temperature=0.8)")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "temperature=0.8" in result.message
        assert handler.context.config.generation.temperature == 0.8

    def test_set_top_p(self, handler, command_parser):
        """Test /set(top_p=0.9) command."""
        command = command_parser.parse_command("/set(top_p=0.9)")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "top_p=0.9" in result.message

    def test_set_max_tokens(self, handler, command_parser):
        """Test /set(max_tokens=1024) command."""
        command = command_parser.parse_command("/set(max_tokens=1024)")
        result = handler.handle_command(command)
        
        assert result.success is True
        assert "max_tokens=1024" in result.message

    def test_set_invalid_parameter(self, handler, command_parser):
        """Test /set() with invalid parameter."""
        command = command_parser.parse_command("/set(invalid_param=0.5)")
        result = handler.handle_command(command)
        
        assert result.success is False
        assert "invalid_param" in result.message

    def test_set_no_parameters(self, handler, command_parser):
        """Test /set() without parameters."""
        command = command_parser.parse_command("/set()")
        result = handler.handle_command(command)
        
        assert result.success is False
        assert "parameters are required" in result.message


class TestCommandParser:
    """Test suite for command parsing."""

    @pytest.fixture
    def parser(self):
        """Create command parser."""
        return CommandParser()

    def test_parse_simple_command(self, parser):
        """Test parsing simple command without arguments."""
        command = parser.parse_command("/help()")
        
        assert command.command == "help"
        assert command.args == []
        assert command.raw_input == "/help()"

    def test_parse_command_with_args(self, parser):
        """Test parsing command with arguments."""
        command = parser.parse_command('/branch("test_branch")')
        
        assert command.command == "branch"
        assert command.args == ["test_branch"]

    def test_parse_command_multiple_args(self, parser):
        """Test parsing command with multiple arguments."""
        command = parser.parse_command('/set(temperature=0.8, top_p=0.9)')
        
        assert command.command == "set"
        assert len(command.args) >= 1

    def test_parse_invalid_command(self, parser):
        """Test parsing invalid command format."""
        command = parser.parse_command("invalid command")
        
        assert command.command == ""
        assert not command.is_valid

    def test_parse_quoted_paths(self, parser):
        """Test parsing commands with quoted file paths."""
        command = parser.parse_command('/attach("/path/to/file with spaces.pdf")')
        
        assert command.command == "attach"
        assert len(command.args) >= 1


class TestIntegrationScenarios:
    """Test complete workflow scenarios."""

    @pytest.fixture
    def full_context(self, command_context):
        """Create full command context with all handlers."""
        return command_context

    def test_complete_conversation_workflow(self, full_context, command_parser):
        """Test complete conversation management workflow."""
        # Start with conversation
        conv_handler = ConversationOperationsHandler(full_context)
        
        # Test delete
        delete_cmd = command_parser.parse_command("/delete()")
        result = conv_handler.handle_command(delete_cmd)
        assert result.success is True
        
        # Test regen
        regen_cmd = command_parser.parse_command("/regen()")
        result = conv_handler.handle_command(regen_cmd)
        assert result.success is True
        
        # Test clear
        clear_cmd = command_parser.parse_command("/clear()")
        result = conv_handler.handle_command(clear_cmd)
        assert result.success is True

    def test_branch_workflow(self, full_context, command_parser):
        """Test complete branching workflow."""
        branch_handler = BranchOperationsHandler(full_context)
        
        # Create branch
        create_cmd = command_parser.parse_command("/branch(experiment)")
        result = branch_handler.handle_command(create_cmd)
        assert result.success is True
        
        # List branches
        list_cmd = command_parser.parse_command("/branches()")
        result = branch_handler.handle_command(list_cmd)
        assert result.success is True
        
        # Switch branch
        switch_cmd = command_parser.parse_command("/switch(experiment)")
        result = branch_handler.handle_command(switch_cmd)
        assert result.success is True

    def test_file_workflow(self, full_context, command_parser):
        """Test complete file operations workflow."""
        file_handler = FileOperationsHandler(full_context)
        
        # Mock successful attachment
        mock_result = AttachmentResult(
            file_info=FileInfo(
                path="/test.txt",
                name="test.txt", 
                size_bytes=100,
                file_type=FileType.TEXT,
                mime_type="text/plain",
                processing_strategy=ProcessingStrategy.FULL_CONTENT
            ),
            text_content="Test content",
            success=True
        )
        file_handler.context.file_handler.attach_file.return_value = mock_result
        
        # Test attach
        attach_cmd = command_parser.parse_command('/attach("test.txt")')
        result = file_handler.handle_command(attach_cmd)
        assert result.success is True
        
        # Test save
        with patch('builtins.open', mock_open()):
            save_cmd = command_parser.parse_command('/save("output.txt")')
            result = file_handler.handle_command(save_cmd)
            assert result.success is True


class TestErrorHandling:
    """Test error handling in commands."""

    @pytest.fixture
    def handler_with_errors(self, command_context):
        """Create handler that simulates errors."""
        handler = ConversationOperationsHandler(command_context)
        handler.context.system_monitor.update_context_usage.side_effect = Exception("Monitor error")
        return handler

    def test_command_handles_exceptions(self, handler_with_errors, command_parser):
        """Test that commands handle exceptions gracefully."""
        command = command_parser.parse_command("/clear()")
        result = handler_with_errors.handle_command(command)
        
        # Should still succeed even if monitor update fails
        assert "error" in result.message.lower() or result.success is True

    def test_file_permission_errors(self, command_context, command_parser):
        """Test handling of file permission errors."""
        file_handler = FileOperationsHandler(command_context)
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            command = command_parser.parse_command('/save("restricted.txt")')
            result = file_handler.handle_command(command)
            
            assert result.success is False
            assert "permission" in result.message.lower() or "access" in result.message.lower()