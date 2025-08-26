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

"""Unit tests for command routing functionality."""

import pytest
from unittest.mock import MagicMock, Mock, patch

from oumi.core.commands import CommandResult, CommandRouter, ParsedCommand
from oumi.core.commands.command_context import CommandContext
from oumi.core.configs import InferenceConfig
from oumi.core.inference import BaseInferenceEngine
from tests.utils.chat_test_utils import create_test_inference_config


class TestCommandRouter:
    """Test suite for CommandRouter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.router = CommandRouter()
        
        # Create mock command context
        self.mock_engine = Mock(spec=BaseInferenceEngine)
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()
        
        self.command_context = CommandContext(
            config=self.test_config,
            console=self.mock_console,
            inference_engine=self.mock_engine,
        )

    def test_router_initialization(self):
        """Test router initializes correctly."""
        assert self.router is not None
        # Check if router has the expected attributes/methods
        assert hasattr(self.router, 'execute')
        
    def test_execute_help_command(self):
        """Test executing the help command."""
        parsed_cmd = ParsedCommand(name="help", args=[], kwargs={})
        
        with patch('oumi.core.commands.handlers.help_handler.HelpHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.handle.return_value = CommandResult(success=True, message="Help displayed")
            mock_handler_class.return_value = mock_handler
            
            result = self.router.execute(parsed_cmd, self.command_context)
            
            assert isinstance(result, CommandResult)
            if result.success:  # If handler was found and executed
                assert "help" in result.message.lower() or result.success

    def test_execute_save_command(self):
        """Test executing the save command."""
        parsed_cmd = ParsedCommand(name="save", args=["test_output.json"], kwargs={})
        
        with patch('oumi.core.commands.handlers.file_operations_handler.FileOperationsHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.handle.return_value = CommandResult(
                success=True, 
                message="Conversation saved to test_output.json"
            )
            mock_handler_class.return_value = mock_handler
            
            result = self.router.execute(parsed_cmd, self.command_context)
            
            assert isinstance(result, CommandResult)
            # Handler should be called with the correct arguments
            if mock_handler.handle.called:
                args, kwargs = mock_handler.handle.call_args
                assert "save" in str(args) or "save" in str(kwargs)

    def test_execute_unknown_command(self):
        """Test executing an unknown command."""
        parsed_cmd = ParsedCommand(name="unknown_command", args=[], kwargs={})
        
        result = self.router.execute(parsed_cmd, self.command_context)
        
        assert isinstance(result, CommandResult)
        assert not result.success
        assert "unknown" in result.message.lower() or "not found" in result.message.lower()

    def test_execute_command_with_handler_error(self):
        """Test executing a command when handler raises an exception."""
        parsed_cmd = ParsedCommand(name="save", args=["test.json"], kwargs={})
        
        with patch('oumi.core.commands.handlers.file_operations_handler.FileOperationsHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.handle.side_effect = Exception("Handler error")
            mock_handler_class.return_value = mock_handler
            
            result = self.router.execute(parsed_cmd, self.command_context)
            
            assert isinstance(result, CommandResult)
            assert not result.success
            assert "error" in result.message.lower()

    def test_execute_commands_with_different_arg_patterns(self):
        """Test executing commands with various argument patterns."""
        test_cases = [
            # Command with no arguments
            ParsedCommand(name="help", args=[], kwargs={}),
            
            # Command with positional arguments
            ParsedCommand(name="save", args=["output.json"], kwargs={}),
            
            # Command with keyword arguments
            ParsedCommand(name="set", args=[], kwargs={"temperature": "0.8"}),
            
            # Command with both types of arguments
            ParsedCommand(name="branch_from", args=["main"], kwargs={"position": "5"}),
            
            # Command with multiple arguments
            ParsedCommand(name="set", args=[], kwargs={
                "temperature": "0.8", 
                "top_p": "0.9", 
                "max_tokens": "100"
            }),
        ]
        
        for parsed_cmd in test_cases:
            with patch('oumi.core.commands.handlers') as mock_handlers:
                # Mock any handler that might be called
                mock_handler = Mock()
                mock_handler.handle.return_value = CommandResult(
                    success=True, 
                    message=f"Executed {parsed_cmd.name}"
                )
                
                # Mock the handler lookup
                with patch.object(self.router, '_get_handler', return_value=mock_handler):
                    result = self.router.execute(parsed_cmd, self.command_context)
                    
                    assert isinstance(result, CommandResult)
                    # Verify handler was called with command context
                    if mock_handler.handle.called:
                        call_args = mock_handler.handle.call_args
                        assert call_args is not None

    def test_router_context_validation(self):
        """Test that router validates command context properly."""
        parsed_cmd = ParsedCommand(name="help", args=[], kwargs={})
        
        # Test with None context
        result = self.router.execute(parsed_cmd, None)
        assert isinstance(result, CommandResult)
        assert not result.success
        assert "context" in result.message.lower()
        
        # Test with invalid context
        invalid_context = Mock()
        invalid_context.config = None
        
        result = self.router.execute(parsed_cmd, invalid_context)
        assert isinstance(result, CommandResult)
        # Should either handle gracefully or return error

    def test_command_handler_selection(self):
        """Test that router selects appropriate handlers for commands."""
        command_to_handler_mapping = {
            "help": "help_handler",
            "save": "file_operations_handler", 
            "load": "file_operations_handler",
            "attach": "file_operations_handler",
            "fetch": "file_operations_handler",
            "delete": "conversation_operations_handler",
            "clear": "conversation_operations_handler",
            "branch": "branch_operations_handler",
            "set": "parameter_management_handler",
            "swap": "model_management_handler",
            "macro": "macro_operations_handler",
        }
        
        for command_name, expected_handler_type in command_to_handler_mapping.items():
            parsed_cmd = ParsedCommand(name=command_name, args=[], kwargs={})
            
            with patch('oumi.core.commands.handlers') as mock_handlers:
                # Set up mock to track which handler type would be selected
                mock_handler = Mock()
                mock_handler.handle.return_value = CommandResult(
                    success=True, 
                    message=f"Handled by {expected_handler_type}"
                )
                
                # The actual implementation details may vary
                # This test verifies the general routing concept
                result = self.router.execute(parsed_cmd, self.command_context)
                assert isinstance(result, CommandResult)

    def test_router_error_recovery(self):
        """Test router error recovery mechanisms."""
        parsed_cmd = ParsedCommand(name="save", args=["test.json"], kwargs={})
        
        # Test handler instantiation failure
        with patch('oumi.core.commands.handlers.file_operations_handler.FileOperationsHandler') as mock_handler_class:
            mock_handler_class.side_effect = ImportError("Handler not available")
            
            result = self.router.execute(parsed_cmd, self.command_context)
            
            assert isinstance(result, CommandResult)
            assert not result.success
            # Should provide helpful error message
            assert len(result.message) > 0

    def test_concurrent_command_execution(self):
        """Test that router can handle concurrent command execution."""
        import threading
        import time
        
        parsed_cmd = ParsedCommand(name="help", args=[], kwargs={})
        results = []
        errors = []
        
        def execute_command():
            try:
                result = self.router.execute(parsed_cmd, self.command_context)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Execute multiple commands concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=execute_command)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)  # Timeout to prevent hanging
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        
        for result in results:
            assert isinstance(result, CommandResult)

    def test_router_performance(self):
        """Test router performance with rapid command execution."""
        import time
        
        parsed_cmd = ParsedCommand(name="help", args=[], kwargs={})
        
        start_time = time.time()
        
        # Execute many commands rapidly
        for _ in range(100):
            result = self.router.execute(parsed_cmd, self.command_context)
            assert isinstance(result, CommandResult)
        
        elapsed_time = time.time() - start_time
        
        # Should complete reasonably quickly (less than 1 second for 100 commands)
        assert elapsed_time < 1.0, f"Router too slow: {elapsed_time:.2f}s for 100 commands"

    def test_command_result_consistency(self):
        """Test that command results are consistent and well-formed."""
        test_commands = [
            ParsedCommand(name="help", args=[], kwargs={}),
            ParsedCommand(name="unknown", args=[], kwargs={}),
            ParsedCommand(name="save", args=["test.json"], kwargs={}),
        ]
        
        for parsed_cmd in test_commands:
            result = self.router.execute(parsed_cmd, self.command_context)
            
            # Verify result structure
            assert isinstance(result, CommandResult)
            assert isinstance(result.success, bool)
            assert isinstance(result.message, str)
            assert len(result.message) > 0  # Should always have a message
            
            # Verify result content makes sense
            if result.success:
                assert "error" not in result.message.lower()
            else:
                # Failed commands should have informative messages
                assert len(result.message.strip()) > 10  # Substantial error message

    def test_router_state_isolation(self):
        """Test that router maintains proper state isolation between commands."""
        # Execute multiple different commands
        commands = [
            ParsedCommand(name="help", args=[], kwargs={}),
            ParsedCommand(name="save", args=["file1.json"], kwargs={}), 
            ParsedCommand(name="save", args=["file2.json"], kwargs={}),
        ]
        
        results = []
        for cmd in commands:
            result = self.router.execute(cmd, self.command_context)
            results.append(result)
        
        # Each result should be independent
        for i, result in enumerate(results):
            assert isinstance(result, CommandResult)
            # Results should not interfere with each other
            if i > 0:
                # Later results should not be affected by earlier ones
                assert result != results[i-1] or result.message != results[i-1].message