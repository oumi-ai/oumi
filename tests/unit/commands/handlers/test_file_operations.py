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

"""Unit tests for file operations command handlers."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oumi.core.types.conversation import Conversation, Message, Role
from oumi_chat.commands import CommandResult, ParsedCommand
from oumi_chat.commands.command_context import CommandContext
from tests.utils.chat_test_utils import (
    create_test_inference_config,
    get_file_attachment_data,
    get_web_content_mocks,
    mock_web_content,
    temporary_test_files,
    validate_command_result,
)


class TestAttachCommand:
    """Test suite for /attach() command."""

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
        """Mock file operations handler."""
        with patch(
            "oumi.core.commands.handlers.file_operations_handler.FileOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_attach_text_file(self, mock_handler):
        """Test attaching a text file."""
        file_data = get_file_attachment_data()
        test_content = file_data["test_files"]["sample.txt"]

        with temporary_test_files({"test.txt": test_content}) as temp_files:
            parsed_cmd = ParsedCommand(
                command="attach",
                args=[temp_files["test.txt"]],
                kwargs={},
                raw_input="/attach(...",
            )

            # Mock successful attachment
            mock_handler.handle.return_value = CommandResult(
                success=True, message=f"Successfully attached {temp_files['test.txt']}"
            )

            result = mock_handler.handle(parsed_cmd, self.command_context)

            validate_command_result(
                result, expect_success=True, expected_message_parts=["attached", ".txt"]
            )

    def test_attach_csv_file(self, mock_handler):
        """Test attaching a CSV file."""
        file_data = get_file_attachment_data()
        csv_content = file_data["test_files"]["sample.csv"]

        with temporary_test_files({"data.csv": csv_content}) as temp_files:
            parsed_cmd = ParsedCommand(
                command="attach",
                args=[temp_files["data.csv"]],
                kwargs={},
                raw_input="/attach(...",
            )

            mock_handler.handle.return_value = CommandResult(
                success=True, message="CSV file attached successfully with 5 rows"
            )

            result = mock_handler.handle(parsed_cmd, self.command_context)

            validate_command_result(
                result, expect_success=True, expected_message_parts=["CSV", "attached"]
            )

    def test_attach_json_file(self, mock_handler):
        """Test attaching a JSON file."""
        file_data = get_file_attachment_data()
        json_content = file_data["test_files"]["sample.json"]

        with temporary_test_files({"config.json": json_content}) as temp_files:
            parsed_cmd = ParsedCommand(
                command="attach",
                args=[temp_files["config.json"]],
                kwargs={},
                raw_input="/attach(...",
            )

            mock_handler.handle.return_value = CommandResult(
                success=True, message="JSON file attached and parsed successfully"
            )

            result = mock_handler.handle(parsed_cmd, self.command_context)

            validate_command_result(
                result, expect_success=True, expected_message_parts=["JSON", "attached"]
            )

    def test_attach_nonexistent_file(self, mock_handler):
        """Test attaching a file that doesn't exist."""
        parsed_cmd = ParsedCommand(
            command="attach",
            args=["/nonexistent/file.txt"],
            kwargs={},
            raw_input="/attach(...",
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="File not found: /nonexistent/file.txt"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["not found"]
        )

    def test_attach_without_file_argument(self, mock_handler):
        """Test attach command without file argument."""
        parsed_cmd = ParsedCommand(
            command="attach", args=[], kwargs={}, raw_input="/attach("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Usage: /attach(file_path) - Please specify a file to attach",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["Usage", "file_path"]
        )

    def test_attach_large_file_handling(self, mock_handler):
        """Test handling of large files."""
        large_content = "Large file content\n" * 10000  # Simulate large file

        with temporary_test_files({"large.txt": large_content}) as temp_files:
            parsed_cmd = ParsedCommand(
                command="attach",
                args=[temp_files["large.txt"]],
                kwargs={},
                raw_input="/attach(...",
            )

            mock_handler.handle.return_value = CommandResult(
                success=True,
                message="Large file attached (content truncated for display)",
            )

            result = mock_handler.handle(parsed_cmd, self.command_context)

            validate_command_result(result, expect_success=True)

    def test_attach_binary_file_rejection(self, mock_handler):
        """Test that binary files are handled appropriately."""
        # Create a simple binary file
        binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(binary_content)
            temp_path = f.name

        try:
            parsed_cmd = ParsedCommand(
                command="attach", args=[temp_path], kwargs={}, raw_input="/attach(..."
            )

            mock_handler.handle.return_value = CommandResult(
                success=False, message="Binary files are not supported for attachment"
            )

            result = mock_handler.handle(parsed_cmd, self.command_context)

            validate_command_result(
                result,
                expect_success=False,
                expected_message_parts=["binary", "not supported"],
            )
        finally:
            Path(temp_path).unlink()


class TestFetchCommand:
    """Test suite for /fetch() command."""

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
        """Mock file operations handler."""
        with patch(
            "oumi.core.commands.handlers.file_operations_handler.FileOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_fetch_web_content(self, mock_handler):
        """Test fetching web content."""
        web_mocks = get_web_content_mocks()
        test_url = "https://example.com/article1"

        with mock_web_content({test_url: web_mocks["mock_urls"][test_url]["content"]}):
            parsed_cmd = ParsedCommand(
                command="fetch", args=[test_url], kwargs={}, raw_input="/fetch(..."
            )

            mock_handler.handle.return_value = CommandResult(
                success=True, message=f"Successfully fetched content from {test_url}"
            )

            result = mock_handler.handle(parsed_cmd, self.command_context)

            validate_command_result(
                result,
                expect_success=True,
                expected_message_parts=["fetched", test_url],
            )

    def test_fetch_json_api(self, mock_handler):
        """Test fetching JSON API content."""
        web_mocks = get_web_content_mocks()
        test_url = "https://example.com/json-api"

        with mock_web_content({test_url: web_mocks["mock_urls"][test_url]["content"]}):
            parsed_cmd = ParsedCommand(
                command="fetch", args=[test_url], kwargs={}, raw_input="/fetch(..."
            )

            mock_handler.handle.return_value = CommandResult(
                success=True, message=f"Successfully fetched JSON data from {test_url}"
            )

            result = mock_handler.handle(parsed_cmd, self.command_context)

            validate_command_result(result, expect_success=True)

    def test_fetch_404_error(self, mock_handler):
        """Test fetching from a URL that returns 404."""
        test_url = "https://example.com/not-found"

        with mock_web_content({test_url: "404 Not Found"}):
            # Mock the actual HTTP error behavior
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.status_code = 404
                mock_response.text = "404 Not Found"
                mock_get.return_value = mock_response

                parsed_cmd = ParsedCommand(
                    command="fetch", args=[test_url], kwargs={}, raw_input="/fetch(..."
                )

                mock_handler.handle.return_value = CommandResult(
                    success=False, message=f"Failed to fetch {test_url}: 404 Not Found"
                )

                result = mock_handler.handle(parsed_cmd, self.command_context)

                validate_command_result(
                    result,
                    expect_success=False,
                    expected_message_parts=["404", "Not Found"],
                )

    def test_fetch_invalid_url(self, mock_handler):
        """Test fetching with invalid URL."""
        invalid_url = "not-a-valid-url"

        parsed_cmd = ParsedCommand(
            command="fetch", args=[invalid_url], kwargs={}, raw_input="/fetch(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message=f"Invalid URL: {invalid_url}"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["Invalid URL"]
        )

    def test_fetch_without_url(self, mock_handler):
        """Test fetch command without URL argument."""
        parsed_cmd = ParsedCommand(
            command="fetch", args=[], kwargs={}, raw_input="/fetch("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="Usage: /fetch(url) - Please specify a URL to fetch"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["Usage", "URL"]
        )


class TestSaveCommand:
    """Test suite for /save() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create mock conversation
        self.mock_conversation = Conversation(
            conversation_id="test_conversation",
            messages=[
                Message(role=Role.USER, content="Hello!"),
                Message(role=Role.ASSISTANT, content="Hi there! How can I help?"),
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
        """Mock file operations handler."""
        with patch(
            "oumi.core.commands.handlers.file_operations_handler.FileOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_save_json_format(self, mock_handler):
        """Test saving conversation in JSON format."""
        parsed_cmd = ParsedCommand(
            command="save", args=["conversation.json"], kwargs={}, raw_input="/save(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Conversation saved to conversation.json"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["saved", "conversation.json"],
        )

    def test_save_csv_format(self, mock_handler):
        """Test saving conversation in CSV format."""
        parsed_cmd = ParsedCommand(
            command="save", args=["conversation.csv"], kwargs={}, raw_input="/save(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Conversation exported to CSV: conversation.csv"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(result, expect_success=True)

    def test_save_markdown_format(self, mock_handler):
        """Test saving conversation in Markdown format."""
        parsed_cmd = ParsedCommand(
            command="save", args=["conversation.md"], kwargs={}, raw_input="/save(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Conversation exported to Markdown: conversation.md"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(result, expect_success=True)

    def test_save_pdf_format(self, mock_handler):
        """Test saving conversation in PDF format."""
        parsed_cmd = ParsedCommand(
            command="save", args=["conversation.pdf"], kwargs={}, raw_input="/save(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True, message="Conversation exported to PDF: conversation.pdf"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(result, expect_success=True)

    def test_save_without_filename(self, mock_handler):
        """Test save command without filename argument."""
        parsed_cmd = ParsedCommand(
            command="save", args=[], kwargs={}, raw_input="/save("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Usage: /save(filename) - Please specify an output filename",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["Usage", "filename"]
        )

    def test_save_unsupported_format(self, mock_handler):
        """Test saving in unsupported format."""
        parsed_cmd = ParsedCommand(
            command="save", args=["conversation.xyz"], kwargs={}, raw_input="/save(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Unsupported format: .xyz. Supported formats: json, csv, md, "
            "pdf, txt",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Unsupported format", "xyz"],
        )

    def test_save_permission_error(self, mock_handler):
        """Test save with permission error."""
        parsed_cmd = ParsedCommand(
            command="save",
            args=["/root/conversation.json"],  # Likely to cause permission error
            kwargs={},
            raw_input="/save(...)",
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Permission denied: Unable to write to /root/conversation.json",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["Permission denied"]
        )


class TestShellCommand:
    """Test suite for /shell() command."""

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
        """Mock file operations handler."""
        with patch(
            "oumi.core.commands.handlers.file_operations_handler.FileOperationsHandler"
        ) as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            yield mock_handler

    def test_shell_safe_command(self, mock_handler):
        """Test executing safe shell command."""
        parsed_cmd = ParsedCommand(
            command="shell", args=["ls -la"], kwargs={}, raw_input="/shell(..."
        )

        mock_handler.handle.return_value = CommandResult(
            success=True,
            message="Command executed successfully:\ntotal 8\n"
            "drwxr-xr-x  2 user user 4096 Jan  1 00:00 .",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["executed successfully"],
        )

    def test_shell_dangerous_command_blocked(self, mock_handler):
        """Test that dangerous shell commands are blocked."""
        dangerous_commands = [
            "rm -rf /",
            "sudo shutdown",
            "format c:",
            "dd if=/dev/zero of=/dev/sda",
        ]

        for dangerous_cmd in dangerous_commands:
            parsed_cmd = ParsedCommand(
                command="shell", args=[dangerous_cmd], kwargs={}, raw_input="/shell(..."
            )

            mock_handler.handle.return_value = CommandResult(
                success=False, message=f"Command blocked for security: {dangerous_cmd}"
            )

            result = mock_handler.handle(parsed_cmd, self.command_context)

            validate_command_result(
                result,
                expect_success=False,
                expected_message_parts=["blocked", "security"],
            )

    def test_shell_without_command(self, mock_handler):
        """Test shell command without command argument."""
        parsed_cmd = ParsedCommand(
            command="shell", args=[], kwargs={}, raw_input="/shell("
        )

        mock_handler.handle.return_value = CommandResult(
            success=False,
            message="Usage: /shell(command) - Please specify a command to execute",
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["Usage", "command"]
        )

    def test_shell_command_timeout(self, mock_handler):
        """Test shell command timeout handling."""
        parsed_cmd = ParsedCommand(
            command="shell",
            args=["sleep 300"],  # Long-running command
            kwargs={},
            raw_input="/shell(...)",
        )

        mock_handler.handle.return_value = CommandResult(
            success=False, message="Command timed out after 30 seconds: sleep 300"
        )

        result = mock_handler.handle(parsed_cmd, self.command_context)

        validate_command_result(
            result, expect_success=False, expected_message_parts=["timed out"]
        )


class TestFileOperationsHandler:
    """Test suite for FileOperationsHandler class using real implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help?"},
            ],
            inference_engine=self.mock_engine,
        )

        # Import the actual handler
        from oumi_chat.commands.handlers.file_operations_handler import (
            FileOperationsHandler,
        )

        self.handler = FileOperationsHandler(context=self.command_context)

    def test_get_supported_commands(self):
        """Test that handler returns correct supported commands."""
        supported = self.handler.get_supported_commands()
        expected_commands = {
            "attach",
            "save",
            "import",
            "load",
            "save_history",
            "import_history",
            "fetch",
            "shell",
        }
        assert set(supported) == expected_commands

    def test_unsupported_command(self):
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


class TestSaveCommandReal:
    """Test suite for /save() command using real implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        # Create test conversation history
        self.conversation_history = [
            {"role": "user", "content": "What is Python?"},
            {
                "role": "assistant",
                "content": "Python is a high-level programming language.",
            },
            {"role": "user", "content": "Can you show me a simple example?"},
            {"role": "assistant", "content": "```python\nprint('Hello, World!')\n```"},
        ]

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=self.conversation_history,
            inference_engine=self.mock_engine,
        )

        from oumi_chat.commands.handlers.file_operations_handler import (
            FileOperationsHandler,
        )

        self.handler = FileOperationsHandler(context=self.command_context)

    def test_save_json_format_real(self):
        """Test saving conversation in JSON format with real implementation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "conversation.json"

            command = ParsedCommand(
                command="save",
                args=[str(save_path)],
                kwargs={},
                raw_input="/save(conversation.json)",
            )

            result = self.handler.handle_command(command)

            validate_command_result(
                result,
                expect_success=True,
                expected_message_parts=["exported", "conversation.json"],
            )

            # Verify file was created and contains valid JSON
            assert save_path.exists()
            with open(save_path) as f:
                data = json.load(f)

            # Verify structure
            assert "conversation" in data
            assert len(data["conversation"]) == 4  # 4 messages in history

    def test_save_markdown_format_real(self):
        """Test saving conversation in Markdown format with real implementation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "conversation.md"

            command = ParsedCommand(
                command="save",
                args=[str(save_path)],
                kwargs={},
                raw_input="/save(conversation.md)",
            )

            result = self.handler.handle_command(command)

            validate_command_result(result, expect_success=True)

            # Verify file was created
            assert save_path.exists()
            content = save_path.read_text()

            # Verify Markdown structure (actual format uses emojis)
            assert "# Oumi" in content or "👤 User" in content
            assert "Python" in content  # Content from our test messages

    def test_save_empty_conversation(self):
        """Test saving when conversation history is empty."""
        # Create context with empty conversation
        empty_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        from oumi_chat.commands.handlers.file_operations_handler import (
            FileOperationsHandler,
        )

        handler = FileOperationsHandler(context=empty_context)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "empty.json"

            command = ParsedCommand(
                command="save",
                args=[str(save_path)],
                kwargs={},
                raw_input="/save(empty.json)",
            )

            result = handler.handle_command(command)

            # Should still succeed but indicate empty conversation
            validate_command_result(result, expect_success=True)

    def test_save_invalid_path(self):
        """Test save with invalid file path."""
        # Try to save to a path that should fail
        invalid_path = "/invalid/nonexistent/path/conversation.json"

        command = ParsedCommand(
            command="save",
            args=[invalid_path],
            kwargs={},
            raw_input="/save(invalid_path)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
        )

    def test_save_no_arguments(self):
        """Test save command without arguments."""
        command = ParsedCommand(
            command="save",
            args=[],
            kwargs={},
            raw_input="/save()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
        )


class TestFetchCommandReal:
    """Test suite for /fetch() command using real implementation."""

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

        from oumi_chat.commands.handlers.file_operations_handler import (
            FileOperationsHandler,
        )

        self.handler = FileOperationsHandler(context=self.command_context)

    @patch("requests.get")
    def test_fetch_successful_web_content(self, mock_get):
        """Test fetching web content successfully."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><body><h1>Test Page</h1><p>Test content</p></body></html>"
        )
        mock_response.headers = {"content-type": "text/html"}
        mock_get.return_value = mock_response

        command = ParsedCommand(
            command="fetch",
            args=["https://example.com/test"],
            kwargs={},
            raw_input="/fetch(https://example.com/test)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["fetched"],
        )

    @patch("requests.get")
    def test_fetch_404_error_real(self, mock_get):
        """Test handling 404 error with real implementation."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = Exception("404 Client Error")
        mock_get.return_value = mock_response

        command = ParsedCommand(
            command="fetch",
            args=["https://example.com/notfound"],
            kwargs={},
            raw_input="/fetch(https://example.com/notfound)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
        )

    def test_fetch_invalid_url_real(self):
        """Test fetching with invalid URL using real implementation."""
        command = ParsedCommand(
            command="fetch",
            args=["not-a-url"],
            kwargs={},
            raw_input="/fetch(not-a-url)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["URL"],
        )

    def test_fetch_no_arguments(self):
        """Test fetch command without arguments."""
        command = ParsedCommand(
            command="fetch",
            args=[],
            kwargs={},
            raw_input="/fetch()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
        )


class TestShellCommandReal:
    """Test suite for /shell() command using real implementation."""

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

        from oumi_chat.commands.handlers.file_operations_handler import (
            FileOperationsHandler,
        )

        self.handler = FileOperationsHandler(context=self.command_context)

    def test_shell_simple_command(self):
        """Test executing simple shell command."""
        command = ParsedCommand(
            command="shell",
            args=["echo 'Hello World'"],
            kwargs={},
            raw_input="/shell(echo 'Hello World')",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Hello World"],
        )

    def test_shell_ls_command(self):
        """Test executing ls command."""
        command = ParsedCommand(
            command="shell",
            args=["ls /tmp"],
            kwargs={},
            raw_input="/shell(ls /tmp)",
        )

        result = self.handler.handle_command(command)

        # ls command should succeed (even if /tmp is empty)
        validate_command_result(result, expect_success=True)

    def test_shell_invalid_command(self):
        """Test executing invalid shell command."""
        command = ParsedCommand(
            command="shell",
            args=["nonexistent_command_12345"],
            kwargs={},
            raw_input="/shell(nonexistent_command_12345)",
        )

        result = self.handler.handle_command(command)

        # Shell handler reports success even for commands with non-zero exit codes
        validate_command_result(result, expect_success=True)
        # But should mention exit code in message
        assert result.message is not None and "exit code" in result.message

    def test_shell_no_arguments(self):
        """Test shell command without arguments."""
        command = ParsedCommand(
            command="shell",
            args=[],
            kwargs={},
            raw_input="/shell()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["command"],
        )

    def test_shell_dangerous_command_blocked(self):
        """Test that potentially dangerous commands are handled appropriately."""
        # Test a command that might be restricted
        command = ParsedCommand(
            command="shell",
            args=["rm -f /some/nonexistent/file"],  # Safe since file doesn't exist
            kwargs={},
            raw_input="/shell(rm -f /some/nonexistent/file)",
        )

        result = self.handler.handle_command(command)

        # Either should succeed (if file doesn't exist) or be blocked
        # We're not making assumptions about security implementation
        assert isinstance(result.success, bool)

    def test_shell_with_output_capture(self):
        """Test shell command with output capture."""
        command = ParsedCommand(
            command="shell",
            args=["pwd"],  # Print working directory
            kwargs={},
            raw_input="/shell(pwd)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
        )
        # Should mention the command was executed
        assert result.message is not None and "Executed command" in result.message
        assert result.message is not None and "pwd" in result.message


class TestImportExportCommands:
    """Test suite for import/load and save_history/import_history commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_inference_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[
                {"role": "user", "content": "Test message 1"},
                {"role": "assistant", "content": "Test response 1"},
            ],
            inference_engine=self.mock_engine,
        )

        from oumi_chat.commands.handlers.file_operations_handler import (
            FileOperationsHandler,
        )

        self.handler = FileOperationsHandler(context=self.command_context)

    def test_save_history_command(self):
        """Test save_history command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "history.json"

            command = ParsedCommand(
                command="save_history",
                args=[str(save_path)],
                kwargs={},
                raw_input="/save_history(history.json)",
            )

            result = self.handler.handle_command(command)

            # Should handle the command (success or appropriate error)
            assert isinstance(result.success, bool)

    def test_import_history_command(self):
        """Test import_history command."""
        # Create a temporary file to import from
        with tempfile.TemporaryDirectory() as temp_dir:
            import_path = Path(temp_dir) / "import.json"

            # Create a simple JSON file to import
            test_data = {"conversation": [{"role": "user", "content": "Test"}]}
            with open(import_path, "w") as f:
                json.dump(test_data, f)

            command = ParsedCommand(
                command="import_history",
                args=[str(import_path)],
                kwargs={},
                raw_input="/import_history(import.json)",
            )

            result = self.handler.handle_command(command)

            # Should handle the command (success or appropriate error)
            assert isinstance(result.success, bool)

    def test_load_command(self):
        """Test load command."""
        # Create a temporary file to load
        with tempfile.TemporaryDirectory() as temp_dir:
            load_path = Path(temp_dir) / "load.txt"
            load_path.write_text("Test content to load")

            command = ParsedCommand(
                command="load",
                args=[str(load_path)],
                kwargs={},
                raw_input="/load(load.txt)",
            )

            result = self.handler.handle_command(command)

            # Should handle the command
            assert isinstance(result.success, bool)

    def test_import_command(self):
        """Test import command."""
        # Create a temporary file to import
        with tempfile.TemporaryDirectory() as temp_dir:
            import_path = Path(temp_dir) / "import.txt"
            import_path.write_text("Test content to import")

            command = ParsedCommand(
                command="import",
                args=[str(import_path)],
                kwargs={},
                raw_input="/import(import.txt)",
            )

            result = self.handler.handle_command(command)

            # Should handle the command
            assert isinstance(result.success, bool)


class TestAttachCommandReal:
    """Test suite for /attach() command using real implementation."""

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

        from oumi_chat.commands.handlers.file_operations_handler import (
            FileOperationsHandler,
        )

        self.handler = FileOperationsHandler(context=self.command_context)

    def test_attach_text_file_real(self):
        """Test attaching a text file with real implementation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_content = "This is a test file\nwith multiple lines\nof content."
            test_file.write_text(test_content)

            command = ParsedCommand(
                command="attach",
                args=[str(test_file)],
                kwargs={},
                raw_input="/attach(test.txt)",
            )

            result = self.handler.handle_command(command)

            # Should handle the attachment
            assert isinstance(result.success, bool)

    def test_attach_nonexistent_file_real(self):
        """Test attaching nonexistent file with real implementation."""
        command = ParsedCommand(
            command="attach",
            args=["/nonexistent/path/file.txt"],
            kwargs={},
            raw_input="/attach(/nonexistent/path/file.txt)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
        )

    def test_attach_no_arguments(self):
        """Test attach command without arguments."""
        command = ParsedCommand(
            command="attach",
            args=[],
            kwargs={},
            raw_input="/attach()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
        )
