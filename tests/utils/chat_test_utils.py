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

"""Comprehensive testing utilities for Oumi chat and webchat features."""

import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import Mock, patch

from oumi.core.commands import CommandParser, CommandResult, CommandRouter
from oumi.core.commands.command_context import CommandContext
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.io_utils import get_oumi_root_directory, load_json


class MockInputHandler:
    """Mock input handler for non-interactive testing."""

    def __init__(self, inputs: list[str]):
        """Initialize with predefined inputs.

        Args:
            inputs: List of input strings to return sequentially.
        """
        self.inputs = iter(inputs)
        self.history: list[str] = []

    def __call__(self, prompt: str = "") -> str:
        """Mock input function that returns next predefined input.

        Args:
            prompt: Input prompt (ignored in mock).

        Returns:
            Next input string.

        Raises:
            EOFError: When all inputs are exhausted.
        """
        try:
            input_str = next(self.inputs)
            self.history.append(input_str)
            return input_str
        except StopIteration:
            raise EOFError("No more mock inputs available")


class MockOutputCapture:
    """Captures output for testing."""

    def __init__(self):
        self.outputs: list[str] = []
        self.console_output: list[str] = []

    def write(self, text: str) -> None:
        """Capture written text."""
        self.outputs.append(text)

    def print(self, *args, **kwargs) -> None:
        """Capture printed text."""
        text = " ".join(str(arg) for arg in args)
        self.console_output.append(text)


class ChatTestSession:
    """Test session for interactive chat functionality."""

    def __init__(
        self,
        config: InferenceConfig,
        mock_inputs: Optional[list[str]] = None,
        capture_output: bool = True,
    ):
        """Initialize chat test session.

        Args:
            config: Inference configuration for the session.
            mock_inputs: Predefined inputs for testing.
            capture_output: Whether to capture output for validation.
        """
        self.config = config
        self.mock_inputs = mock_inputs or []
        self.capture_output = capture_output

        # Test state
        self.conversation_history: list[Conversation] = []
        self.command_history: list[tuple[str, CommandResult]] = []
        self.output_capture = MockOutputCapture() if capture_output else None
        self.input_handler = MockInputHandler(mock_inputs) if mock_inputs else None

        # Mock objects for testing - initialize immediately for direct access
        self.mock_engine: Mock = Mock(spec=BaseInferenceEngine)
        self.mock_console: Mock = Mock()
        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        # Session state
        self._session_active: bool = False
        self._current_conversation: Optional[Conversation] = None

    @contextmanager
    def mock_interactive_session(self):
        """Context manager for mocked interactive session."""
        patches = []

        try:
            # Mock input function
            if self.input_handler:
                input_patch = patch("builtins.input", self.input_handler)
                patches.append(input_patch)
                input_patch.start()

            # Mock console output
            if self.capture_output:
                console_patch = patch("rich.console.Console")
                patches.append(console_patch)
                mock_console = console_patch.start()
                if mock_console.return_value is not None:
                    mock_console.return_value.print.side_effect = (
                        self.output_capture.print
                    )
                # Update the existing mock console with capture behavior
                if (
                    self.mock_console is not None
                    and hasattr(self.mock_console, "print")
                ):
                    self.mock_console.print.side_effect = (
                        self.output_capture.print
                    )

            # Mock inference engine builder to return our existing mock
            engine_patch = patch(
                "oumi.builders.inference_engines.build_inference_engine"
            )
            patches.append(engine_patch)
            mock_engine_builder = engine_patch.start()
            mock_engine_builder.return_value = self.mock_engine

            yield self

        finally:
            # Clean up patches
            for patch_obj in reversed(patches):
                patch_obj.stop()

    def inject_command(self, command: str) -> CommandResult:
        """Inject a command into the chat session.

        Args:
            command: Command string to execute.

        Returns:
            Result of command execution.
        """
        # Sync current conversation to command context before executing commands
        self._sync_to_command_context()

        parser = CommandParser()
        router = CommandRouter(self.command_context)

        # Parse the command
        parsed_command = parser.parse_command(command)
        if parsed_command is None:
            result = CommandResult(
                success=False, message=f"Failed to parse command: {command}"
            )
        else:
            # Execute the command
            result = router.handle_command(parsed_command)

        # Sync back from command context after command execution
        self._sync_from_command_context()

        # Record in history
        self.command_history.append((command, result))
        return result

    def inject_user_message(self, message: str) -> Message:
        """Inject a user message into the conversation.

        Args:
            message: User message text.

        Returns:
            Created user message.
        """
        user_msg = Message(role=Role.USER, content=message)
        if self.conversation_history:
            self.conversation_history[-1].messages.append(user_msg)
        else:
            conv = Conversation(messages=[user_msg])
            self.conversation_history.append(conv)
        return user_msg

    def inject_assistant_response(self, response: str) -> Message:
        """Inject an assistant response into the conversation.

        Args:
            response: Assistant response text.

        Returns:
            Created assistant message.
        """
        assistant_msg = Message(role=Role.ASSISTANT, content=response)
        if self.conversation_history:
            self.conversation_history[-1].messages.append(assistant_msg)
        else:
            conv = Conversation(messages=[assistant_msg])
            self.conversation_history.append(conv)
        return assistant_msg

    def get_last_conversation(self) -> Optional[Conversation]:
        """Get the most recent conversation."""
        return self.conversation_history[-1] if self.conversation_history else None

    def get_command_results(self) -> list[tuple[str, CommandResult]]:
        """Get all command execution results."""
        return self.command_history.copy()

    def assert_command_success(
        self, command: str, expected_message: Optional[str] = None
    ):
        """Assert that a command executed successfully.

        Args:
            command: Command that was executed.
            expected_message: Expected success message (optional).

        Raises:
            AssertionError: If command was not successful.
        """
        for cmd, result in self.command_history:
            if cmd == command:
                assert result.success, f"Command '{command}' failed: {result.message}"
                if expected_message and result.message:
                    assert expected_message in result.message, (
                        f"Expected message '{expected_message}' not found in "
                        f"'{result.message}'"
                    )
                return

        raise AssertionError(f"Command '{command}' was not executed")

    def assert_command_failure(
        self, command: str, expected_error: Optional[str] = None
    ):
        """Assert that a command failed as expected.

        Args:
            command: Command that was executed.
            expected_error: Expected error message (optional).

        Raises:
            AssertionError: If command was successful.
        """
        for cmd, result in self.command_history:
            if cmd == command:
                assert not result.success, (
                    f"Command '{command}' unexpectedly succeeded: {result.message}"
                )
                if expected_error and result.message:
                    assert expected_error in result.message, (
                        f"Expected error '{expected_error}' not found in "
                        f"'{result.message}'"
                    )
                return

        raise AssertionError(f"Command '{command}' was not executed")

    def assert_conversation_length(self, expected_length: int):
        """Assert that conversation has expected number of messages.

        Args:
            expected_length: Expected number of messages.
        """
        if not self.conversation_history:
            assert expected_length == 0, "No conversation history found"
            return

        last_conv = self.conversation_history[-1]
        actual_length = len(last_conv.messages)
        assert actual_length == expected_length, (
            f"Expected {expected_length} messages, got {actual_length}"
        )

    def assert_last_message_role(self, expected_role: Role):
        """Assert that the last message has the expected role.

        Args:
            expected_role: Expected message role.
        """
        assert self.conversation_history, "No conversation history found"
        last_conv = self.conversation_history[-1]
        assert last_conv.messages, "No messages in conversation"

        last_msg = last_conv.messages[-1]
        assert last_msg.role == expected_role, (
            f"Expected role {expected_role}, got {last_msg.role}"
        )

    def start_session(self) -> CommandResult:
        """Start a new chat session.

        Returns:
            Command result indicating session start status.
        """
        if self._session_active:
            return CommandResult(
                success=False,
                message="Session is already active. End current session first.",
            )

        self._session_active = True
        # Use unique session ID based on object id and time to ensure uniqueness
        # across sessions
        import time

        session_id = f"test_session_{id(self)}_{int(time.time() * 1000000)}"
        self._current_conversation = Conversation(
            conversation_id=session_id, messages=[]
        )

        return CommandResult(success=True, message="Chat session started successfully")

    def end_session(self) -> CommandResult:
        """End the current chat session.

        Returns:
            Command result indicating session end status.
        """
        if not self._session_active:
            return CommandResult(success=False, message="No active session to end")

        if self._current_conversation and self._current_conversation.messages:
            self.conversation_history.append(self._current_conversation)

        self._session_active = False
        self._current_conversation = None

        return CommandResult(success=True, message="Chat session ended successfully")

    def is_active(self) -> bool:
        """Check if the chat session is currently active.

        Returns:
            True if session is active, False otherwise.
        """
        return self._session_active

    def send_message(self, message: str) -> CommandResult:
        """Send a message in the chat session and get response.

        Args:
            message: User message to send.

        Returns:
            Command result with assistant response.
        """
        if not self._session_active:
            return CommandResult(
                success=False, message="No active session. Start a session first."
            )

        if not self._current_conversation:
            # Use unique conversation ID based on object id and time
            import time

            conv_id = f"conversation_{id(self)}_{int(time.time() * 1000000)}"
            self._current_conversation = Conversation(
                conversation_id=conv_id, messages=[]
            )

        # Add user message
        user_message = Message(role=Role.USER, content=message)
        self._current_conversation.messages.append(user_message)

        # Generate mock assistant response
        assistant_response = f"Mock assistant response to: {message[:50]}..."
        assistant_message = Message(role=Role.ASSISTANT, content=assistant_response)
        self._current_conversation.messages.append(assistant_message)

        return CommandResult(success=True, message=assistant_response)

    def execute_command(self, command: str) -> CommandResult:
        """Execute a chat command.

        Args:
            command: Command string to execute.

        Returns:
            Command result from execution.
        """
        # Delegate to existing inject_command method
        return self.inject_command(command)

    def _sync_to_command_context(self):
        """Sync current conversation to command context for branch operations."""
        if self._current_conversation and self._current_conversation.messages:
            # Convert conversation messages to the format expected by command context
            context_messages = []
            for msg in self._current_conversation.messages:
                context_messages.append(
                    {"role": msg.role.value.lower(), "content": msg.content}
                )
            self.command_context.conversation_history.clear()
            self.command_context.conversation_history.extend(context_messages)
        else:
            # If no current conversation, make sure command context is empty
            self.command_context.conversation_history.clear()

    def _sync_from_command_context(self):
        """Sync command context back to current conversation after branch operations."""
        if self.command_context.conversation_history and self._current_conversation:
            # Convert context messages back to conversation format
            new_messages = []
            from oumi.core.types.conversation import Role

            for msg in self.command_context.conversation_history:
                role_str = msg.get("role", "user")
                if role_str.lower() == "assistant":
                    role = Role.ASSISTANT
                else:
                    role = Role.USER
                new_messages.append(Message(role=role, content=msg.get("content", "")))

            self._current_conversation.messages = new_messages

    def get_conversation(self) -> Optional[Conversation]:
        """Get the current conversation.

        Returns:
            Current conversation if active, None otherwise.
        """
        return self._current_conversation

    def clear_history(self):
        """Clear all conversation and command history."""
        self.conversation_history.clear()
        self.command_history.clear()
        self._session_active = False
        self._current_conversation = None
        if self.output_capture:
            self.output_capture.outputs.clear()
            self.output_capture.console_output.clear()


def create_test_inference_config(
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct", **kwargs
) -> InferenceConfig:
    """Create optimized inference config for chat testing.

    Args:
        model_name: Model to use for testing.
        **kwargs: Additional config overrides.

    Returns:
        Inference configuration optimized for testing.
    """
    # Default optimized parameters for testing
    defaults = {
        "model": ModelParams(
            model_name=model_name,
            model_max_length=512,  # Small context for fast testing
            torch_dtype_str="float16",
            trust_remote_code=True,
            device_map="auto",
        ),
        "generation": GenerationParams(
            max_new_tokens=50,  # Short responses for fast testing
            temperature=0.0,  # Deterministic for testing
            seed=42,  # Reproducible results
        ),
    }

    # Merge with provided overrides
    defaults.update(kwargs)
    return InferenceConfig(**defaults)


def create_vision_test_config(**kwargs) -> InferenceConfig:
    """Create vision model config for multimodal testing.

    Args:
        **kwargs: Additional config overrides.

    Returns:
        Vision-optimized inference configuration.
    """
    return create_test_inference_config(
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct", **kwargs
    )


@contextmanager
def temporary_test_files(file_contents: dict[str, str]):
    """Create temporary files for testing file operations.

    Args:
        file_contents: Mapping of filename to content.

    Yields:
        Dictionary mapping filenames to temporary file paths.
    """
    temp_files = {}
    temp_paths = []

    try:
        for filename, content in file_contents.items():
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=Path(filename).suffix, delete=False
            ) as f:
                f.write(content)
                temp_files[filename] = f.name
                temp_paths.append(f.name)

        yield temp_files

    finally:
        # Clean up temporary files
        for temp_path in temp_paths:
            try:
                Path(temp_path).unlink()
            except FileNotFoundError:
                pass


def mock_web_content(urls_to_content: dict[str, str]):
    """Mock web content for /fetch() command testing.

    Args:
        urls_to_content: Mapping of URLs to their content.

    Returns:
        Mock patch object for requests.get.
    """

    def mock_get(url, **kwargs):
        mock_response = Mock()
        if url in urls_to_content:
            mock_response.text = urls_to_content[url]
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
        else:
            mock_response.status_code = 404
            mock_response.text = "Not Found"
        return mock_response

    return patch("requests.get", side_effect=mock_get)


def assert_valid_conversation_export(
    export_path: Union[str, Path], expected_format: str, min_messages: int = 1
):
    """Assert that a conversation export file is valid.

    Args:
        export_path: Path to the exported file.
        expected_format: Expected file format ('json', 'csv', 'md', etc.).
        min_messages: Minimum expected number of messages.
    """
    export_path = Path(export_path)
    assert export_path.exists(), f"Export file does not exist: {export_path}"
    assert export_path.stat().st_size > 0, f"Export file is empty: {export_path}"

    if expected_format.lower() == "json":
        with open(export_path) as f:
            data = json.load(f)
            assert isinstance(data, (dict, list)), "Invalid JSON structure"

    elif expected_format.lower() == "csv":
        import csv

        with open(export_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) >= min_messages + 1, (
                "Not enough CSV rows (including header)"
            )

    elif expected_format.lower() == "md":
        content = export_path.read_text()
        assert len(content.strip()) > 0, "Markdown export is empty"
        assert "**User:**" in content or "**Assistant:**" in content, (
            "Missing conversation markers"
        )


def create_test_image_bytes() -> bytes:
    """Create minimal test image data for vision testing.

    Returns:
        PNG image data as bytes.
    """
    # Create a minimal 1x1 PNG image
    import base64

    # 1x1 red pixel PNG encoded in base64
    png_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAW"
        "jR9awAAAABJRU5ErkJggg=="
    )
    return base64.b64decode(png_data)


class TestFileCleanupManager:
    """Context manager for ensuring test files are cleaned up."""

    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up all registered files and directories."""
        self.cleanup_all()

    def create_temp_file(
        self, suffix: str = "", content: str = "", mode: str = "w"
    ) -> str:
        """Create a temporary file and register it for cleanup.

        Args:
            suffix: File suffix/extension.
            content: Initial file content.
            mode: File open mode.

        Returns:
            Path to the created temporary file.
        """
        temp_file = tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False)
        if content:
            temp_file.write(content)
        temp_file.close()

        self.temp_files.append(temp_file.name)
        return temp_file.name

    def create_temp_dir(self) -> str:
        """Create a temporary directory and register it for cleanup.

        Returns:
            Path to the created temporary directory.
        """
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def register_file(self, file_path: str):
        """Register an existing file for cleanup.

        Args:
            file_path: Path to file to be cleaned up.
        """
        if file_path not in self.temp_files:
            self.temp_files.append(file_path)

    def register_dir(self, dir_path: str):
        """Register an existing directory for cleanup.

        Args:
            dir_path: Path to directory to be cleaned up.
        """
        if dir_path not in self.temp_dirs:
            self.temp_dirs.append(dir_path)

    def cleanup_all(self):
        """Clean up all registered files and directories."""
        # Clean up files first
        for file_path in self.temp_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors

        # Clean up directories
        for dir_path in self.temp_dirs:
            try:
                import shutil

                shutil.rmtree(dir_path, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors

        # Clear the lists
        self.temp_files.clear()
        self.temp_dirs.clear()


def ensure_test_cleanup():
    """Decorator to ensure test cleanup even if tests fail.

    Usage:
        @ensure_test_cleanup()
        def test_function():
            # Test code here
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with TestFileCleanupManager():
                return func(*args, **kwargs)

        return wrapper

    return decorator


def cleanup_test_files_in_directory(
    directory: Union[str, Path], patterns: Optional[list[str]] = None
):
    """Clean up test files in a specific directory.

    Args:
        directory: Directory to clean.
        patterns: List of glob patterns to match. If None, uses default test patterns.
    """
    if patterns is None:
        patterns = [
            "test_*.json",
            "test_*.txt",
            "test_*.pdf",
            "test_*.csv",
            "test_*.md",
            "*_test_*",
            "stress_test_*",
            "analysis_report*",
            "project_analysis*",
            "*_attachment*",
            "*_cleanup_test_*",
            "deeply_nested*",
            "sales_data*",
            "*_report*",
            "file1.json",
            "file2.json",
            "output.json",
            "file.txt",
            "test.json",
            "refinement_*.md",
            "demo.cast",
            "'mixed\"",
            '"unclosed',
        ]

    directory = Path(directory)
    if not directory.exists():
        return

    for pattern in patterns:
        for file_path in directory.glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors


class CommandSequenceBuilder:
    """Builder for creating complex command sequences for testing."""

    def __init__(self, session: ChatTestSession):
        self.session = session
        self.commands: list[str] = []

    def add_command(self, command: str) -> "CommandSequenceBuilder":
        """Add a command to the sequence."""
        self.commands.append(command)
        return self

    def add_user_input(self, message: str) -> "CommandSequenceBuilder":
        """Add user input between commands."""
        # This will be handled by the session during execution
        self.commands.append(f"__USER_INPUT__{message}")
        return self

    def execute_sequence(self) -> list[CommandResult]:
        """Execute the built command sequence."""
        results = []

        for command in self.commands:
            if command.startswith("__USER_INPUT__"):
                message = command.replace("__USER_INPUT__", "")
                self.session.inject_user_message(message)
                # Simulate assistant response
                self.session.inject_assistant_response("Test response")
            else:
                result = self.session.inject_command(command)
                results.append(result)

        return results

    def assert_all_successful(self):
        """Assert that all commands in the sequence were successful."""
        results = self.execute_sequence()
        for i, result in enumerate(results):
            assert result.success, (
                f"Command {i + 1} failed: {self.commands[i]} - {result.message}"
            )


def load_chat_test_data(filename: str) -> dict[str, Any]:
    """Load test data from the chat test data directory.

    Args:
        filename: Name of the test data file.

    Returns:
        Loaded test data as dictionary.
    """
    # Get the project root directory (go up from src/oumi to the project root)
    project_root = get_oumi_root_directory().parent.parent
    data_path = project_root / "tests" / "testdata" / "chat" / filename
    return load_json(data_path)


def get_sample_conversations() -> list[dict[str, Any]]:
    """Load sample conversations for testing.

    Returns:
        List of sample conversation data.
    """
    data = load_chat_test_data("sample_conversations.json")
    if isinstance(data, list):
        return data
    # Handle case where data is wrapped in a dict
    return data.get("conversations", [data]) if isinstance(data, dict) else []


def get_web_content_mocks() -> dict[str, Any]:
    """Load mock web content for /fetch() testing.

    Returns:
        Dictionary of mock URL responses.
    """
    return load_chat_test_data("web_content_mocks.json")


def get_file_attachment_data() -> dict[str, Any]:
    """Load test file content for /attach() testing.

    Returns:
        Dictionary of test file contents.
    """
    return load_chat_test_data("file_attachments.json")


def get_test_macro_template() -> str:
    """Load test macro template.

    Returns:
        Jinja template content for macro testing.
    """
    template_path = (
        get_oumi_root_directory().parent.parent
        / "tests"
        / "testdata"
        / "chat"
        / "test_macro.jinja"
    )
    return template_path.read_text()


# Utility functions for common test patterns
def get_chat_test_models() -> dict[str, ModelParams]:
    """Get model configurations optimized for chat testing.

    Returns:
        Dictionary of model configurations for testing.
    """
    return {
        "text_chat": ModelParams(
            model_name="HuggingFaceTB/SmolLM-135M-Instruct",
            model_max_length=512,
            torch_dtype_str="float16",
            trust_remote_code=True,
        ),
        "vision_chat": ModelParams(
            model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            model_max_length=512,
            torch_dtype_str="float16",
            trust_remote_code=True,
        ),
        "cpu_chat": ModelParams(
            model_name="unsloth/SmolLM-135M-Instruct-GGUF",
            model_kwargs={"filename": "SmolLM-135M-Instruct-Q4_K_M.gguf"},
            model_max_length=512,
            torch_dtype_str="float16",
        ),
    }


def validate_command_result(
    result: CommandResult,
    expect_success: bool = True,
    expected_message_parts: Optional[list[str]] = None,
    unexpected_message_parts: Optional[list[str]] = None,
):
    """Validate a command result against expectations.

    Args:
        result: Command result to validate.
        expect_success: Whether command should have succeeded.
        expected_message_parts: Parts that should be in the result message.
        unexpected_message_parts: Parts that should NOT be in the result message.
    """
    assert result.success == expect_success, (
        f"Expected success={expect_success}, got success={result.success}. "
        f"Message: {result.message}"
    )

    if expected_message_parts:
        for part in expected_message_parts:
            message = result.message or ""
            assert part.lower() in message.lower(), (
                f"Expected message part '{part}' not found in: {result.message}"
            )

    if unexpected_message_parts:
        for part in unexpected_message_parts:
            message = result.message or ""
            assert part.lower() not in message.lower(), (
                f"Unexpected message part '{part}' found in: {result.message}"
            )


# Performance testing utilities
def measure_command_performance(session: ChatTestSession, command: str) -> float:
    """Measure command execution time.

    Args:
        session: Chat test session.
        command: Command to measure.

    Returns:
        Execution time in seconds.
    """
    start_time = time.time()
    session.inject_command(command)
    return time.time() - start_time
