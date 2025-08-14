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

"""Base class for command handlers."""

from abc import ABC, abstractmethod
from typing import Optional

from oumi.core.commands.command_context import CommandContext
from oumi.core.commands.command_parser import ParsedCommand


class CommandResult:
    """Result of executing a command.

    Attributes:
        success: Whether the command executed successfully.
        message: Optional message to display to the user.
        should_exit: Whether the chat session should exit.
        should_continue: Whether to continue processing (vs. skip inference).
        user_input_override: Override input text for inference.
        is_regeneration: Whether this is a regen operation (skip history addition).
    """

    def __init__(
        self,
        success: bool = True,
        message: Optional[str] = None,
        should_exit: bool = False,
        should_continue: bool = True,
        user_input_override: Optional[str] = None,
        is_regeneration: bool = False,
    ):
        """Initialize CommandResult.

        Args:
            success: Whether the command executed successfully.
            message: Optional message to display to the user.
            should_exit: Whether the chat session should exit.
            should_continue: Whether to continue processing (vs. skip inference).
            user_input_override: Override input text for inference.
            is_regeneration: Whether this is a regen operation (skip history addition).
        """
        self.success = success
        self.message = message
        self.should_exit = should_exit
        self.should_continue = should_continue
        self.user_input_override = user_input_override
        self.is_regeneration = is_regeneration


class BaseCommandHandler(ABC):
    """Abstract base class for command handlers.

    All command handlers should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, context: CommandContext):
        """Initialize the handler with shared context.

        Args:
            context: Shared command context containing dependencies.
        """
        self.context = context
        self.console = context.console
        self.config = context.config
        self.conversation_history = context.conversation_history
        self.inference_engine = context.inference_engine
        self.system_monitor = context.system_monitor
        self._style = context._style

    @abstractmethod
    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports.

        Returns:
            List of command names (without the / prefix).
        """
        pass

    @abstractmethod
    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a parsed command.

        Args:
            command: The parsed command to handle.

        Returns:
            CommandResult indicating the outcome.
        """
        pass

    def display_command_error(self, error_message: str):
        """Display a command error message with styling."""
        error_style = getattr(self._style, "error_style", "red")
        self.console.print(f"❌ {error_message}", style=error_style)

    def display_command_success(self, message: str):
        """Display a command success message with styling."""
        success_style = getattr(self._style, "success_style", "green")
        use_emoji = getattr(self._style, "use_emoji", True)

        if use_emoji:
            self.console.print(f"✅ {message}", style=success_style)
        else:
            self.console.print(f"Success: {message}", style=success_style)

    def _get_conversation_tokens(self) -> int:
        """Get accurate token count for current conversation using context manager.

        Returns:
            Accurate token count using tiktoken-based estimation.
        """
        if hasattr(self.context, 'context_window_manager'):
            conversation_text = ""
            for msg in self.conversation_history:
                if isinstance(msg, dict):
                    # Handle regular messages
                    if "content" in msg:
                        conversation_text += str(msg["content"]) + "\n"
                    # Handle attachment messages
                    elif msg.get("role") == "attachment" and "text_content" in msg:
                        conversation_text += str(msg["text_content"]) + "\n"
                    elif msg.get("role") == "attachment" and "content" in msg:
                        conversation_text += str(msg["content"]) + "\n"
            
            return self.context.context_window_manager.estimate_tokens(conversation_text)
        
        # Fallback to character-based estimation if context manager not available
        total_chars = 0
        for msg in self.conversation_history:
            if msg.get("role") == "attachment":
                content = msg.get("text_content", "") or msg.get("content", "")
            else:
                content = msg.get("content", "")
            total_chars += len(str(content))

        # Rough estimation: ~4 chars per token
        return total_chars // 4

    def _update_context_in_monitor(self):
        """Update context usage in system monitor if available."""
        if self.system_monitor and hasattr(self.system_monitor, "update_context_usage"):
            estimated_tokens = self._get_conversation_tokens()
            max_context = getattr(self.config.model, "model_max_length", 4096)
            self.system_monitor.update_context_usage(estimated_tokens)
            if hasattr(self.system_monitor, "update_max_context_tokens"):
                self.system_monitor.update_max_context_tokens(max_context)

        # Auto-save after context updates (conversation modifications)
        self._auto_save_if_enabled()

    def _auto_save_if_enabled(self):
        """Auto-save chat if enabled and available."""
        try:
            # Access the file operations handler via context
            if (
                hasattr(self.context, "_command_router")
                and self.context._command_router
            ):
                file_operations_handler = self.context._command_router._handlers.get(
                    "file_operations"
                )
                if file_operations_handler and hasattr(
                    file_operations_handler, "auto_save_chat"
                ):
                    file_operations_handler.auto_save_chat()
        except Exception:
            # Silently fail auto-save to avoid interrupting user experience
            pass
