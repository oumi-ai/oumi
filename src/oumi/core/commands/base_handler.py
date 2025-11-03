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
        if hasattr(self.context, "context_window_manager"):
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

            return self.context.context_window_manager.estimate_tokens(
                conversation_text
            )

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
        """Update context usage and conversation turns in system monitor."""
        if self.system_monitor and hasattr(self.system_monitor, "update_context_usage"):
            estimated_tokens = self._get_conversation_tokens()

            # Get max context from current config
            # (which gets updated during model swaps)
            # Use context.config to ensure we get the most up-to-date config
            current_config = getattr(self.context, "config", self.config)
            max_context = getattr(current_config.model, "model_max_length", None)

            # If no max_context found in config, try to get it using the
            # engine-specific logic
            if not max_context:
                max_context = self._get_context_length_for_engine(current_config)

            # If still no max_context, try to get it from inference engine
            if (
                not max_context
                and hasattr(self.context, "inference_engine")
                and self.context.inference_engine
            ):
                engine_config = getattr(
                    self.context.inference_engine, "model_config", None
                )
                if engine_config:
                    # Check various possible attribute names for context length
                    for attr in [
                        "model_max_length",
                        "max_model_len",
                        "max_tokens",
                        "context_length",
                    ]:
                        engine_context = getattr(engine_config, attr, None)
                        if engine_context and engine_context > 0:
                            max_context = engine_context
                            break

                # Also check if the engine itself has a max_context attribute
                if not max_context and hasattr(
                    self.context.inference_engine, "max_context_length"
                ):
                    engine_max = getattr(
                        self.context.inference_engine, "max_context_length", None
                    )
                    if engine_max and engine_max > 0:
                        max_context = engine_max

            # Final fallback to prevent None values
            if not max_context or max_context <= 0:
                max_context = 4096

            self.system_monitor.update_context_usage(estimated_tokens)
            if hasattr(self.system_monitor, "update_max_context_tokens"):
                self.system_monitor.update_max_context_tokens(max_context)

            # Also update conversation turn count (matches main inference loop logic)
            if hasattr(self.system_monitor, "update_conversation_turns"):
                assistant_messages = [
                    msg
                    for msg in self.conversation_history
                    if msg.get("role") == "assistant"
                ]
                self.system_monitor.update_conversation_turns(len(assistant_messages))

        # Auto-save after context updates (conversation modifications)
        self._auto_save_if_enabled()

    def _get_context_length_for_engine(self, config) -> Optional[int]:
        """Get the appropriate context length for the given engine configuration.

        Args:
            config: The inference configuration.

        Returns:
            Context length in tokens.
        """
        engine_type = str(config.engine) if config.engine else "NATIVE"
        # For local engines, check model_max_length
        if (
            "NATIVE" in engine_type
            or "VLLM" in engine_type
            or "LLAMACPP" in engine_type
        ):
            max_length = getattr(config.model, "model_max_length", None)
            if max_length is not None and max_length > 0:
                return max_length

        # For API engines, use hardcoded context limits based on model patterns
        model_name = getattr(config.model, "model_name", "").lower()

        # Anthropic context limits
        if "ANTHROPIC" in engine_type or "claude" in model_name:
            if "opus" in model_name:
                return 200000  # Claude Opus
            elif "sonnet" in model_name:
                return 200000  # Claude 3.5 Sonnet / 3.7 Sonnet
            elif "haiku" in model_name:
                return 200000  # Claude Haiku
            else:
                return 200000  # Default for Claude models

        # OpenAI context limits
        elif "OPENAI" in engine_type or "gpt" in model_name:
            if "gpt-4o" in model_name:
                return 128000  # GPT-4o
            elif "gpt-4" in model_name:
                return 128000  # GPT-4
            elif "gpt-3.5" in model_name:
                return 16385  # GPT-3.5-turbo
            else:
                return 128000  # Default for OpenAI models

        # Together AI context limits (varies by model)
        elif "TOGETHER" in engine_type:
            if "llama" in model_name:
                if "405b" in model_name:
                    return 131072  # Llama 3.1 405B
                elif any(x in model_name for x in ["70b", "8b"]):
                    return 131072  # Llama 3.1 70B/8B
                else:
                    return 32768  # Default Llama
            elif "deepseek" in model_name:
                return 32768  # DeepSeek models
            elif "qwen" in model_name:
                return 32768  # Qwen models
            else:
                return 32768  # Default for Together models

        # Other API engines - return reasonable defaults
        elif any(x in engine_type for x in ["DEEPSEEK", "GOOGLE", "GEMINI"]):
            return 128000  # Default for other API models

        # If we can't determine, return None to continue fallback chain
        return None

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
