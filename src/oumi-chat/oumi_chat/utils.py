"""Utility functions for oumi-chat.

This module consolidates all utility functions into a single file for simplicity.
"""

from pathlib import Path
from typing import Any, Optional

from rich.console import Console


# ============================================================================
# Model Information Utilities
# ============================================================================


def get_context_length_for_engine(config) -> Optional[int]:
    """Get the appropriate context length for the given engine configuration.

    Args:
        config: The inference configuration object with engine and model attributes.

    Returns:
        Context length in tokens, or None if it cannot be determined.
    """
    engine_type = str(config.engine) if config.engine else "NATIVE"

    # For local engines, check model_max_length
    if "NATIVE" in engine_type or "VLLM" in engine_type or "LLAMACPP" in engine_type:
        max_length = getattr(config.model, "model_max_length", None)
        if max_length is not None and max_length > 0:
            return max_length

    # For API engines, use known context limits
    model_name = getattr(config.model, "model_name", "").lower()

    # Anthropic context limits
    if "ANTHROPIC" in engine_type or "claude" in model_name:
        return 200000  # All Claude 3+ models

    # OpenAI context limits
    if "OPENAI" in engine_type or "gpt" in model_name:
        if "gpt-3.5" in model_name:
            return 16385
        return 128000  # GPT-4, GPT-4o

    # Together AI
    if "TOGETHER" in engine_type:
        if "llama" in model_name:
            return 128000
        return 32768

    # DeepSeek
    if "DEEPSEEK" in engine_type or "deepseek" in model_name:
        return 32768

    # Google Gemini
    if "GOOGLE" in engine_type or "GEMINI" in engine_type or "gemini" in model_name:
        return 128000

    return None


# ============================================================================
# File Validation Utilities
# ============================================================================


def validate_and_sanitize_file_path(file_path: str) -> tuple[bool, str, str]:
    """Validate and sanitize a file path for security and safety.

    Args:
        file_path: The file path to validate.

    Returns:
        Tuple of (is_valid, sanitized_path, error_message).
    """
    try:
        from pathvalidate import ValidationError, is_valid_filepath, sanitize_filepath
    except ImportError:
        return (False, "", "pathvalidate library required")

    if not file_path:
        return False, "", "File path cannot be empty"

    # Check for unmatched quotes
    stripped = file_path.strip()
    for quote in ["'", '"']:
        if stripped.startswith(quote) != stripped.endswith(quote):
            return False, "", f"Unmatched quote: {quote}"

    # Strip whitespace and quotes
    cleaned = file_path.strip().strip("\"'")
    if not cleaned or cleaned.isspace():
        return False, "", "Path is empty or whitespace"

    # Detect platform
    import os

    platform = "windows" if os.name == "nt" else "posix"

    # Sanitize
    try:
        sanitized = sanitize_filepath(cleaned, platform=platform, max_len=255)
    except ValidationError as e:
        return False, "", f"Invalid path: {e}"

    # Verify valid
    if not is_valid_filepath(sanitized, platform=platform):
        return False, "", "Invalid characters or format"

    # Prevent path traversal
    if ".." in sanitized:
        return False, "", "Path traversal detected"

    # No quotes in filename
    if any(q in Path(sanitized).name for q in ["'", '"']):
        return False, "", "Filename cannot contain quotes"

    return True, sanitized, ""


# ============================================================================
# Token Counting Utilities
# ============================================================================


def count_conversation_tokens(
    conversation_history: list[dict[str, Any]],
    context_window_manager=None,
) -> int:
    """Count tokens in conversation history.

    Args:
        conversation_history: List of conversation messages.
        context_window_manager: Optional manager for accurate counting.

    Returns:
        Estimated token count.
    """
    if context_window_manager:
        # Accurate tiktoken estimation
        text = ""
        for msg in conversation_history:
            if isinstance(msg, dict):
                if "content" in msg:
                    text += str(msg["content"]) + "\n"
                elif msg.get("role") == "attachment" and "text_content" in msg:
                    text += str(msg["text_content"]) + "\n"
                elif msg.get("role") == "attachment" and "content" in msg:
                    text += str(msg["content"]) + "\n"
        return context_window_manager.estimate_tokens(text)

    # Fallback: character-based
    total_chars = 0
    for msg in conversation_history:
        if msg.get("role") == "attachment":
            content = msg.get("text_content", "") or msg.get("content", "")
        else:
            content = msg.get("content", "")
        total_chars += len(str(content))

    return total_chars // 4  # ~4 chars per token


# ============================================================================
# Conversation Rendering Utilities
# ============================================================================


def render_conversation_history(
    conversation_history: list[dict[str, Any]],
    console: Console,
    config: Optional[Any] = None,
    command_context: Optional[Any] = None,
) -> None:
    """Render conversation history to the console.

    Args:
        conversation_history: List of messages.
        console: Rich console for output.
        config: Optional inference config.
        command_context: Optional command context.
    """
    try:
        from unittest.mock import MagicMock

        from oumi.core.types.conversation import Conversation, Message, Role
        from oumi.infer import _display_user_message, _format_conversation_response

        style_params = None
        if config and hasattr(config, "style"):
            style_params = config.style
        elif command_context and hasattr(command_context, "config"):
            style_params = getattr(command_context.config, "style", None)

        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                is_command = content.strip().startswith("/")
                _display_user_message(
                    console=console,
                    user_text=content,
                    style_params=style_params,
                    is_command=is_command,
                )
            elif role == "assistant":
                message_obj = Message(role=Role.ASSISTANT, content=content)
                conversation = Conversation(messages=[message_obj])

                model_name = "Assistant"
                if config and hasattr(config, "model"):
                    model_name = getattr(config.model, "model_name", "Assistant")
                    if "/" in model_name:
                        model_name = model_name.split("/")[-1]

                _format_conversation_response(
                    conversation=conversation,
                    console=console,
                    model_name=model_name,
                    style_params=style_params,
                    command_context=command_context,
                )

    except Exception as e:
        from oumi.utils.logging import logger

        logger.warning(f"Failed to render history: {e}")
        from rich.text import Text

        console.print(Text("Could not display history.", style="dim yellow"))


def render_branch_switch_header(
    console: Console, branch_name: str, use_emoji: bool = True
) -> None:
    """Render a branch switch header.

    Args:
        console: Rich console.
        branch_name: Name of branch.
        use_emoji: Whether to include emoji.
    """
    from rich.panel import Panel
    from rich.text import Text

    header_text = f"🌿 Switched to branch: {branch_name}"
    if not use_emoji:
        header_text = f"Switched to branch: {branch_name}"

    header = Panel(
        Text(header_text, style="bold cyan", justify="center"),
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(header)
    console.print()


def clear_and_render_branch(
    console: Console,
    branch: Any,
    conversation_history: list[dict[str, Any]],
    config: Optional[Any] = None,
    command_context: Optional[Any] = None,
) -> None:
    """Clear terminal and render branch with conversation.

    Args:
        console: Rich console.
        branch: Branch object.
        conversation_history: List of messages.
        config: Optional config.
        command_context: Optional context.
    """
    try:
        console.clear()

        branch_name = getattr(branch, "name", "Unknown Branch")
        use_emoji = True
        if config and hasattr(config, "style"):
            use_emoji = getattr(config.style, "use_emoji", True)

        render_branch_switch_header(console, branch_name, use_emoji)

        if conversation_history and len(conversation_history) > 0:
            render_conversation_history(
                conversation_history, console, config, command_context
            )
        else:
            from rich.text import Text

            console.print(
                Text(
                    "No conversation history on this branch yet.",
                    style="dim",
                    justify="center",
                )
            )
            console.print()

    except Exception as e:
        from oumi.utils.logging import logger

        logger.warning(f"Failed to refresh display: {e}")
