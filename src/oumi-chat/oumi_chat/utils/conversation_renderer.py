"""Conversation rendering utilities.

Provides functions for rendering conversation history to the terminal.
"""

from typing import Any, Optional

from rich.console import Console


def render_conversation_history(
    conversation_history: list[dict[str, Any]],
    console: Console,
    config: Optional[Any] = None,
    command_context: Optional[Any] = None,
) -> None:
    """Render conversation history to the console.

    Args:
        conversation_history: List of conversation messages (dicts with 'role' and 'content').
        console: Rich console for output.
        config: Optional inference config for style parameters.
        command_context: Optional command context for advanced formatting.
    """
    try:
        from unittest.mock import MagicMock

        from oumi.core.types.conversation import Conversation, Message, Role
        from oumi.infer import _display_user_message, _format_conversation_response

        # Get style params
        style_params = None
        if config and hasattr(config, "style"):
            style_params = config.style
        elif command_context and hasattr(command_context, "config"):
            style_params = getattr(command_context.config, "style", None)

        # Render each message
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                # Display user message
                is_command = content.strip().startswith("/")
                _display_user_message(
                    console=console,
                    user_text=content,
                    style_params=style_params,
                    is_command=is_command,
                )
            elif role == "assistant":
                # Display assistant message
                message_obj = Message(role=Role.ASSISTANT, content=content)
                conversation = Conversation(messages=[message_obj])

                # Get model name
                model_name = "Assistant"
                if config and hasattr(config, "model"):
                    model_name = getattr(config.model, "model_name", "Assistant")
                    # Simplify model name (e.g., "org/model-name" -> "model-name")
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
        # Log warning but don't crash
        from oumi.utils.logging import logger

        logger.warning(f"Failed to render conversation history: {e}")
        from rich.text import Text

        console.print(
            Text("Could not display conversation history.", style="dim yellow")
        )


def render_branch_switch_header(
    console: Console,
    branch_name: str,
    use_emoji: bool = True,
) -> None:
    """Render a branch switch header.

    Args:
        console: Rich console for output.
        branch_name: Name of the branch being switched to.
        use_emoji: Whether to include emoji in the header.
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
    """Clear terminal and render branch with conversation history.

    Args:
        console: Rich console for output.
        branch: Branch object with name attribute.
        conversation_history: List of conversation messages.
        config: Optional inference config.
        command_context: Optional command context.
    """
    try:
        # Clear terminal
        console.clear()

        # Show branch switch header
        branch_name = getattr(branch, "name", "Unknown Branch")
        use_emoji = True
        if config and hasattr(config, "style"):
            use_emoji = getattr(config.style, "use_emoji", True)

        render_branch_switch_header(console, branch_name, use_emoji)

        # Render conversation history
        if conversation_history and len(conversation_history) > 0:
            render_conversation_history(
                conversation_history, console, config, command_context
            )
        else:
            # Show empty branch message
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
        # Log warning but don't crash
        from oumi.utils.logging import logger

        logger.warning(f"Failed to refresh conversation display: {e}")
