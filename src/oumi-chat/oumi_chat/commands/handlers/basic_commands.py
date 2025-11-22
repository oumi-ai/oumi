"""Basic command handlers using decorator-based system.

This module demonstrates the simpler decorator-based approach for defining
commands. These can coexist with class-based handlers.
"""

from oumi_chat.commands.base_handler import CommandResult
from oumi_chat.commands.command_context import CommandContext
from oumi_chat.commands.command_parser import ParsedCommand
from oumi_chat.commands.decorators import command


@command("help", "Show available commands and usage information", examples=["/help()"])
def cmd_help(context: CommandContext, parsed_command: ParsedCommand) -> CommandResult:
    """Handle the /help() command.

    Args:
        context: Shared command context.
        parsed_command: Parsed command (no arguments expected).

    Returns:
        CommandResult with help text.
    """
    from rich.panel import Panel
    from rich.text import Text

    # This would generate comprehensive help - simplified for demo
    help_text = Text()
    help_text.append("Available Commands:\n\n", style="bold")
    help_text.append("Basic:\n", style="bold cyan")
    help_text.append("  /help() - Show this help message\n")
    help_text.append("  /exit() - Exit the chat session\n")
    help_text.append("  /clear() - Clear conversation history\n\n")
    help_text.append("Files:\n", style="bold cyan")
    help_text.append("  /attach(path) - Attach a file\n")
    help_text.append("  /save(path) - Save conversation\n")
    help_text.append("  /import(path) - Import conversation\n\n")
    help_text.append("Type any command name for more details.\n", style="dim")

    panel = Panel(
        help_text,
        title="Oumi Chat Help",
        border_style="cyan",
        padding=(1, 2),
    )
    context.console.print(panel)

    return CommandResult(
        success=True,
        message=None,  # Already displayed
        should_continue=False,
    )


@command("exit", "Exit the interactive chat session", examples=["/exit()"])
def cmd_exit(context: CommandContext, parsed_command: ParsedCommand) -> CommandResult:
    """Handle the /exit() command.

    Args:
        context: Shared command context.
        parsed_command: Parsed command (no arguments expected).

    Returns:
        CommandResult with should_exit=True.
    """
    from rich.text import Text

    context.console.print(
        Text("👋 Goodbye!", style="bold magenta")
        if getattr(context.config.style, "use_emoji", True)
        else Text("Goodbye!", style="bold magenta")
    )

    return CommandResult(
        success=True,
        should_exit=True,
        should_continue=False,
    )


@command("clear", "Clear entire conversation history", examples=["/clear()"])
def cmd_clear(context: CommandContext, parsed_command: ParsedCommand) -> CommandResult:
    """Handle the /clear() command.

    Args:
        context: Shared command context.
        parsed_command: Parsed command (no arguments expected).

    Returns:
        CommandResult indicating success.
    """
    # Clear the conversation history
    context.conversation_history.clear()

    # Clear terminal
    context.console.clear()

    from rich.text import Text

    context.console.print(
        Text("✨ Conversation cleared!", style="bold green")
        if getattr(context.config.style, "use_emoji", True)
        else Text("Conversation cleared!", style="bold green")
    )

    return CommandResult(
        success=True,
        message="Conversation history cleared",
        should_continue=False,
    )
