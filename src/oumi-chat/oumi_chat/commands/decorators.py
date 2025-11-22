"""Decorator-based command registration system.

This module provides a simpler, more Pythonic way to define commands using
decorators instead of class-based handlers. It can coexist with the existing
class-based system for backward compatibility.

Example:
    @command("help", "Show available commands", examples=["/help()"])
    def cmd_help(context, parsed_command):
        # Handle the command
        return CommandResult(success=True, message="Help text...")

    # Register with router
    router.register_function_handler(cmd_help)
"""

from typing import Callable, Optional

from oumi_chat.commands.base_handler import CommandResult
from oumi_chat.commands.command_context import CommandContext
from oumi_chat.commands.command_parser import ParsedCommand

# Global registry of decorated command functions
_COMMAND_FUNCTIONS: dict[str, "CommandFunction"] = {}


class CommandFunction:
    """Wrapper for a command function with metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[[CommandContext, ParsedCommand], CommandResult],
        examples: Optional[list[str]] = None,
    ):
        """Initialize command function.

        Args:
            name: Command name (without / prefix).
            description: Short description of what the command does.
            func: The function that handles the command.
            examples: List of example usage patterns.
        """
        self.name = name
        self.description = description
        self.func = func
        self.examples = examples or [f"/{name}()"]

    def __call__(
        self, context: CommandContext, parsed_command: ParsedCommand
    ) -> CommandResult:
        """Execute the command function.

        Args:
            context: Shared command context.
            parsed_command: Parsed command with arguments.

        Returns:
            Result of executing the command.
        """
        return self.func(context, parsed_command)


def command(
    name: str,
    description: str,
    examples: Optional[list[str]] = None,
) -> Callable:
    """Decorator to register a command handler function.

    Args:
        name: Command name (without / prefix).
        description: Short description of what the command does.
        examples: Optional list of example usage patterns.

    Returns:
        Decorator function that wraps the command handler.

    Example:
        @command("clear", "Clear conversation history")
        def cmd_clear(context, parsed_command):
            context.conversation_history.clear()
            return CommandResult(success=True, message="Cleared")
    """

    def decorator(func: Callable[[CommandContext, ParsedCommand], CommandResult]):
        """Wrap the function and register it."""
        cmd_func = CommandFunction(name, description, func, examples)
        _COMMAND_FUNCTIONS[name] = cmd_func
        return func  # Return original function for potential direct use

    return decorator


def get_command_function(name: str) -> Optional[CommandFunction]:
    """Get a registered command function by name.

    Args:
        name: Command name to look up.

    Returns:
        CommandFunction if found, None otherwise.
    """
    return _COMMAND_FUNCTIONS.get(name)


def get_all_command_functions() -> dict[str, CommandFunction]:
    """Get all registered command functions.

    Returns:
        Dictionary mapping command names to CommandFunction objects.
    """
    return _COMMAND_FUNCTIONS.copy()


def has_command_function(name: str) -> bool:
    """Check if a command function is registered.

    Args:
        name: Command name to check.

    Returns:
        True if the command is registered, False otherwise.
    """
    return name in _COMMAND_FUNCTIONS
