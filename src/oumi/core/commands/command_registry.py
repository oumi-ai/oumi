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

"""Centralized command registry for Oumi interactive commands.

This module defines all available commands in one place to ensure consistency
across command parsing, completion, and help systems.
"""

from dataclasses import dataclass


@dataclass
class CommandInfo:
    """Information about an interactive command."""

    name: str
    """The command name (e.g., 'help', 'attach', 'fetch')."""

    description: str
    """Short description of what the command does."""

    examples: list[str]
    """List of example usage patterns for autocomplete."""


class CommandRegistry:
    """Central registry for all Oumi interactive commands."""

    def __init__(self):
        """Initialize the command registry with all available commands."""
        self._commands = {
            # Basic commands
            "help": CommandInfo(
                name="help",
                description="Show available commands and usage information",
                examples=["/help()"],
            ),
            "exit": CommandInfo(
                name="exit",
                description="Exit the interactive chat session",
                examples=["/exit()"],
            ),
            # File operations
            "attach": CommandInfo(
                name="attach",
                description=(
                    "Attach a file to the conversation (images, PDFs, text, etc.)"
                ),
                examples=[
                    "/attach(file_path)",
                    "/attach(document.pdf)",
                    "/attach(image.jpg)",
                ],
            ),
            "fetch": CommandInfo(
                name="fetch",
                description="Fetch web content and add to conversation context",
                examples=[
                    "/fetch(url)",
                    "/fetch(https://example.com)",
                    "/fetch(docs.python.org)",
                ],
            ),
            "shell": CommandInfo(
                name="shell",
                description="Execute safe shell commands and capture output",
                examples=[
                    "/shell(command)",
                    "/shell(ls -la)",
                    "/shell(python --version)",
                    "/shell(git status)",
                ],
            ),
            "save": CommandInfo(
                name="save",
                description="Save the current conversation to various formats",
                examples=[
                    "/save(output_path)",
                    "/save(output.pdf)",
                    "/save(output.json)",
                    "/save(output.csv)",
                    "/save(output.md)",
                    "/save(output.html)",
                ],
            ),
            "import": CommandInfo(
                name="import",
                description="Import conversation data from supported file formats",
                examples=[
                    "/import(input_file)",
                    "/import(data.csv)",
                    "/import(chat.json)",
                    "/import(conversation.xlsx)",
                ],
            ),
            "save_history": CommandInfo(
                name="save_history",
                description="Save complete conversation state (all branches + config)",
                examples=[
                    "/save_history(output.json)",
                    "/save_history(complete.json)",
                    "/save_history(backup.json)",
                ],
            ),
            "import_history": CommandInfo(
                name="import_history",
                description="Restore complete conversation state",
                examples=[
                    "/import_history(input.json)",
                    "/import_history(complete.json)",
                    "/import_history(backup.json)",
                ],
            ),
            "load": CommandInfo(
                name="load",
                description=(
                    "Load a previously saved chat from cache by ID"
                ),
                examples=[
                    "/load(chat_id)",
                    "/load(recent)",
                    "/load(session_20241201_143022)",
                ],
            ),
            # Conversation management
            "delete": CommandInfo(
                name="delete",
                description="Delete the previous conversation turn",
                examples=["/delete()"],
            ),
            "regen": CommandInfo(
                name="regen",
                description="Regenerate the last assistant response",
                examples=["/regen()"],
            ),
            "clear": CommandInfo(
                name="clear",
                description="Clear entire conversation history",
                examples=["/clear()"],
            ),
            "clear_thoughts": CommandInfo(
                name="clear_thoughts",
                description=(
                    "Remove thinking content from conversation history while "
                    "preserving responses"
                ),
                examples=["/clear_thoughts()"],
            ),
            "compact": CommandInfo(
                name="compact",
                description=(
                    "Compress conversation history to save context window space"
                ),
                examples=["/compact()"],
            ),
            "show": CommandInfo(
                name="show",
                description="View a specific conversation position",
                examples=[
                    "/show()",
                    "/show(1)",
                    "/show(2)",
                    "/show(3)",
                ],
            ),
            "render": CommandInfo(
                name="render",
                description="Record conversation playback as asciinema recording",
                examples=[
                    "/render(output.cast)",
                    "/render(conversation.cast)",
                    "/render(demo.cast)",
                ],
            ),
            # Generation parameters
            "set": CommandInfo(
                name="set",
                description=(
                    "Adjust generation parameters (temperature, top_p, max_tokens, "
                    "etc.)"
                ),
                examples=[
                    "/set(temperature=0.7)",
                    "/set(top_p=0.9)",
                    "/set(max_tokens=2048)",
                    "/set(sampling=true)",
                    "/set(seed=42)",
                ],
            ),
            # Branching
            "branch": CommandInfo(
                name="branch",
                description="Create a new conversation branch from current point",
                examples=["/branch()", "/branch(branch_name)"],
            ),
            "branch_from": CommandInfo(
                name="branch_from",
                description="Create a branch from specific assistant message position",
                examples=[
                    "/branch_from(name,pos)",
                    "/branch_from(experiment,2)",
                    "/branch_from(test,3)",
                ],
            ),
            "switch": CommandInfo(
                name="switch",
                description="Switch to a different conversation branch",
                examples=["/switch(branch_name)", "/switch(main)", "/switch(branch_1)"],
            ),
            "branches": CommandInfo(
                name="branches",
                description="List all conversation branches",
                examples=["/branches()"],
            ),
            "branch_delete": CommandInfo(
                name="branch_delete",
                description="Delete a conversation branch",
                examples=["/branch_delete(branch_name)"],
            ),
            # Thinking modes
            "full_thoughts": CommandInfo(
                name="full_thoughts",
                description="Toggle between compressed and full thinking display modes",
                examples=["/full_thoughts()"],
            ),
            # Model management
            "swap": CommandInfo(
                name="swap",
                description=(
                    "Switch to a different model or config for inference while "
                    "preserving conversation history"
                ),
                examples=[
                    "/swap(model_name)",
                    "/swap(engine:model_name)",
                    "/swap(config:path/to/config.yaml)",
                ],
            ),
            "list_engines": CommandInfo(
                name="list_engines",
                description=(
                    "List available inference engines and their supported model "
                    "examples"
                ),
                examples=["/list_engines()"],
            ),
            # Macro system
            "macro": CommandInfo(
                name="macro",
                description="Execute a Jinja template-based conversation macro",
                examples=[
                    "/macro(template_path)",
                    "/macro(judge.jinja)",
                    "/macro(code_repair.jinja)",
                    "/macro(creative_writing.jinja)",
                    "/macro(macros/template.jinja)",
                ],
            ),
        }

    def get_command(self, name: str) -> CommandInfo:
        """Get command info by name.

        Args:
            name: The command name to look up.

        Returns:
            CommandInfo for the command.

        Raises:
            KeyError: If the command is not found.
        """
        return self._commands[name.lower()]

    def has_command(self, name: str) -> bool:
        """Check if a command exists in the registry.

        Args:
            name: The command name to check.

        Returns:
            True if the command exists, False otherwise.
        """
        return name.lower() in self._commands

    def get_all_commands(self) -> dict[str, CommandInfo]:
        """Get all registered commands.

        Returns:
            Dictionary mapping command names to CommandInfo objects.
        """
        return self._commands.copy()

    def get_command_names(self) -> list[str]:
        """Get a list of all command names.

        Returns:
            List of command names.
        """
        return list(self._commands.keys())

    def get_all_examples(self) -> list[str]:
        """Get all command examples for autocomplete.

        Returns:
            Flattened list of all command examples.
        """
        examples = []
        for command_info in self._commands.values():
            examples.extend(command_info.examples)
        return examples


# Global registry instance
COMMAND_REGISTRY = CommandRegistry()
