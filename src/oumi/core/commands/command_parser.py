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

"""Command parser for interactive Oumi inference commands."""

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ParsedCommand:
    """Represents a parsed command from user input.
    
    Examples:
        /help() -> command="help", args=[], kwargs={}, raw_input="/help()"
        /attach(image.jpg) -> command="attach", args=["image.jpg"], kwargs={}, raw_input="/attach(image.jpg)"
        /set(temperature=0.8) -> command="set", args=[], kwargs={"temperature": "0.8"}, raw_input="/set(temperature=0.8)"
    """
    
    command: str
    """The command name (e.g., 'help', 'exit', 'attach')."""
    
    args: list[str]
    """Positional arguments passed to the command."""
    
    kwargs: dict[str, str]
    """Keyword arguments passed to the command."""
    
    raw_input: str
    """The original user input that was parsed."""


class CommandParser:
    """Parser for interactive commands in Oumi inference.
    
    Supports the following command syntaxes:
    - /command() - Simple command with no arguments
    - /command(arg1, arg2) - Command with positional arguments
    - /command(key=value) - Command with keyword arguments
    - /command(arg1, key=value) - Mixed positional and keyword arguments
    
    Examples:
        /help()
        /exit()
        /attach(path/to/file.txt)
        /set(temperature=0.8, top_p=0.9)
        /save(/path/to/output.pdf)
    """
    
    # Regex patterns for command parsing
    COMMAND_PATTERN = re.compile(
        r"^/(\w+)\s*(?:\((.*?)\))?\s*$", re.DOTALL
    )
    
    ARG_PATTERN = re.compile(
        r"""
        (?:
            (\w+)\s*=\s*           # keyword argument start: key=
            (?:
                "([^"]*)"          # quoted value
                |
                '([^']*)'          # single quoted value
                |
                ([^,\)]+)          # unquoted value
            )
            |
            (?:
                "([^"]*)"          # quoted positional argument
                |
                '([^']*)'          # single quoted positional argument
                |
                ([^,=\)]+?)        # unquoted positional argument
            )
        )
        (?:\s*,\s*|$)              # comma separator or end
        """,
        re.VERBOSE
    )
    
    def __init__(self):
        """Initialize the command parser."""
        self._available_commands = {
            "help": "Show available commands and usage information",
            "exit": "Exit the interactive chat session",
            "attach": "Attach a file to the conversation (images, PDFs, text, etc.)",
            "delete": "Delete the previous conversation turn",
            "regen": "Regenerate the last assistant response",
            "save": "Save the current conversation to a PDF file",
            "set": "Adjust generation parameters (temperature, top_p, max_tokens, etc.)",
            "compact": "Compress conversation history to save context window space",
            "branch": "Create a new conversation branch from current point",
            "switch": "Switch to a different conversation branch",
            "branches": "List all conversation branches",
            "branch_delete": "Delete a conversation branch",
            # Note: /ml and /sl are handled by the input system, not as regular commands
        }
    
    def is_command(self, input_text: str) -> bool:
        """Check if the input text is a command.
        
        Args:
            input_text: The user input to check.
            
        Returns:
            True if the input starts with '/' and looks like a command.
        """
        if not input_text or not isinstance(input_text, str):
            return False
        
        stripped = input_text.strip()
        return stripped.startswith('/') and len(stripped) > 1
    
    def parse_command(self, input_text: str) -> Optional[ParsedCommand]:
        """Parse a command string into its components.
        
        Args:
            input_text: The command string to parse.
            
        Returns:
            ParsedCommand object if parsing succeeds, None if invalid.
            
        Examples:
            >>> parser = CommandParser()
            >>> cmd = parser.parse_command("/help()")
            >>> cmd.command
            'help'
            >>> cmd.args
            []
            >>> cmd.kwargs
            {}
        """
        if not self.is_command(input_text):
            return None
        
        # Match the overall command structure
        match = self.COMMAND_PATTERN.match(input_text.strip())
        if not match:
            return None
        
        command_name = match.group(1).lower()
        args_str = match.group(2) or ""
        
        # Parse arguments
        args = []
        kwargs = {}
        
        if args_str.strip():
            # Find all argument matches
            for arg_match in self.ARG_PATTERN.finditer(args_str):
                # Groups: 1=key, 2=quoted_value, 3=single_quoted_value, 4=unquoted_value,
                #         5=quoted_pos, 6=single_quoted_pos, 7=unquoted_pos
                key = arg_match.group(1)
                
                if key:
                    # Keyword argument - extract value from appropriate group
                    value = (arg_match.group(2) or       # double quoted
                            arg_match.group(3) or        # single quoted
                            arg_match.group(4) or "")    # unquoted
                    kwargs[key.strip()] = value.strip()
                else:
                    # Positional argument - extract from appropriate group
                    pos_arg = (arg_match.group(5) or     # double quoted
                              arg_match.group(6) or      # single quoted  
                              arg_match.group(7) or "")  # unquoted
                    if pos_arg:
                        args.append(pos_arg.strip())
        
        return ParsedCommand(
            command=command_name,
            args=args,
            kwargs=kwargs,
            raw_input=input_text
        )
    
    def get_available_commands(self) -> dict[str, str]:
        """Get a dictionary of available commands and their descriptions.
        
        Returns:
            Dict mapping command names to their descriptions.
        """
        return self._available_commands.copy()
    
    def is_valid_command(self, command_name: str) -> bool:
        """Check if a command name is valid/supported.
        
        Args:
            command_name: The command name to check.
            
        Returns:
            True if the command is supported.
        """
        return command_name.lower() in self._available_commands
    
    def get_command_help(self, command_name: str) -> Optional[str]:
        """Get help text for a specific command.
        
        Args:
            command_name: The command to get help for.
            
        Returns:
            Help text for the command, or None if not found.
        """
        return self._available_commands.get(command_name.lower())
    
    def validate_command(self, parsed_command: ParsedCommand) -> tuple[bool, str]:
        """Validate a parsed command for correctness.
        
        Args:
            parsed_command: The parsed command to validate.
            
        Returns:
            Tuple of (is_valid, error_message). If is_valid is True, 
            error_message will be empty.
        """
        if not self.is_valid_command(parsed_command.command):
            return False, f"Unknown command: '{parsed_command.command}'"
        
        # Command-specific validation
        if parsed_command.command == "attach":
            if not parsed_command.args:
                return False, "attach command requires a file path argument"
        elif parsed_command.command == "save":
            if not parsed_command.args:
                return False, "save command requires an output path argument"
        elif parsed_command.command == "set":
            if not parsed_command.kwargs:
                return False, "set command requires parameter=value arguments"
        elif parsed_command.command == "switch":
            if not parsed_command.args:
                return False, "switch command requires a branch name or ID"
        elif parsed_command.command == "branch_delete":
            if not parsed_command.args:
                return False, "branch_delete command requires a branch name or ID"
        elif parsed_command.command in ["help", "exit", "delete", "regen", "compact", "branch", "branches"]:
            # These commands don't require arguments
            pass
        
        return True, ""