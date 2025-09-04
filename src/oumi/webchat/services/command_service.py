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

"""Command service for Oumi WebChat server."""

import io
from typing import Dict, List, Optional, Any, Tuple, Union

from rich.console import Console

from oumi.core.commands.command_context import CommandContext
from oumi.core.commands.command_parser import CommandParser, ParsedCommand
from oumi.core.commands.command_result import CommandResult
from oumi.core.commands.command_router import CommandRouter
from oumi.utils.logging import logger


class CommandService:
    """Handles command parsing and execution for WebChat."""
    
    def __init__(self):
        """Initialize command service."""
        self.command_parser = CommandParser()
    
    def parse_command(self, command_str: str) -> Optional[ParsedCommand]:
        """Parse a command string into a ParsedCommand object.
        
        Args:
            command_str: Command string to parse.
            
        Returns:
            ParsedCommand object if valid, None if not a command.
        """
        if not self.command_parser.is_command(command_str):
            return None
        
        return self.command_parser.parse_command(command_str)
    
    def execute_command(
        self, 
        command: Union[str, ParsedCommand], 
        command_context: CommandContext,
        capture_output: bool = False
    ) -> Tuple[CommandResult, Optional[str]]:
        """Execute a command with the given context.
        
        Args:
            command: Command string or ParsedCommand object.
            command_context: Command context to execute with.
            capture_output: Whether to capture console output.
            
        Returns:
            Tuple of (command_result, console_output) where console_output is 
            only provided if capture_output is True.
        """
        # Parse command if string
        if isinstance(command, str):
            if not self.command_parser.is_command(command):
                return CommandResult(success=False, message=f"Not a valid command: {command}"), None
            parsed_command = self.command_parser.parse_command(command)
        else:
            parsed_command = command
        
        # Create command router with context
        router = CommandRouter(command_context)
        
        # Set up output capture if requested
        captured_output = None
        original_console = None
        
        if capture_output:
            # Create a string buffer to capture console output
            string_buffer = io.StringIO()
            temp_console = Console(file=string_buffer, width=80)
            
            # Save the original console and replace with capture console
            original_console = command_context.console
            command_context.console = temp_console
        
        try:
            # Execute command
            logger.debug(f"Executing command: {parsed_command.command} with args: {parsed_command.args}")
            result = router.handle_command(parsed_command)
            logger.debug(f"Command result: success={result.success}, message={result.message}, should_continue={result.should_continue}")
            
            # Capture output if requested
            if capture_output:
                captured_output = string_buffer.getvalue()
                
                # Format full message with both result and console output
                if captured_output.strip():
                    if result.message:
                        full_output = f"{result.message}\n\n{captured_output.strip()}"
                    else:
                        full_output = captured_output.strip()
                else:
                    full_output = result.message or ""
                
                captured_output = full_output
        
        finally:
            # Restore original console if it was replaced
            if original_console is not None:
                command_context.console = original_console
        
        return result, captured_output
    
    def execute_command_with_args(
        self,
        command: str,
        args: List[str],
        command_context: CommandContext,
        capture_output: bool = False
    ) -> Tuple[CommandResult, Optional[str]]:
        """Execute a command with explicit arguments.
        
        Args:
            command: Command name to execute.
            args: List of command arguments.
            command_context: Command context to execute with.
            capture_output: Whether to capture console output.
            
        Returns:
            Tuple of (command_result, console_output) where console_output is
            only provided if capture_output is True.
        """
        # Create parsed command
        parsed_command = ParsedCommand(
            command=command,
            args=args,
            kwargs={},
            raw_input=f"/{command}({','.join(args)})",
        )
        
        return self.execute_command(parsed_command, command_context, capture_output)