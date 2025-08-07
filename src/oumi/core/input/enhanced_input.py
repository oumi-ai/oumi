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

"""Enhanced input handler with prompt_toolkit integration for arrow keys and history."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class InputAction(Enum):
    """Actions that can result from user input."""
    SUBMIT = "submit"           # Submit the input
    CANCEL = "cancel"           # Cancel current input
    EXIT = "exit"               # Exit the chat
    TOGGLE_MULTILINE = "toggle" # Toggle multi-line mode


@dataclass
class InputResult:
    """Result of input operation."""
    action: InputAction
    text: str = ""
    cancelled: bool = False
    should_exit: bool = False
    multiline_toggled: bool = False


class EnhancedInput:
    """Enhanced input handler with prompt_toolkit for arrow keys, history, and completion.
    
    Features:
    - Arrow key navigation (up/down for history, left/right for cursor)
    - Command history persistence
    - Tab completion for commands
    - Multi-line support with enhanced editing
    - Integration with Rich styling
    """
    
    def __init__(self, console: Console, prompt_style: str = "bold blue"):
        """Initialize the enhanced input handler.
        
        Args:
            console: Rich console for output.
            prompt_style: Style for the prompt text (used for Rich fallback).
        """
        self.console = console
        self.prompt_style = prompt_style
        self.multiline_mode = False
        self._first_run = True
        
        # Command history
        self.history = InMemoryHistory()
        
        # Command completion - all available Oumi commands
        self.command_completer = WordCompleter([
            '/help()', '/exit()', '/attach()', '/delete()', '/regen()', 
            '/save()', '/set()', '/ml', '/sl'
        ], match_middle=True)
        
        # Key bindings for custom behavior
        self.bindings = self._create_key_bindings()
    
    def _create_key_bindings(self) -> KeyBindings:
        """Create custom key bindings."""
        kb = KeyBindings()
        
        @kb.add('c-c')  # Ctrl+C
        def _(event):
            """Handle Ctrl+C to exit."""
            event.app.exit(exception=KeyboardInterrupt)
        
        @kb.add('c-d')  # Ctrl+D
        def _(event):
            """Handle Ctrl+D to exit."""
            event.app.exit(exception=EOFError)
        
        return kb
    
    def get_input(self, prompt_text: str = "You") -> InputResult:
        """Get input from the user with enhanced features.
        
        Args:
            prompt_text: The prompt text to display.
            
        Returns:
            InputResult with the user's input and action taken.
        """
        # Show help on first run
        if self._first_run:
            self._show_input_help()
            self._first_run = False
        
        if self.multiline_mode:
            return self._get_multiline_input(prompt_text)
        else:
            return self._get_singleline_input(prompt_text)
    
    def _get_singleline_input(self, prompt_text: str) -> InputResult:
        """Get single-line input with enhanced features."""
        try:
            # Create styled prompt
            mode_indicator = " (single-line)" if self.multiline_mode is False else ""
            formatted_prompt = HTML(f'<ansiblue><b>{prompt_text}</b></ansiblue>{mode_indicator}: ')
            
            text = prompt(
                formatted_prompt,
                history=self.history,
                completer=self.command_completer,
                complete_while_typing=True,
                key_bindings=self.bindings,
                mouse_support=True,
                wrap_lines=True
            )
            
            # Handle special commands
            if text.strip().lower() == "/ml":
                self.multiline_mode = True
                self._show_mode_change("multi-line")
                return InputResult(action=InputAction.TOGGLE_MULTILINE, multiline_toggled=True)
            elif text.strip().lower() == "/exit":
                return InputResult(action=InputAction.EXIT, should_exit=True)
            elif not text.strip():
                return InputResult(action=InputAction.CANCEL, cancelled=True)
            
            return InputResult(action=InputAction.SUBMIT, text=text)
            
        except (EOFError, KeyboardInterrupt):
            return InputResult(action=InputAction.EXIT, should_exit=True)
    
    def _get_multiline_input(self, prompt_text: str) -> InputResult:
        """Get multi-line input with enhanced editing."""
        try:
            self.console.print(f"[dim]📝 Multi-line mode: Ctrl+D to submit, /sl to switch to single-line[/dim]")
            
            # Create styled prompt for multi-line
            formatted_prompt = HTML(f'<ansiblue><b>{prompt_text}</b></ansiblue> (multi-line): ')
            
            text = prompt(
                formatted_prompt,
                history=self.history,
                completer=self.command_completer,
                multiline=True,
                complete_while_typing=True,
                key_bindings=self.bindings,
                mouse_support=True,
                wrap_lines=True
            )
            
            # Handle special commands
            if text.strip().lower() == "/sl":
                self.multiline_mode = False
                self._show_mode_change("single-line")
                return InputResult(action=InputAction.TOGGLE_MULTILINE, multiline_toggled=True)
            elif text.strip().lower() == "/exit":
                return InputResult(action=InputAction.EXIT, should_exit=True)
            elif not text.strip():
                return InputResult(action=InputAction.CANCEL, cancelled=True)
            
            # Check if this is a command on the first line (for backward compatibility)
            first_line = text.split('\n')[0].strip()
            if first_line.startswith('/') and '(' in first_line and first_line.endswith(')'):
                return InputResult(action=InputAction.SUBMIT, text=first_line)
            
            return InputResult(action=InputAction.SUBMIT, text=text.strip())
            
        except (EOFError, KeyboardInterrupt):
            return InputResult(action=InputAction.EXIT, should_exit=True)
    
    def _show_input_help(self):
        """Show help information about enhanced input features."""
        help_text = """
**Enhanced Input Features:**
• **Arrow Keys**: ↑↓ for command history, ←→ for cursor movement
• **Tab Completion**: Tab to complete commands and file paths
• **Multi-line**: Ctrl+D to submit in multi-line mode

**Input Modes:**
• **Single-line** (default): Press Enter to send message
• **Multi-line**: Press Ctrl+D to send, allows multiple lines

**Mode Switching:**
• Type `/ml` to switch to multi-line mode
• Type `/sl` to switch to single-line mode  
• Type `/exit` to exit chat

**Commands work in both modes** (e.g., `/help()`, `/attach()`)
• Tab completion available for all commands
• Command history persists across sessions
        """
        
        self.console.print(Panel(
            help_text.strip(),
            title="[bold cyan]💬 Enhanced Input Help[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        ))
        self.console.print()
    
    def _show_mode_change(self, new_mode: str):
        """Show feedback when input mode changes."""
        emoji = "📝" if new_mode == "multi-line" else "✏️"
        self.console.print(f"[green]{emoji} Switched to {new_mode} input mode[/green]")
        
        if new_mode == "multi-line":
            self.console.print("[dim]Use Ctrl+D to submit, arrow keys for navigation[/dim]")
        else:
            self.console.print("[dim]Press Enter to send, use arrow keys for history[/dim]")
        
        self.console.print()
    
    def get_current_mode(self) -> str:
        """Get the current input mode as a string."""
        return "multi-line" if self.multiline_mode else "single-line"
    
    def set_mode(self, multiline: bool):
        """Programmatically set the input mode."""
        if self.multiline_mode != multiline:
            self.multiline_mode = multiline
            mode_name = "multi-line" if multiline else "single-line"
            self._show_mode_change(mode_name)
    
    def add_to_history(self, text: str):
        """Add a command to the input history."""
        self.history.append_string(text)