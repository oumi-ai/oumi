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

"""Multi-line input handler for Oumi interactive chat."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
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


class MultiLineInput:
    """Enhanced input handler with multi-line support.
    
    Supports two input modes:
    1. Single-line mode (default): Enter submits, type /ml to switch to multi-line
    2. Multi-line mode: Enter adds new line, empty line submits, /sl to switch back
    
    Special inputs:
    - /ml - Switch to multi-line mode
    - /sl - Switch to single-line mode  
    - /exit - Exit chat
    - Empty input in multi-line mode - Submit
    """
    
    def __init__(self, console: Console, prompt_style: str = "bold blue"):
        """Initialize the input handler.
        
        Args:
            console: Rich console for output.
            prompt_style: Style for the prompt text.
        """
        self.console = console
        self.prompt_style = prompt_style
        self.multiline_mode = False
        self._first_run = True
    
    def get_input(self, prompt: str = "You") -> InputResult:
        """Get input from the user with mode-aware handling.
        
        Args:
            prompt: The prompt text to display.
            
        Returns:
            InputResult with the user's input and action taken.
        """
        # Show help on first run
        if self._first_run:
            self._show_input_help()
            self._first_run = False
        
        if self.multiline_mode:
            return self._get_multiline_input(prompt)
        else:
            return self._get_singleline_input(prompt)
    
    def _get_singleline_input(self, prompt: str) -> InputResult:
        """Get single-line input with mode switching support."""
        try:
            mode_indicator = "[dim](single-line)[/dim]" if self.multiline_mode is False else ""
            full_prompt = f"[{self.prompt_style}]{prompt}[/{self.prompt_style}] {mode_indicator}"
            
            text = Prompt.ask(full_prompt, console=self.console, default="")
            
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
    
    def _get_multiline_input(self, prompt: str) -> InputResult:
        """Get multi-line input with line-by-line collection."""
        lines = []
        line_number = 0
        
        try:
            self.console.print(f"[dim]📝 Multi-line mode: Enter empty line to submit, /sl to switch to single-line[/dim]")
            
            while True:
                line_number += 1
                
                if line_number == 1:
                    # First line with main prompt
                    line_prompt = f"[{self.prompt_style}]{prompt}[/{self.prompt_style}]"
                else:
                    # Continuation lines with visual indicator
                    spaces = " " * len(prompt)
                    line_prompt = f"[{self.prompt_style}]{spaces}[/{self.prompt_style}][dim]│[/dim]"
                
                line = Prompt.ask(line_prompt, console=self.console, default="")
                
                # Handle special commands
                if line.strip().lower() == "/sl":
                    self.multiline_mode = False
                    self._show_mode_change("single-line")
                    if lines:
                        # If we have content, submit it
                        text = "\n".join(lines).strip()
                        return InputResult(action=InputAction.SUBMIT, text=text)
                    else:
                        return InputResult(action=InputAction.TOGGLE_MULTILINE, multiline_toggled=True)
                elif line.strip().lower() == "/exit":
                    return InputResult(action=InputAction.EXIT, should_exit=True)
                elif line.strip() == "" and lines:
                    # Empty line submits the input if we have content
                    break
                elif line.strip() == "" and not lines:
                    # First line is empty, cancel
                    return InputResult(action=InputAction.CANCEL, cancelled=True)
                
                lines.append(line)
            
            text = "\n".join(lines).strip()
            return InputResult(action=InputAction.SUBMIT, text=text)
            
        except (EOFError, KeyboardInterrupt):
            return InputResult(action=InputAction.EXIT, should_exit=True)
    
    def _show_input_help(self):
        """Show help information about input modes."""
        help_text = """
**Input Modes:**
• **Single-line** (default): Press Enter to send message
• **Multi-line**: Press Enter to add new line, empty line to send

**Mode Switching:**
• Type `/ml` to switch to multi-line mode
• Type `/sl` to switch to single-line mode  
• Type `/exit` to exit chat

**Commands work in both modes** (e.g., `/help()`, `/attach()`)
        """
        
        self.console.print(Panel(
            help_text.strip(),
            title="[bold cyan]💬 Input Help[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        ))
        self.console.print()
    
    def _show_mode_change(self, new_mode: str):
        """Show feedback when input mode changes."""
        emoji = "📝" if new_mode == "multi-line" else "✏️"
        self.console.print(f"[green]{emoji} Switched to {new_mode} input mode[/green]")
        
        if new_mode == "multi-line":
            self.console.print("[dim]Press Enter to add new lines, empty line to submit[/dim]")
        else:
            self.console.print("[dim]Press Enter to send message[/dim]")
        
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