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

from rich.console import Console
from rich.panel import Panel

try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
except ImportError as e:
    raise ImportError(
        "prompt_toolkit is required for Oumi input handling. "
        "Please install it with: pip install prompt_toolkit"
    ) from e


class InputAction(Enum):
    """Actions that can result from user input."""

    SUBMIT = "submit"  # Submit the input
    CANCEL = "cancel"  # Cancel current input
    EXIT = "exit"  # Exit the chat
    TOGGLE_MULTILINE = "toggle"  # Toggle multi-line mode


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
    2. Multi-line mode: Enter adds new line, Ctrl+D submits, /sl to switch back

    Special inputs:
    - /ml - Switch to multi-line mode
    - /sl - Switch to single-line mode
    - /exit - Exit chat
    - Ctrl+D in multi-line mode - Submit
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
        self.history = InMemoryHistory()

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
            # Create styled prompt using prompt_toolkit
            mode_indicator = " (single-line)" if self.multiline_mode is False else ""
            formatted_prompt = HTML(
                f"<ansiblue><b>{prompt}</b></ansiblue>{mode_indicator}: "
            )

            text = pt_prompt(
                formatted_prompt,
                history=self.history,
                mouse_support=False,
                wrap_lines=True,
            )

            # Handle special commands
            if text.strip().lower() == "/ml":
                self.multiline_mode = True
                self._show_mode_change("multi-line")
                return InputResult(
                    action=InputAction.TOGGLE_MULTILINE, multiline_toggled=True
                )
            elif text.strip().lower() == "/exit":
                return InputResult(action=InputAction.EXIT, should_exit=True)
            elif not text.strip():
                return InputResult(action=InputAction.CANCEL, cancelled=True)

            return InputResult(action=InputAction.SUBMIT, text=text)

        except (EOFError, KeyboardInterrupt):
            return InputResult(action=InputAction.EXIT, should_exit=True)

    def _get_multiline_input(self, prompt: str) -> InputResult:
        """Get multi-line input using prompt_toolkit multiline support."""
        try:
            self.console.print(
                "[dim]📝 Multi-line mode: Enter for new line, Ctrl+D to submit, "
                "/sl to switch[/dim]"
            )

            # Create custom key bindings for multiline mode
            bindings = KeyBindings()

            @bindings.add('c-d')
            def _(event):
                """Handle Ctrl+D to submit in multiline mode."""
                # Accept the input (submit)
                event.app.exit(result=event.app.current_buffer.text)

            # Create styled prompt for multi-line
            formatted_prompt = HTML(
                f"<ansiblue><b>{prompt}</b></ansiblue> (multi-line): "
            )

            text = pt_prompt(
                formatted_prompt,
                history=self.history,
                multiline=True,
                mouse_support=False,
                wrap_lines=True,
                key_bindings=bindings,
            )

            # Handle special commands
            if text.strip().lower() == "/sl":
                self.multiline_mode = False
                self._show_mode_change("single-line")
                return InputResult(
                    action=InputAction.TOGGLE_MULTILINE, multiline_toggled=True
                )
            elif text.strip().lower() == "/exit":
                return InputResult(action=InputAction.EXIT, should_exit=True)
            elif not text.strip():
                return InputResult(action=InputAction.CANCEL, cancelled=True)

            # Check if this is a command on the first line
            first_line = text.split("\n")[0].strip()
            if (
                first_line.startswith("/")
                and "(" in first_line
                and first_line.endswith(")")
            ):
                # This looks like a command on the first line - submit it immediately
                return InputResult(action=InputAction.SUBMIT, text=first_line)

            return InputResult(action=InputAction.SUBMIT, text=text.strip())

        except KeyboardInterrupt:
            return InputResult(action=InputAction.EXIT, should_exit=True)
        except EOFError:
            # In multiline mode, Ctrl+D is handled by key binding, so EOFError here
            # means empty input was submitted
            return InputResult(action=InputAction.CANCEL, cancelled=True)

    def _show_input_help(self):
        """Show help information about input modes."""
        help_text = """
**Input Modes:**
• **Single-line** (default): Press Enter to send message
• **Multi-line**: Press Enter to add new line, Ctrl+D to send

**Mode Switching:**
• Type `/ml` to switch to multi-line mode
• Type `/sl` to switch to single-line mode  
• Type `/exit` to exit chat

**Commands work in both modes** (e.g., `/help()`, `/attach()`)
• In multi-line mode, commands on the first line are submitted immediately
        """

        self.console.print(
            Panel(
                help_text.strip(),
                title="[bold cyan]💬 Input Help[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        self.console.print()

    def _show_mode_change(self, new_mode: str):
        """Show feedback when input mode changes."""
        emoji = "📝" if new_mode == "multi-line" else "✏️"
        self.console.print(f"[green]{emoji} Switched to {new_mode} input mode[/green]")

        if new_mode == "multi-line":
            self.console.print("[dim]Press Enter for new line, Ctrl+D to submit[/dim]")
        else:
            self.console.print("[dim]Press Enter to send, Ctrl+D to exit[/dim]")

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
