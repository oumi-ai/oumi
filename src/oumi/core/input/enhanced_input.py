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

from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel

# Import shared types from multiline_input to avoid duplication
from oumi.core.input.multiline_input import InputAction, InputResult


class CommandCompleter(Completer):
    """Custom completer for Oumi interactive commands."""

    def __init__(self):
        """Initialize the command completer."""
        # All available commands with their syntax
        self.commands = {
            # Basic commands
            "help": ["/help()"],
            "exit": ["/exit()"],
            # File operations
            "attach": ["/attach(file_path)"],
            "save": [
                "/save(output_path)",
                "/save(output.pdf)",
                "/save(output.json)",
                "/save(output.csv)",
                "/save(output.md)",
                "/save(output.html)",
            ],
            "import": [
                "/import(input_file)",
                "/import(data.csv)",
                "/import(chat.json)",
                "/import(conversation.xlsx)",
            ],
            # Conversation management
            "delete": ["/delete()"],
            "regen": ["/regen()"],
            "clear": ["/clear()"],
            "clear_thoughts": ["/clear_thoughts()"],
            "compact": ["/compact()"],
            # Generation parameters
            "set": [
                "/set(temperature=0.7)",
                "/set(top_p=0.9)",
                "/set(max_tokens=2048)",
                "/set(sampling=true)",
                "/set(seed=42)",
            ],
            # Branching
            "branch": ["/branch()", "/branch(branch_name)"],
            "switch": ["/switch(branch_name)", "/switch(main)", "/switch(branch_1)"],
            "branches": ["/branches()"],
            "branch_delete": ["/branch_delete(branch_name)"],
            # Thinking modes
            "full_thoughts": ["/full_thoughts()"],
            # Model management
            "swap": [
                "/swap(model_name)",
                "/swap(engine:model_name)",
                "/swap(config:path/to/config.yaml)",
            ],
            "list_engines": ["/list_engines()"],
            # Mode switching (handled by input system)
            "ml": ["/ml"],
            "sl": ["/sl"],
        }

        # Flatten all completions
        self.all_completions = []
        for cmd_list in self.commands.values():
            self.all_completions.extend(cmd_list)

    def get_completions(self, document: Document, complete_event):
        """Get completions for the current input."""
        text = document.text_before_cursor

        # Only complete if we're typing a command (starts with /)
        if not text.startswith("/"):
            return

        # Find matching completions
        matches = []
        for completion in self.all_completions:
            if completion.lower().startswith(text.lower()):
                # Exact prefix match - this is what we want
                matches.append((completion, 0))  # Priority 0 (highest)
            elif text.lower() in completion.lower() and len(text) > 2:
                # Partial match - lower priority
                matches.append((completion, 1))  # Priority 1 (lower)

        # Sort by priority, then alphabetically
        matches.sort(key=lambda x: (x[1], x[0]))

        # Yield completions
        for completion, priority in matches:
            if completion.lower().startswith(text.lower()):
                # For prefix matches, add only the remaining part
                completion_text = completion[len(text) :]
                yield Completion(completion_text, start_position=0)
            else:
                # For partial matches, replace the entire current text
                yield Completion(completion, start_position=-len(text))


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

        # Set up persistent command history
        self.history = self._setup_history()

        # Use custom command completer
        self.command_completer = CommandCompleter()

        # Key bindings for custom behavior
        self.bindings = self._create_key_bindings()

    def _setup_history(self):
        """Set up persistent command history."""
        try:
            # Create history directory in user's home
            history_dir = Path.home() / ".oumi"
            history_dir.mkdir(exist_ok=True)

            history_file = history_dir / "command_history"
            return FileHistory(str(history_file))
        except Exception:
            # Fallback to in-memory history if file system issues
            return InMemoryHistory()

    def _create_key_bindings(self) -> KeyBindings:
        """Create custom key bindings."""
        kb = KeyBindings()

        @kb.add("c-c")  # Ctrl+C
        def _(event):
            """Handle Ctrl+C to exit."""
            event.app.exit(exception=KeyboardInterrupt)

        # Don't bind Ctrl+D globally - let prompt_toolkit handle it naturally
        # In single-line mode, Ctrl+D will exit
        # In multi-line mode, Ctrl+D will submit (handled by prompt_toolkit)

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
            formatted_prompt = HTML(
                f"<ansiblue><b>{prompt_text}</b></ansiblue>{mode_indicator}: "
            )

            text = prompt(
                formatted_prompt,
                history=self.history,
                completer=self.command_completer,
                complete_while_typing=True,
                key_bindings=self.bindings,
                mouse_support=True,
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

    def _get_multiline_input(self, prompt_text: str) -> InputResult:
        """Get multi-line input with enhanced editing."""
        try:
            self.console.print(
                "[dim]📝 Multi-line mode: Enter for new line, Ctrl+D to submit, /sl to switch[/dim]"
            )

            # Create styled prompt for multi-line
            formatted_prompt = HTML(
                f"<ansiblue><b>{prompt_text}</b></ansiblue> (multi-line): "
            )

            text = prompt(
                formatted_prompt,
                history=self.history,
                completer=self.command_completer,
                multiline=True,
                complete_while_typing=True,
                key_bindings=self.bindings,
                mouse_support=True,
                wrap_lines=True,
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

            # Check if this is a command on the first line (for backward compatibility)
            first_line = text.split("\n")[0].strip()
            if (
                first_line.startswith("/")
                and "(" in first_line
                and first_line.endswith(")")
            ):
                return InputResult(action=InputAction.SUBMIT, text=first_line)

            return InputResult(action=InputAction.SUBMIT, text=text.strip())

        except (EOFError, KeyboardInterrupt):
            return InputResult(action=InputAction.EXIT, should_exit=True)

    def _show_input_help(self):
        """Show help information about enhanced input features."""
        help_text = """
**Enhanced Input Features:**
• **Arrow Keys**: ↑↓ for command history, ←→ for cursor movement
• **Tab Completion**: Tab to complete all 17 interactive commands with proper syntax
• **Exit**: Ctrl+C or Ctrl+D to exit chat

**Input Modes:**
• **Single-line** (default): Press Enter to send message, Ctrl+D to exit
• **Multi-line**: Press Enter for new line, Ctrl+D to submit message

**Mode Switching:**
• Type `/ml` to switch to multi-line mode
• Type `/sl` to switch to single-line mode
• Type `/exit` to exit chat

**Commands work in both modes** (e.g., `/help()`, `/attach()`, `/save()`)
• Tab completion shows command syntax and examples (try typing `/s` + Tab)
• Command history saved to ~/.oumi/command_history and persists across sessions
        """

        self.console.print(
            Panel(
                help_text.strip(),
                title="[bold cyan]💬 Enhanced Input Help[/bold cyan]",
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

    def add_to_history(self, text: str):
        """Add a command to the input history."""
        self.history.append_string(text)
