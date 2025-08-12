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

"""Macro operations command handler."""

import re

from rich.panel import Panel

from oumi.core.commands.base_handler import BaseCommandHandler, CommandResult
from oumi.core.commands.command_parser import ParsedCommand
from oumi.core.input.enhanced_input import EnhancedInput
from oumi.core.input.multiline_input import InputAction


class MacroOperationsHandler(BaseCommandHandler):
    """Handles macro-related commands: macro."""

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return ["macro"]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a macro operations command."""
        if command.command == "macro":
            return self._handle_macro(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unsupported command: {command.command}",
                should_continue=False,
            )

    def _handle_macro(self, command: ParsedCommand) -> CommandResult:
        """Handle the /macro(path) command to execute Jinja template-based macros."""
        try:
            if not self.context.macro_manager:
                return CommandResult(
                    success=False,
                    message="Macro functionality not available. Install jinja2: pip install jinja2",
                    should_continue=False,
                )

            if not command.args:
                return CommandResult(
                    success=False,
                    message="macro command requires a template path argument",
                    should_continue=False,
                )

            macro_path = command.args[0].strip()

            # Load and validate macro
            success, error_msg, macro_info = self.context.macro_manager.load_macro(
                macro_path
            )
            if not success:
                return CommandResult(
                    success=False,
                    message=f"Failed to load macro: {error_msg}",
                    should_continue=False,
                )

            # Display macro summary
            self._display_macro_summary(macro_info)

            # Collect field values if needed
            field_values = {}
            if macro_info.fields:
                self.console.print(
                    "\\n📝 Please provide values for the following fields:\\n"
                )
                self.console.print(
                    "[dim]Tip: For multiline input, type /ml to switch to multi-line mode, or paste content directly.[/dim]\\n"
                )

                for field in macro_info.fields:
                    field_value = self._collect_field_value(field)

                    # Check if user cancelled (empty return from required field cancellation)
                    if field.required and not field_value:
                        return CommandResult(
                            success=False,
                            message="Macro execution cancelled by user",
                            should_continue=False,
                        )

                    field_values[field.name] = field_value
                    self.console.print()  # Add spacing between fields

            # Render macro
            try:
                rendered_content = self.context.macro_manager.render_macro(
                    macro_info, field_values
                )

                # Validate rendered content doesn't exceed context window
                estimated_tokens = len(rendered_content) // 4  # Rough estimation
                max_context = getattr(self.config.model, "model_max_length", 4096)
                current_tokens = self._estimate_conversation_tokens()

                if (
                    current_tokens + estimated_tokens > max_context * 0.9
                ):  # Use 90% threshold
                    validation_error = (
                        f"Rendered macro content (~{estimated_tokens} tokens) would exceed "
                        f"context window limit. Current: {current_tokens}, Max: {max_context}"
                    )
                    return CommandResult(
                        success=False,
                        message=f"Rendered macro exceeds context window:\\n{validation_error}",
                        should_continue=False,
                    )

            except Exception as e:
                return CommandResult(
                    success=False,
                    message=f"Error rendering macro: {e}",
                    should_continue=False,
                )

            # Parse and execute macro conversation
            conversation_turns = self._parse_macro_turns(rendered_content)

            # Execute macro conversation
            success_msg = self._execute_macro_conversation(
                conversation_turns, macro_info
            )

            return CommandResult(
                success=True,
                message=success_msg,
                should_continue=True,
                user_input_override=conversation_turns[0]
                if conversation_turns
                else None,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error executing macro: {str(e)}",
                should_continue=False,
            )

    def _display_macro_summary(self, macro_info) -> None:
        """Display a summary of the loaded macro."""
        use_emoji = getattr(self._style, "use_emoji", True)

        # Create summary content
        summary_lines = [
            f"**Name:** {macro_info.name}",
            f"**Description:** {macro_info.description}",
            f"**Estimated turns:** {macro_info.turns}",
            f"**Fields to fill:** {len(macro_info.fields)}",
        ]

        if macro_info.fields:
            summary_lines.append("\\n**Field Details:**")
            for field in macro_info.fields:
                field_desc = field.description or "No description"
                required_text = "Required" if field.required else "Optional"
                summary_lines.append(
                    f"  • `{field.name}`: {field_desc} ({required_text})"
                )

        # Display the macro summary
        title = "🎯 Macro Summary" if use_emoji else "Macro Summary"
        content = "\\n".join(summary_lines)

        panel = Panel(
            content,
            title=title,
            border_style=getattr(self._style, "assistant_border_style", "cyan"),
        )
        self.console.print(panel)

    def _collect_field_value(self, field) -> str:
        """Interactively collect a value for a macro field using EnhancedInput.

        This method uses the same input system as the main chat loop to ensure
        consistent handling of complex multiline content and avoid terminal state conflicts.
        """
        # Create enhanced input handler with consistent styling
        field_input = EnhancedInput(self.console, "bold cyan")

        # Build descriptive prompt text
        prompt_parts = [f"[bold cyan]{field.name}[/bold cyan]"]
        if field.description:
            prompt_parts.append(f"({field.description})")
        if field.placeholder:
            prompt_parts.append(f"[dim]{field.placeholder}[/dim]")

        prompt_text = " ".join(prompt_parts)

        while True:
            try:
                # Display the field prompt
                self.console.print(f"{prompt_text}: ", end="")

                # Get input using the same system as main chat loop
                input_result = field_input.get_input("")

                # Handle different input actions
                if input_result.should_exit:
                    self.console.print("\\n[yellow]Macro execution cancelled[/yellow]")
                    return ""  # Signal cancellation

                if input_result.cancelled:
                    self.console.print("[yellow]Input cancelled[/yellow]")
                    continue  # Retry input

                if input_result.multiline_toggled:
                    # Mode was toggled, get input again
                    continue

                if input_result.action != InputAction.SUBMIT:
                    # Handle other actions that don't submit input
                    continue

                # Process the submitted input
                value = input_result.text.strip() if input_result.text else ""

                # Handle empty input for optional fields with placeholder
                if not value and not field.required and field.placeholder:
                    value = field.placeholder
                    self.console.print(f"[dim]Using default: {value}[/dim]")

                # Validate required fields
                if field.required and not value:
                    self.console.print(
                        "[red]This field is required! Please enter a value.[/red]"
                    )
                    continue  # Retry input

                return value

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                self.console.print("\\n[red]Macro execution cancelled[/red]")
                return ""  # Signal cancellation

    def _parse_macro_turns(self, rendered_content: str) -> list[str]:
        """Parse rendered macro content into conversation turns."""
        turns = []

        # Try to detect conversation structure
        # Look for patterns like "User:", "Human:", etc.

        # Split by common role indicators
        role_patterns = [
            r"^(User|Human):\\s*(.+?)(?=^(?:Assistant|AI|Bot|User|Human):|$)",
            r"^(Assistant|AI|Bot):\\s*(.+?)(?=^(?:User|Human|Assistant|AI|Bot):|$)",
        ]

        current_turn = ""
        in_user_section = False

        lines = rendered_content.split("\\n")
        for line in lines:
            line = line.strip()

            # Check for role indicators
            if re.match(r"^(User|Human):", line, re.IGNORECASE):
                # Save previous turn if exists
                if current_turn and in_user_section:
                    turns.append(current_turn.strip())

                # Start new user turn
                current_turn = re.sub(
                    r"^(User|Human):\\s*", "", line, flags=re.IGNORECASE
                )
                in_user_section = True

            elif re.match(r"^(Assistant|AI|Bot):", line, re.IGNORECASE):
                # Save previous user turn if exists
                if current_turn and in_user_section:
                    turns.append(current_turn.strip())

                # Assistant responses are not added as new turns for execution
                current_turn = ""
                in_user_section = False

            else:
                # Continue current section
                if in_user_section and line:
                    if current_turn:
                        current_turn += "\\n" + line
                    else:
                        current_turn = line

        # Add final turn if we're in a user section
        if current_turn and in_user_section:
            turns.append(current_turn.strip())

        # If no structured turns found, treat entire content as single turn
        if not turns and rendered_content.strip():
            turns = [rendered_content.strip()]

        return turns

    def _execute_macro_conversation(
        self, conversation_turns: list[str], macro_info
    ) -> str:
        """Execute a multi-turn macro conversation."""
        if not conversation_turns:
            return "No conversation turns found in macro"

        # For now, we'll execute just the first turn and let the normal flow handle it
        # Multi-turn execution would require more complex state management

        turns_count = len(conversation_turns)
        if turns_count == 1:
            return f"Executed macro '{macro_info.name}' with 1 conversation turn"
        else:
            return (
                f"Executing macro '{macro_info.name}' - starting with turn 1 of {turns_count}.\\n"
                f"Note: Multi-turn macros require manual continuation for subsequent turns."
            )
