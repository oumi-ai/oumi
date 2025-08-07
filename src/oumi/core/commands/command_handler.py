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

"""Command handler for interactive Oumi inference commands."""

from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from oumi.core.attachments import ContextWindowManager, FileHandler
from oumi.core.commands.command_parser import ParsedCommand
from oumi.core.commands.compaction_engine import CompactionEngine
from oumi.core.commands.conversation_branches import ConversationBranchManager
from oumi.core.configs import InferenceConfig
from oumi.core.inference import BaseInferenceEngine


class CommandResult:
    """Result of executing a command.

    Attributes:
        success: Whether the command executed successfully.
        message: Optional message to display to the user.
        should_exit: Whether the chat session should exit.
        should_continue: Whether to continue processing (vs. skip inference).
        user_input_override: Override input text for inference (used by /regen).
    """

    def __init__(
        self,
        success: bool = True,
        message: Optional[str] = None,
        should_exit: bool = False,
        should_continue: bool = True,
        user_input_override: Optional[str] = None,
    ):
        self.success = success
        self.message = message
        self.should_exit = should_exit
        self.should_continue = should_continue
        self.user_input_override = user_input_override


class CommandHandler:
    """Handles execution of interactive commands in Oumi inference.

    This class processes parsed commands and executes the appropriate actions,
    such as displaying help, exiting the session, or managing conversation state.
    """

    def __init__(
        self,
        console: Console,
        config: InferenceConfig,
        conversation_history: list,
        inference_engine: BaseInferenceEngine,
    ):
        """Initialize the command handler.

        Args:
            console: Rich console for output.
            config: Inference configuration.
            conversation_history: List of conversation messages.
            inference_engine: The inference engine being used.
        """
        self.console = console
        self.config = config
        self.conversation_history = conversation_history
        self.inference_engine = inference_engine
        self._style = config.style

        # Initialize file attachment system
        # Try multiple possible attribute names for max context length
        max_context = None
        possible_context_attrs = [
            "model_max_length",  # Standard transformers
            "max_model_len",  # VLLM
            "max_tokens",  # Some configs
            "context_length",  # Alternative name
            "max_context_len",  # Another alternative
            "max_seq_len",  # Sequence length
        ]

        for attr in possible_context_attrs:
            max_context = getattr(config.model, attr, None)
            if max_context is not None:
                # Debug info - this will help us see which attribute was found
                print(f"[DEBUG] Found context length via '{attr}': {max_context}")
                break

        # Final fallback to 4096 if nothing found
        if max_context is None:
            max_context = 4096

        model_name = getattr(config.model, "model_name", "default")

        context_manager = ContextWindowManager(
            max_context_length=max_context, model_name=model_name
        )
        self.file_handler = FileHandler(context_manager)

        # Initialize compaction engine
        self.compaction_engine = CompactionEngine(inference_engine, config.model)

        # Initialize branch manager
        self.branch_manager = ConversationBranchManager()
        # Set the main branch's conversation history to use our reference
        self.branch_manager.branches["main"].conversation_history = conversation_history

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a parsed command and return the result.

        Args:
            command: The parsed command to execute.

        Returns:
            CommandResult indicating the outcome.
        """
        try:
            # Route to specific handler based on command name
            if command.command == "help":
                return self._handle_help(command)
            elif command.command == "exit":
                return self._handle_exit(command)
            elif command.command == "attach":
                return self._handle_attach(command)
            elif command.command == "delete":
                return self._handle_delete(command)
            elif command.command == "regen":
                return self._handle_regen(command)
            elif command.command == "save":
                return self._handle_save(command)
            elif command.command == "set":
                return self._handle_set(command)
            elif command.command == "compact":
                return self._handle_compact(command)
            elif command.command == "branch":
                return self._handle_branch(command)
            elif command.command == "switch":
                return self._handle_switch(command)
            elif command.command == "branches":
                return self._handle_branches(command)
            elif command.command == "branch_delete":
                return self._handle_branch_delete(command)
            else:
                return CommandResult(
                    success=False,
                    message=f"Unknown command: '{command.command}'",
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error executing command '{command.command}': {str(e)}",
            )

    def _handle_help(self, command: ParsedCommand) -> CommandResult:
        """Handle the /help() command."""
        help_content = self._generate_help_content()

        # Get style attributes with fallbacks
        title_style = getattr(self._style, "assistant_title_style", "bold cyan")
        border_style = getattr(self._style, "assistant_border_style", "cyan")
        padding = getattr(self._style, "assistant_padding", (1, 2))
        expand = getattr(self._style, "expand_panels", False)
        use_emoji = getattr(self._style, "use_emoji", True)

        emoji = "📋 " if use_emoji else ""

        # Display help using styled panel
        self.console.print(
            Panel(
                Markdown(help_content),
                title=f"[{title_style}]{emoji}Oumi Interactive Commands[/{title_style}]",
                border_style=border_style,
                padding=padding,
                expand=expand,
            )
        )

        return CommandResult(success=True, should_continue=False)

    def _handle_exit(self, command: ParsedCommand) -> CommandResult:
        """Handle the /exit() command."""
        # Get style attributes with fallbacks
        use_emoji = getattr(self._style, "use_emoji", True)
        custom_theme = getattr(self._style, "custom_theme", None)

        # Display goodbye message
        emoji = "👋 " if use_emoji else ""
        goodbye_style = (
            custom_theme.get("warning", "yellow") if custom_theme else "yellow"
        )

        self.console.print(f"\n[{goodbye_style}]{emoji}Goodbye![/{goodbye_style}]")

        return CommandResult(success=True, should_exit=True, should_continue=False)

    def _handle_attach(self, command: ParsedCommand) -> CommandResult:
        """Handle the /attach(path) command."""
        if not command.args:
            return CommandResult(
                success=False,
                message="attach command requires a file path argument",
                should_continue=False,
            )

        file_path = command.args[0].strip()

        try:
            # Estimate current conversation tokens
            conversation_tokens = self._estimate_conversation_tokens()

            # Process the file
            attachment_result = self.file_handler.attach_file(
                file_path, conversation_tokens
            )

            if attachment_result.success:
                # Display attachment info
                self._display_attachment_result(attachment_result)

                # Add content to conversation (this will be used in the next inference)
                self._add_attachment_to_conversation(attachment_result)

                success_message = f"Attached {attachment_result.file_info.name}"
                if attachment_result.context_info:
                    success_message += f" - {attachment_result.context_info}"

                return CommandResult(
                    success=True, message=success_message, should_continue=False
                )
            else:
                return CommandResult(
                    success=False,
                    message=attachment_result.warning_message
                    or f"Failed to attach {file_path}",
                    should_continue=False,
                )

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return CommandResult(
                success=False,
                message=f"Error attaching file: {str(e)}\n\nTraceback:\n{error_details}",
                should_continue=False,
            )

    def _handle_delete(self, command: ParsedCommand) -> CommandResult:
        """Handle the /delete() command to remove the last conversation turn."""
        try:
            if not self.conversation_history:
                return CommandResult(
                    success=False,
                    message="No conversation history to delete",
                    should_continue=False,
                )

            # Find and remove the last complete turn (user + assistant pair)
            deleted_count = self._delete_last_turn()

            if deleted_count == 0:
                return CommandResult(
                    success=False,
                    message="No complete conversation turn found to delete",
                    should_continue=False,
                )

            # Show success message
            turn_word = "turn" if deleted_count == 2 else "message"
            success_message = (
                f"Deleted last conversation {turn_word} ({deleted_count} messages)"
            )

            self._display_delete_success(success_message)

            return CommandResult(
                success=True, message=success_message, should_continue=False
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error deleting conversation turn: {str(e)}",
                should_continue=False,
            )

    def _handle_regen(self, command: ParsedCommand) -> CommandResult:
        """Handle the /regen() command to regenerate the last assistant response."""
        try:
            if not self.conversation_history:
                return CommandResult(
                    success=False,
                    message="No conversation history to regenerate from",
                    should_continue=False,
                )

            # Get the last user message for regeneration
            last_user_input = self._get_last_user_input()

            if not last_user_input:
                return CommandResult(
                    success=False,
                    message="No user message found to regenerate response for",
                    should_continue=False,
                )

            # Remove the last assistant response if it exists
            self._remove_last_assistant_response()

            # Show regeneration message
            self._display_regen_status()

            # Store the user input for inference and signal continuation
            return CommandResult(
                success=True,
                message=f"Regenerating response for: {last_user_input[:50]}...",
                should_continue=True,
                user_input_override=last_user_input,  # This will be used by the main loop
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error regenerating response: {str(e)}",
                should_continue=False,
            )

    def _handle_save(self, command: ParsedCommand) -> CommandResult:
        """Handle the /save(path) command to export conversation to PDF."""
        if not command.args:
            return CommandResult(
                success=False,
                message="save command requires a file path argument",
                should_continue=False,
            )

        file_path = command.args[0].strip()

        # Ensure .pdf extension
        if not file_path.lower().endswith(".pdf"):
            file_path += ".pdf"

        try:
            success, message = self._export_conversation_to_pdf(file_path)
            return CommandResult(
                success=success, message=message, should_continue=False
            )
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return CommandResult(
                success=False,
                message=f"Error saving conversation: {str(e)}\n\nTraceback:\n{error_details}",
                should_continue=False,
            )

    def _handle_set(self, command: ParsedCommand) -> CommandResult:
        """Handle the /set(param=value) command to adjust generation parameters."""
        if not command.kwargs and not command.args:
            return CommandResult(
                success=False,
                message="set command requires parameter=value arguments (e.g., /set(temperature=0.8, top_p=0.9))",
                should_continue=False,
            )

        # Track successful parameter changes
        changes_made = []
        errors = []

        # Process keyword arguments (preferred method)
        for param, value in command.kwargs.items():
            success, message = self._update_parameter(param, value)
            if success:
                changes_made.append(message)
            else:
                errors.append(message)

        # Process positional arguments in param=value format (fallback)
        for arg in command.args:
            if "=" not in arg:
                errors.append(f"Invalid format: '{arg}' (expected param=value)")
                continue

            param, value = arg.split("=", 1)
            param = param.strip()
            value = value.strip()

            success, message = self._update_parameter(param, value)
            if success:
                changes_made.append(message)
            else:
                errors.append(message)

        # Prepare result message
        if changes_made and not errors:
            # All changes successful
            message = "✅ " + "\n".join(changes_made)
            return CommandResult(success=True, message=message, should_continue=False)
        elif changes_made and errors:
            # Partial success
            message = (
                "⚠️ Partial success:\n✅ "
                + "\n".join(changes_made)
                + "\n❌ "
                + "\n".join(errors)
            )
            return CommandResult(success=True, message=message, should_continue=False)
        else:
            # All failed
            message = "❌ " + "\n".join(errors)
            return CommandResult(success=False, message=message, should_continue=False)

    def _update_parameter(self, param: str, value: str) -> tuple[bool, str]:
        """Update a generation parameter.

        Returns:
            Tuple of (success, message)
        """
        try:
            # Map common parameter names to config paths
            param_map = {
                "temperature": ("generation", "temperature", float),
                "top_p": ("generation", "top_p", float),
                "top_k": ("generation", "top_k", int),
                "max_tokens": ("generation", "max_new_tokens", int),
                "max_new_tokens": ("generation", "max_new_tokens", int),
                "sampling": (
                    "generation",
                    "use_sampling",
                    lambda x: x.lower() in ["true", "1", "yes", "on"],
                ),
                "seed": (
                    "generation",
                    "seed",
                    lambda x: int(x) if x.lower() != "none" else None,
                ),
                "frequency_penalty": ("generation", "frequency_penalty", float),
                "presence_penalty": ("generation", "presence_penalty", float),
                "min_p": ("generation", "min_p", float),
                "num_beams": ("generation", "num_beams", int),
            }

            if param not in param_map:
                available_params = ", ".join(sorted(param_map.keys()))
                return (
                    False,
                    f"Unknown parameter '{param}'. Available: {available_params}",
                )

            section, attr_name, converter = param_map[param]

            # Convert value
            try:
                converted_value = converter(value)
            except (ValueError, AttributeError):
                return (
                    False,
                    f"Invalid value for '{param}': '{value}' (expected {converter.__name__})",
                )

            # Validate ranges for certain parameters
            validation_errors = self._validate_parameter_value(param, converted_value)
            if validation_errors:
                return False, validation_errors

            # Update the config
            config_section = getattr(self.config, section)
            old_value = getattr(config_section, attr_name)
            setattr(config_section, attr_name, converted_value)

            return True, f"Set {param}: {old_value} → {converted_value}"

        except Exception as e:
            return False, f"Error updating {param}: {str(e)}"

    def _validate_parameter_value(self, param: str, value) -> Optional[str]:
        """Validate parameter values are within acceptable ranges."""
        if param == "temperature" and (value < 0 or value > 2.0):
            return f"temperature must be between 0.0 and 2.0, got {value}"
        elif param == "top_p" and (value <= 0 or value > 1.0):
            return f"top_p must be between 0.0 and 1.0, got {value}"
        elif param == "top_k" and value < 0:
            return f"top_k must be >= 0, got {value}"
        elif param in ["max_tokens", "max_new_tokens"] and value <= 0:
            return f"{param} must be > 0, got {value}"
        elif param in ["frequency_penalty", "presence_penalty"] and (
            value < -2.0 or value > 2.0
        ):
            return f"{param} must be between -2.0 and 2.0, got {value}"
        elif param == "min_p" and (value < 0 or value > 1.0):
            return f"min_p must be between 0.0 and 1.0, got {value}"
        elif param == "num_beams" and value < 1:
            return f"num_beams must be >= 1, got {value}"

        return None

    def _export_conversation_to_pdf(self, file_path: str) -> tuple[bool, str]:
        """Export conversation to PDF.

        Returns:
            Tuple of (success, message)
        """
        try:
            # Try to import reportlab for PDF generation
            try:
                from reportlab.lib.colors import black, blue, green
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
                from reportlab.lib.units import inch
                from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

                have_reportlab = True
            except ImportError:
                have_reportlab = False

            if not have_reportlab:
                # Fallback to plain text export
                return self._export_conversation_to_text(
                    file_path.replace(".pdf", ".txt")
                )

            if not self.conversation_history:
                return False, "No conversation history to export"

            # Create PDF document
            doc = SimpleDocTemplate(
                file_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )

            # Get styles
            styles = getSampleStyleSheet()

            # Create custom styles
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=18,
                textColor=blue,
                spaceAfter=30,
            )

            user_header_style = ParagraphStyle(
                "UserHeader",
                parent=styles["Heading2"],
                fontSize=14,
                textColor=green,
                spaceBefore=12,
                spaceAfter=6,
            )

            assistant_header_style = ParagraphStyle(
                "AssistantHeader",
                parent=styles["Heading2"],
                fontSize=14,
                textColor=blue,
                spaceBefore=12,
                spaceAfter=6,
            )

            content_style = ParagraphStyle(
                "Content",
                parent=styles["Normal"],
                fontSize=11,
                spaceAfter=12,
                leftIndent=20,
            )

            # Build story
            story = []

            # Title
            from datetime import datetime

            model_name = getattr(self.config.model, "model_name", "Unknown Model")
            title_text = f"Oumi Chat Export - {model_name}"
            story.append(Paragraph(title_text, title_style))

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            story.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
            story.append(Spacer(1, 20))

            # Add conversation turns
            for i, msg in enumerate(self.conversation_history):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Skip attachment markers
                if role == "attachment":
                    continue

                # Format content for PDF (escape HTML)
                content = content.replace("<", "&lt;").replace(">", "&gt;")

                if role == "user":
                    emoji = "🧑 " if self._style.use_emoji else ""
                    story.append(Paragraph(f"{emoji}User:", user_header_style))
                elif role == "assistant":
                    emoji = "🤖 " if self._style.use_emoji else ""
                    story.append(
                        Paragraph(f"{emoji}Assistant:", assistant_header_style)
                    )
                else:
                    story.append(Paragraph(f"{role.title()}:", styles["Heading2"]))

                # Split long content into paragraphs
                paragraphs = content.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), content_style))

                # Add spacing between turns
                if i < len(self.conversation_history) - 1:
                    story.append(Spacer(1, 12))

            # Build PDF
            doc.build(story)

            # Count messages
            user_msgs = len(
                [m for m in self.conversation_history if m.get("role") == "user"]
            )
            assistant_msgs = len(
                [m for m in self.conversation_history if m.get("role") == "assistant"]
            )

            return (
                True,
                f"✅ Exported conversation to {file_path} ({user_msgs} user messages, {assistant_msgs} assistant responses)",
            )

        except Exception as e:
            return False, f"Failed to export PDF: {str(e)}"

    def _export_conversation_to_text(self, file_path: str) -> tuple[bool, str]:
        """Fallback: Export conversation to plain text when reportlab is not available."""
        try:
            if not self.conversation_history:
                return False, "No conversation history to export"

            with open(file_path, "w", encoding="utf-8") as f:
                # Header
                from datetime import datetime

                model_name = getattr(self.config.model, "model_name", "Unknown Model")
                f.write(f"Oumi Chat Export - {model_name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

                # Conversation turns
                for i, msg in enumerate(self.conversation_history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    # Skip attachment markers
                    if role == "attachment":
                        continue

                    if role == "user":
                        f.write("👤 USER:\n")
                    elif role == "assistant":
                        f.write("🤖 ASSISTANT:\n")
                    else:
                        f.write(f"{role.upper()}:\n")

                    f.write(f"{content}\n\n")

                    if i < len(self.conversation_history) - 1:
                        f.write("-" * 30 + "\n\n")

            # Count messages
            user_msgs = len(
                [m for m in self.conversation_history if m.get("role") == "user"]
            )
            assistant_msgs = len(
                [m for m in self.conversation_history if m.get("role") == "assistant"]
            )

            return (
                True,
                f"✅ Exported conversation to {file_path} (text format - install reportlab for PDF) ({user_msgs} user messages, {assistant_msgs} assistant responses)",
            )

        except Exception as e:
            return False, f"Failed to export text file: {str(e)}"

    def _show_not_implemented_message(self, command_name: str, phase: str):
        """Show a styled message for not-yet-implemented commands."""
        message = f"Command '/{command_name}()' is planned for {phase} implementation."

        self.console.print(
            Panel(
                Text(message, style="dim white"),
                title=f"[{self._style.error_title_style}]🚧 Coming Soon[/{self._style.error_title_style}]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    def _generate_help_content(self) -> str:
        """Generate the help content as Markdown."""
        help_content = f"""
## Available Commands

### Basic Commands
- **`/help()`** - Show this help message
- **`/exit()`** - Exit the interactive chat session

### Input Modes
- **`/ml`** - Switch to multi-line input mode
- **`/sl`** - Switch to single-line input mode

### File Operations
- **`/attach(path)`** - Attach files to conversation
  - Supports: images (JPG, PNG, etc.), PDFs, text files, CSV, JSON, Markdown
  - Example: `/attach(document.pdf)` or `/attach(image.jpg)`

### Conversation Management
- **`/delete()`** - Delete the previous conversation turn
- **`/regen()`** - Regenerate the last assistant response

### Parameter Adjustment
- **`/set(param=value)`** - Adjust generation parameters
  - Examples:
    - `/set(temperature=0.8)` - More creative responses
    - `/set(top_p=0.9)` - Nucleus sampling
    - `/set(max_tokens=2048)` - Longer responses
    - `/set(sampling=true)` - Enable sampling
  - Available parameters: temperature, top_p, top_k, max_tokens, sampling, seed, frequency_penalty, presence_penalty, min_p, num_beams

### Export
- **`/save(path)`** - Save conversation to PDF (or text if reportlab not available)
  - Example: `/save(chat_history.pdf)` or `/save(my_chat)`
  - Exports formatted conversation with timestamps and role indicators

### Context Management
- **`/compact()`** - Compress conversation history to save context window space
  - Summarizes older messages while preserving recent exchanges
  - Helps when approaching context window limits
  - Shows token savings after compaction

### Conversation Branching (TMux-style)
- **`/branch()`** - Create a new conversation branch from current point
  - Fork the conversation to explore different paths
  - Maximum of 5 branches allowed
- **`/switch(name)`** - Switch to a different conversation branch
  - Example: `/switch(main)` or `/switch(branch_1)`
- **`/branches()`** - List all conversation branches
  - Shows branch names, creation time, and message preview
- **`/branch_delete(name)`** - Delete a conversation branch
  - Example: `/branch_delete(branch_2)`
  - Cannot delete the main branch

## Input Modes

### Single-line Mode (Default)
- Press **Enter** to send your message
- Type `/ml` to switch to multi-line mode

### Multi-line Mode
- Press **Enter** to add a new line
- Press **Enter** on an empty line to send
- Type `/sl` to switch back to single-line mode

## Usage Notes
- Commands must start with `/` and be the first thing in your message
- Use parentheses for arguments: `/command(arg1, arg2)`
- Use `key=value` format for parameters: `/set(temperature=0.8)`
- Commands work in both input modes
- Commands are case-insensitive

## Examples
```
/help()
/ml                        # Switch to multi-line mode
/exit()
/attach(my_document.pdf)
/set(temperature=0.7, top_p=0.9)
/save(conversation.pdf)
```

{"🎨 **Tip**: You can customize the appearance with different style themes in your config!" if getattr(self._style, "use_emoji", True) else "Tip: You can customize the appearance with different style themes in your config!"}
        """

        return help_content.strip()

    def display_command_error(self, error_message: str):
        """Display a command error message with styling."""
        error_style = getattr(self._style, "error_style", "red")
        error_title_style = getattr(self._style, "error_title_style", "bold red")
        error_border_style = getattr(self._style, "error_border_style", "red")
        expand = getattr(self._style, "expand_panels", False)

        self.console.print(
            Panel(
                f"[{error_style}]{error_message}[/{error_style}]",
                title=f"[{error_title_style}]❌ Command Error[/{error_title_style}]",
                border_style=error_border_style,
                expand=expand,
            )
        )

    def display_command_success(self, message: str):
        """Display a command success message with styling."""
        custom_theme = getattr(self._style, "custom_theme", None)
        use_emoji = getattr(self._style, "use_emoji", True)

        success_style = (
            custom_theme.get("success", "green") if custom_theme else "green"
        )

        emoji = "✅ " if use_emoji else ""

        self.console.print(
            Panel(
                f"[{success_style}]{emoji}{message}[/{success_style}]",
                title=f"[bold {success_style}]Command Success[/bold {success_style}]",
                border_style=success_style,
                padding=(0, 1),
            )
        )

    def _estimate_conversation_tokens(self) -> int:
        """Estimate the token count of the current conversation history."""
        if not self.conversation_history:
            return 0

        # Simple estimation: combine all conversation text
        text_content = ""
        for message in self.conversation_history:
            if isinstance(message, dict) and "content" in message:
                text_content += str(message["content"]) + "\n"

        # Use context manager to estimate tokens
        max_context = getattr(self.config.model, "model_max_length", 4096)
        model_name = getattr(self.config.model, "model_name", "default")
        context_manager = ContextWindowManager(max_context, model_name)

        return context_manager.estimate_tokens(text_content)

    def _display_attachment_result(self, result):
        """Display the attachment result with rich formatting."""
        # Choose appropriate emoji and color based on file type
        file_type = result.file_info.file_type.value
        emoji_map = {
            "image": "📷",
            "pdf": "📄",
            "text": "📝",
            "csv": "📊",
            "json": "🔧",
            "markdown": "📝",
            "code": "💾",
            "unknown": "❓",
        }
        emoji = emoji_map.get(file_type, "📎")

        # Create display content
        size_mb = result.file_info.size_bytes / (1024 * 1024)
        content = f"{emoji} **{result.file_info.name}**\n"
        content += f"Type: {file_type.title()} ({size_mb:.2f} MB)\n"

        if result.context_info:
            content += f"Processing: {result.context_info}\n"

        if result.warning_message:
            content += f"\n⚠️  {result.warning_message}"

        # Choose border color based on success/warning
        border_color = "yellow" if result.warning_message else "green"
        title_style = f"bold {border_color}"

        self.console.print(
            Panel(
                content,
                title=f"[{title_style}]📎 File Attached[/{title_style}]",
                border_style=border_color,
                padding=(0, 1),
            )
        )

    def _add_attachment_to_conversation(self, result):
        """Add attachment content to conversation history for next inference."""
        # Simply add the text content to conversation history
        # This will be prepended to the next user message
        attachment_entry = {
            "role": "attachment",
            "file_name": result.file_info.name,
            "file_type": result.file_info.file_type.value,
            "text_content": result.text_content,
            "processing_strategy": result.file_info.processing_strategy.value,
        }

        # Store in conversation history for reference
        self.conversation_history.append(attachment_entry)

    # Conversation management helper methods

    def _delete_last_turn(self) -> int:
        """Delete the last complete conversation turn.

        Returns:
            int: Number of messages deleted (0, 1, or 2).
        """
        if not self.conversation_history:
            return 0

        deleted_count = 0

        # Remove the last assistant message if it exists
        if (
            self.conversation_history
            and self.conversation_history[-1].get("role") == "assistant"
        ):
            self.conversation_history.pop()
            deleted_count += 1

        # Remove the last user message if it exists
        if (
            self.conversation_history
            and self.conversation_history[-1].get("role") == "user"
        ):
            self.conversation_history.pop()
            deleted_count += 1

        return deleted_count

    def _get_last_user_input(self) -> Optional[str]:
        """Get the last user input from conversation history.

        Returns:
            The last user message content, or None if not found.
        """
        # Go through history backwards to find the last user message
        for entry in reversed(self.conversation_history):
            if entry.get("role") == "user":
                return entry.get("content", "")
        return None

    def _remove_last_assistant_response(self) -> bool:
        """Remove the last assistant response from conversation history.

        Returns:
            True if a response was removed, False otherwise.
        """
        if (
            self.conversation_history
            and self.conversation_history[-1].get("role") == "assistant"
        ):
            self.conversation_history.pop()
            return True
        return False

    def _display_delete_success(self, message: str):
        """Display success message for delete operation."""
        # Get style attributes with fallbacks
        success_style = getattr(self._style, "success_style", "bold green")
        title_style = getattr(self._style, "assistant_title_style", "bold cyan")
        border_style = getattr(self._style, "assistant_border_style", "green")
        use_emoji = getattr(self._style, "use_emoji", True)
        expand = getattr(self._style, "expand_panels", False)

        emoji = "🗑️ " if use_emoji else ""

        self.console.print(
            Panel(
                Text(message, style=success_style),
                title=f"[{title_style}]{emoji}Conversation Updated[/{title_style}]",
                border_style=border_style,
                padding=(0, 1),
                expand=expand,
            )
        )

    def _display_regen_status(self):
        """Display status message for regeneration operation."""
        # Get style attributes with fallbacks
        status_style = getattr(self._style, "status_style", "yellow")
        title_style = getattr(self._style, "assistant_title_style", "bold cyan")
        border_style = getattr(self._style, "status_border_style", "yellow")
        use_emoji = getattr(self._style, "use_emoji", True)
        expand = getattr(self._style, "expand_panels", False)

        emoji = "🔄 " if use_emoji else ""

        self.console.print(
            Panel(
                Text("Regenerating last response...", style=status_style),
                title=f"[{title_style}]{emoji}Regenerating[/{title_style}]",
                border_style=border_style,
                padding=(0, 1),
                expand=expand,
            )
        )

    def _handle_compact(self, command: ParsedCommand) -> CommandResult:
        """Handle the /compact() command to compress conversation history."""
        try:
            # Show compaction status
            self._display_compact_status("Analyzing conversation history...")

            # Get original token count
            original_stats = self.compaction_engine.estimate_token_reduction(
                self.conversation_history, self.conversation_history
            )

            # Perform compaction
            compacted_history, summary = self.compaction_engine.compact_conversation(
                self.conversation_history,
                preserve_recent=2,  # Keep last 2 turns
            )

            if not summary:
                return CommandResult(
                    success=False,
                    message="Failed to generate conversation summary",
                    should_continue=False,
                )

            # Get compacted stats
            compacted_stats = self.compaction_engine.estimate_token_reduction(
                self.conversation_history, compacted_history
            )

            # Replace conversation history with compacted version
            self.conversation_history.clear()
            self.conversation_history.extend(compacted_history)

            # Display results
            self._display_compact_results(original_stats, compacted_stats, summary)

            return CommandResult(
                success=True,
                message=f"Compacted conversation: {compacted_stats['tokens_saved']} tokens saved ({compacted_stats['reduction_percent']:.1f}% reduction)",
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error compacting conversation: {str(e)}",
                should_continue=False,
            )

    def _display_compact_status(self, message: str):
        """Display status message for compaction operation."""
        # Get style attributes with fallbacks
        status_style = getattr(self._style, "status_style", "yellow")
        title_style = getattr(self._style, "assistant_title_style", "bold cyan")
        border_style = getattr(self._style, "status_border_style", "yellow")
        use_emoji = getattr(self._style, "use_emoji", True)
        expand = getattr(self._style, "expand_panels", False)

        emoji = "🗜️ " if use_emoji else ""

        self.console.print(
            Panel(
                Text(message, style=status_style),
                title=f"[{title_style}]{emoji}Compacting Context[/{title_style}]",
                border_style=border_style,
                padding=(0, 1),
                expand=expand,
            )
        )

    def _display_compact_results(
        self, original_stats: dict, compacted_stats: dict, summary: str
    ):
        """Display the results of conversation compaction."""
        # Get style attributes
        use_emoji = getattr(self._style, "use_emoji", True)
        title_style = getattr(self._style, "assistant_title_style", "bold cyan")
        border_style = getattr(self._style, "assistant_border_style", "cyan")
        expand = getattr(self._style, "expand_panels", False)

        emoji = "📊 " if use_emoji else ""

        # Create results text
        results_text = f"""**Compaction Results:**

Original: {original_stats["original_tokens"]} tokens
Compacted: {compacted_stats["compacted_tokens"]} tokens
Saved: {compacted_stats["tokens_saved"]} tokens ({compacted_stats["reduction_percent"]:.1f}% reduction)

**Summary of compressed content:**
{summary[:500]}{"..." if len(summary) > 500 else ""}"""

        self.console.print(
            Panel(
                Markdown(results_text),
                title=f"[{title_style}]{emoji}Context Compacted[/{title_style}]",
                border_style=border_style,
                padding=(1, 2),
                expand=expand,
            )
        )

    def _handle_branch(self, command: ParsedCommand) -> CommandResult:
        """Handle the /branch() command to create a new conversation branch."""
        try:
            # Get optional name from args
            name = command.args[0] if command.args else None

            # Create branch from current branch
            success, message, new_branch = self.branch_manager.create_branch(
                from_branch_id=self.branch_manager.current_branch_id, name=name
            )

            if success and new_branch:
                # Display success message with branch info
                self._display_branch_created(new_branch)
                return CommandResult(
                    success=True,
                    message=f"Created branch '{new_branch.name}' (ID: {new_branch.id})",
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False, message=message, should_continue=False
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error creating branch: {str(e)}",
                should_continue=False,
            )

    def _handle_switch(self, command: ParsedCommand) -> CommandResult:
        """Handle the /switch(branch_name) command to switch branches."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="switch command requires a branch name or ID",
                    should_continue=False,
                )

            branch_identifier = command.args[0]

            # Try to find branch by ID first, then by name
            target_branch = None
            if branch_identifier in self.branch_manager.branches:
                target_branch_id = branch_identifier
                target_branch = self.branch_manager.branches[branch_identifier]
            else:
                # Try to find by name
                target_branch = self.branch_manager.get_branch_by_name(
                    branch_identifier
                )
                target_branch_id = target_branch.id if target_branch else None

            if not target_branch:
                # Show available branches in error message
                available = [
                    f"'{b.name}' ({b.id})"
                    for b in self.branch_manager.branches.values()
                ]
                return CommandResult(
                    success=False,
                    message=f"Branch '{branch_identifier}' not found. Available: {', '.join(available)}",
                    should_continue=False,
                )

            # Switch to the branch
            success, message, branch = self.branch_manager.switch_branch(
                target_branch_id
            )

            if success:
                # Update our conversation history reference
                self.conversation_history.clear()
                self.conversation_history.extend(branch.conversation_history)

                # Display switch notification
                self._display_branch_switched(branch)

                return CommandResult(
                    success=True,
                    message=f"Switched to branch '{branch.name}'",
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False, message=message, should_continue=False
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error switching branch: {str(e)}",
                should_continue=False,
            )

    def _handle_branches(self, command: ParsedCommand) -> CommandResult:
        """Handle the /branches() command to list all branches."""
        try:
            branches_info = self.branch_manager.list_branches()
            self._display_branches_list(branches_info)

            current_branch = self.branch_manager.get_current_branch()
            return CommandResult(
                success=True,
                message=f"Showing {len(branches_info)} branches (current: {current_branch.name})",
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error listing branches: {str(e)}",
                should_continue=False,
            )

    def _handle_branch_delete(self, command: ParsedCommand) -> CommandResult:
        """Handle the /branch_delete(name) command to delete a branch."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="branch_delete command requires a branch name or ID",
                    should_continue=False,
                )

            branch_identifier = command.args[0]

            # Find the branch to delete
            target_branch = None
            target_branch_id = None

            if branch_identifier in self.branch_manager.branches:
                target_branch_id = branch_identifier
                target_branch = self.branch_manager.branches[branch_identifier]
            else:
                # Try to find by name
                target_branch = self.branch_manager.get_branch_by_name(
                    branch_identifier
                )
                target_branch_id = target_branch.id if target_branch else None

            if not target_branch:
                available = [
                    f"'{b.name}' ({b.id})"
                    for b in self.branch_manager.branches.values()
                    if b.id != "main"
                ]
                return CommandResult(
                    success=False,
                    message=f"Branch '{branch_identifier}' not found. Available: {', '.join(available)}",
                    should_continue=False,
                )

            # Delete the branch
            success, message = self.branch_manager.delete_branch(target_branch_id)

            if success:
                # If we switched to main, update conversation history
                if self.branch_manager.current_branch_id == "main":
                    current_branch = self.branch_manager.get_current_branch()
                    self.conversation_history.clear()
                    self.conversation_history.extend(
                        current_branch.conversation_history
                    )

                self._display_branch_deleted(target_branch.name)

                return CommandResult(
                    success=True, message=message, should_continue=False
                )
            else:
                return CommandResult(
                    success=False, message=message, should_continue=False
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error deleting branch: {str(e)}",
                should_continue=False,
            )

    def _display_branch_created(self, branch):
        """Display notification that a branch was created."""
        use_emoji = getattr(self._style, "use_emoji", True)
        title_style = getattr(self._style, "assistant_title_style", "bold cyan")
        border_style = getattr(self._style, "assistant_border_style", "green")
        expand = getattr(self._style, "expand_panels", False)

        emoji = "🌿 " if use_emoji else ""

        content = f"**Branch '{branch.name}'** created from current conversation point\nBranch ID: {branch.id}"

        self.console.print(
            Panel(
                Markdown(content),
                title=f"[{title_style}]{emoji}New Branch Created[/{title_style}]",
                border_style=border_style,
                padding=(0, 1),
                expand=expand,
            )
        )

    def _display_branch_switched(self, branch):
        """Display notification that branch was switched."""
        use_emoji = getattr(self._style, "use_emoji", True)
        title_style = getattr(self._style, "assistant_title_style", "bold cyan")
        border_style = getattr(self._style, "assistant_border_style", "blue")
        expand = getattr(self._style, "expand_panels", False)

        emoji = "🔀 " if use_emoji else ""

        preview = branch.get_preview(60)
        content = f"**Switched to '{branch.name}'**\nLast activity: {branch.last_active.strftime('%H:%M:%S')}\nPreview: {preview}"

        self.console.print(
            Panel(
                Markdown(content),
                title=f"[{title_style}]{emoji}Branch Switched[/{title_style}]",
                border_style=border_style,
                padding=(0, 1),
                expand=expand,
            )
        )

    def _display_branch_deleted(self, branch_name: str):
        """Display notification that branch was deleted."""
        use_emoji = getattr(self._style, "use_emoji", True)
        title_style = getattr(self._style, "assistant_title_style", "bold cyan")
        border_style = getattr(self._style, "assistant_border_style", "red")
        expand = getattr(self._style, "expand_panels", False)

        emoji = "🗑️ " if use_emoji else ""

        self.console.print(
            Panel(
                Text(f"Branch '{branch_name}' has been deleted", style="white"),
                title=f"[{title_style}]{emoji}Branch Deleted[/{title_style}]",
                border_style=border_style,
                padding=(0, 1),
                expand=expand,
            )
        )

    def _display_branches_list(self, branches_info):
        """Display a list of all branches."""
        use_emoji = getattr(self._style, "use_emoji", True)
        title_style = getattr(self._style, "assistant_title_style", "bold cyan")
        border_style = getattr(self._style, "assistant_border_style", "cyan")
        expand = getattr(self._style, "expand_panels", False)

        emoji = "🌳 " if use_emoji else ""

        # Create markdown table
        content_lines = [
            "| Branch | Status | Messages | Preview |",
            "|--------|--------|----------|---------|",
        ]

        for info in branches_info:
            status = "**Current**" if info["is_current"] else "Active"
            name_display = f"**{info['name']}**" if info["is_current"] else info["name"]
            preview = (
                info["preview"][:50] + "..."
                if len(info["preview"]) > 50
                else info["preview"]
            )

            content_lines.append(
                f"| {name_display} | {status} | {info['message_count']} | {preview} |"
            )

        content = "\n".join(content_lines)

        self.console.print(
            Panel(
                Markdown(content),
                title=f"[{title_style}]{emoji}Conversation Branches[/{title_style}]",
                border_style=border_style,
                padding=(1, 2),
                expand=expand,
            )
        )
