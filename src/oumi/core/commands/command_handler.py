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

import copy
import os
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
from oumi.core.thinking import ThinkingProcessor


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
        system_monitor=None,
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
        self.system_monitor = system_monitor

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
        
        # Initialize main branch with current model state
        main_branch = self.branch_manager.branches["main"]
        main_branch.model_name = getattr(config.model, "model_name", None)
        main_branch.engine_type = str(config.engine) if hasattr(config, 'engine') else None
        main_branch.model_config = self._serialize_model_config(config.model)
        main_branch.generation_config = self._serialize_generation_config(getattr(config, 'generation', None))

        # Initialize thinking processor and display settings
        self.thinking_processor = ThinkingProcessor()
        self.show_full_thoughts = False  # Default to compressed thinking view
    
    def _serialize_model_config(self, model_config) -> dict:
        """Serialize model config to a dictionary."""
        if not model_config:
            return {}
        
        # Extract key attributes from model config
        return {
            "model_name": getattr(model_config, "model_name", None),
            "torch_dtype_str": getattr(model_config, "torch_dtype_str", None),
            "model_max_length": getattr(model_config, "model_max_length", None),
            "attn_implementation": getattr(model_config, "attn_implementation", None),
            "trust_remote_code": getattr(model_config, "trust_remote_code", None),
        }
    
    def _serialize_generation_config(self, generation_config) -> dict:
        """Serialize generation config to a dictionary."""
        if not generation_config:
            return {}
        
        # Extract key attributes from generation config
        return {
            "max_new_tokens": getattr(generation_config, "max_new_tokens", None),
            "temperature": getattr(generation_config, "temperature", None),
            "top_p": getattr(generation_config, "top_p", None),
            "top_k": getattr(generation_config, "top_k", None),
            "sampling": getattr(generation_config, "use_sampling", None),
        }
    
    def _save_current_model_state_to_branch(self, branch_id: str):
        """Save current model state to the specified branch."""
        branch = self.branch_manager.branches.get(branch_id)
        if branch:
            branch.model_name = getattr(self.config.model, "model_name", None)
            branch.engine_type = str(self.config.engine) if hasattr(self.config, 'engine') else None
            branch.model_config = self._serialize_model_config(self.config.model)
            branch.generation_config = self._serialize_generation_config(getattr(self.config, 'generation', None))
    
    def _restore_model_state_from_branch(self, branch):
        """Restore model state from a branch and rebuild inference engine if needed."""
        try:
            # Check if we need to restore model state
            branch_model = branch.model_name
            branch_engine = branch.engine_type
            current_model = getattr(self.config.model, "model_name", None)
            current_engine = str(self.config.engine) if hasattr(self.config, 'engine') else None
            
            # Only restore if the model/engine is different
            if branch_model != current_model or branch_engine != current_engine:
                # Restore model configuration
                if branch.model_config:
                    for key, value in branch.model_config.items():
                        if hasattr(self.config.model, key) and value is not None:
                            setattr(self.config.model, key, value)
                
                # Restore generation configuration  
                if branch.generation_config and hasattr(self.config, 'generation'):
                    for key, value in branch.generation_config.items():
                        if hasattr(self.config.generation, key) and value is not None:
                            setattr(self.config.generation, key, value)
                
                # Restore engine type
                if branch_engine:
                    from oumi.core.configs import InferenceEngineType
                    try:
                        self.config.engine = InferenceEngineType(branch_engine)
                    except ValueError:
                        # If enum conversion fails, keep current engine
                        pass
                
                # Rebuild inference engine with restored config
                from oumi.builders.inference_engines import build_inference_engine
                self.inference_engine = build_inference_engine(
                    engine_type=self.config.engine,
                    model_params=self.config.model,
                    remote_params=getattr(self.config, 'remote_params', None),
                )
                
                # Update context usage in system monitor since restored model may have different max_length
                self._update_context_in_monitor()
                
        except Exception as e:
            # If restoration fails, log but don't crash - just keep current model
            print(f"Warning: Could not restore model state from branch: {e}")

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
            elif command.command == "full_thoughts":
                return self._handle_full_thoughts(command)
            elif command.command == "clear_thoughts":
                return self._handle_clear_thoughts(command)
            elif command.command == "clear":
                return self._handle_clear(command)
            elif command.command == "import":
                return self._handle_import(command)
            elif command.command == "swap":
                return self._handle_swap(command)
            elif command.command == "list_engines":
                return self._handle_list_engines(command)
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

            # Update context usage in system monitor
            self._update_context_in_monitor()

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
        """Handle the /save(path) command to export conversation to various formats."""
        if not command.args:
            return CommandResult(
                success=False,
                message="save command requires a file path argument",
                should_continue=False,
            )

        file_path = command.args[0].strip()
        
        # Detect format from extension or use optional format parameter
        format_type = command.kwargs.get('format', None)
        
        if format_type is None:
            # Detect from extension
            ext = file_path.lower().split('.')[-1] if '.' in file_path else None
            format_map = {
                'pdf': 'pdf',
                'txt': 'text',
                'md': 'markdown',
                'json': 'json',
                'csv': 'csv',
                'html': 'html'
            }
            format_type = format_map.get(ext, 'pdf')  # Default to PDF
            
            # Add extension if missing
            if ext not in format_map:
                file_path += f".{format_type}"
        
        try:
            # Route to appropriate export method based on format
            export_methods = {
                'pdf': self._export_conversation_to_pdf,
                'text': self._export_conversation_to_text,
                'markdown': self._export_conversation_to_markdown,
                'json': self._export_conversation_to_json,
                'csv': self._export_conversation_to_csv,
                'html': self._export_conversation_to_html
            }
            
            export_method = export_methods.get(format_type)
            if not export_method:
                return CommandResult(
                    success=False,
                    message=f"Unsupported format: {format_type}. Supported: {', '.join(export_methods.keys())}",
                    should_continue=False,
                )
            
            success, message = export_method(file_path)
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
    
    def _export_conversation_to_markdown(self, file_path: str) -> tuple[bool, str]:
        """Export conversation to Markdown format."""
        try:
            if not self.conversation_history:
                return False, "No conversation history to export"

            with open(file_path, "w", encoding="utf-8") as f:
                # Header
                from datetime import datetime
                
                model_name = getattr(self.config.model, "model_name", "Unknown Model")
                f.write(f"# Oumi Chat Export\n\n")
                f.write(f"**Model:** {model_name}  \n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
                f.write("---\n\n")
                
                # Conversation turns
                for i, msg in enumerate(self.conversation_history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    
                    # Skip attachment markers
                    if role == "attachment":
                        continue
                    
                    if role == "user":
                        f.write("## 👤 User\n\n")
                    elif role == "assistant":
                        f.write("## 🤖 Assistant\n\n")
                    else:
                        f.write(f"## {role.title()}\n\n")
                    
                    # Escape any existing markdown formatting
                    # Keep code blocks intact
                    lines = content.split('\n')
                    in_code_block = False
                    for line in lines:
                        if line.strip().startswith('```'):
                            in_code_block = not in_code_block
                        f.write(line + '\n')
                    
                    f.write("\n---\n\n")
            
            # Count messages
            user_msgs = len([m for m in self.conversation_history if m.get("role") == "user"])
            assistant_msgs = len([m for m in self.conversation_history if m.get("role") == "assistant"])
            
            return True, f"✅ Exported conversation to {file_path} (Markdown format) ({user_msgs} user messages, {assistant_msgs} assistant responses)"
            
        except Exception as e:
            return False, f"Failed to export Markdown file: {str(e)}"
    
    def _export_conversation_to_json(self, file_path: str) -> tuple[bool, str]:
        """Export conversation to JSON format."""
        try:
            import json
            from datetime import datetime
            
            if not self.conversation_history:
                return False, "No conversation history to export"
            
            # Prepare export data
            export_data = {
                "metadata": {
                    "model": getattr(self.config.model, "model_name", "Unknown Model"),
                    "exported_at": datetime.now().isoformat(),
                    "message_count": len(self.conversation_history)
                },
                "messages": []
            }
            
            # Add messages
            for msg in self.conversation_history:
                if msg.get("role") != "attachment":  # Skip attachment markers
                    export_data["messages"].append({
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "timestamp": datetime.now().isoformat()  # Approximate
                    })
            
            # Write JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            # Count messages
            user_msgs = len([m for m in export_data["messages"] if m["role"] == "user"])
            assistant_msgs = len([m for m in export_data["messages"] if m["role"] == "assistant"])
            
            return True, f"✅ Exported conversation to {file_path} (JSON format) ({user_msgs} user messages, {assistant_msgs} assistant responses)"
            
        except Exception as e:
            return False, f"Failed to export JSON file: {str(e)}"
    
    def _export_conversation_to_csv(self, file_path: str) -> tuple[bool, str]:
        """Export conversation to CSV format."""
        try:
            import csv
            from datetime import datetime
            
            if not self.conversation_history:
                return False, "No conversation history to export"
            
            with open(file_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(["timestamp", "role", "content"])
                
                # Messages
                for msg in self.conversation_history:
                    if msg.get("role") != "attachment":  # Skip attachment markers
                        writer.writerow([
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Approximate
                            msg.get("role", "unknown"),
                            msg.get("content", "")
                        ])
            
            # Count messages
            user_msgs = len([m for m in self.conversation_history if m.get("role") == "user"])
            assistant_msgs = len([m for m in self.conversation_history if m.get("role") == "assistant"])
            
            return True, f"✅ Exported conversation to {file_path} (CSV format) ({user_msgs} user messages, {assistant_msgs} assistant responses)"
            
        except Exception as e:
            return False, f"Failed to export CSV file: {str(e)}"
    
    def _export_conversation_to_html(self, file_path: str) -> tuple[bool, str]:
        """Export conversation to HTML format."""
        try:
            from datetime import datetime
            import html
            
            if not self.conversation_history:
                return False, "No conversation history to export"
            
            model_name = getattr(self.config.model, "model_name", "Unknown Model")
            
            # HTML template
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oumi Chat Export - {model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .message {{
            background: #fff;
            padding: 15px 20px;
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .user {{
            border-left: 4px solid #007bff;
        }}
        .assistant {{
            border-left: 4px solid #28a745;
        }}
        .role {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .content {{
            white-space: pre-wrap;
            line-height: 1.5;
        }}
        code {{
            background: #f0f0f0;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre {{
            background: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Oumi Chat Export</h1>
        <p><strong>Model:</strong> {html.escape(model_name)}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
            
            # Add messages
            for msg in self.conversation_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                # Skip attachment markers
                if role == "attachment":
                    continue
                
                role_class = role if role in ["user", "assistant"] else "other"
                role_emoji = "👤" if role == "user" else "🤖" if role == "assistant" else "💬"
                
                html_content += f"""
    <div class="message {role_class}">
        <div class="role">{role_emoji} {html.escape(role.title())}</div>
        <div class="content">{html.escape(content)}</div>
    </div>
"""
            
            html_content += """
</body>
</html>"""
            
            # Write HTML
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # Count messages
            user_msgs = len([m for m in self.conversation_history if m.get("role") == "user"])
            assistant_msgs = len([m for m in self.conversation_history if m.get("role") == "assistant"])
            
            return True, f"✅ Exported conversation to {file_path} (HTML format) ({user_msgs} user messages, {assistant_msgs} assistant responses)"
            
        except Exception as e:
            return False, f"Failed to export HTML file: {str(e)}"

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
- **`/clear()`** - Clear entire conversation history and start fresh

### Parameter Adjustment
- **`/set(param=value)`** - Adjust generation parameters
  - Examples:
    - `/set(temperature=0.8)` - More creative responses
    - `/set(top_p=0.9)` - Nucleus sampling
    - `/set(max_tokens=2048)` - Longer responses
    - `/set(sampling=true)` - Enable sampling
  - Available parameters: temperature, top_p, top_k, max_tokens, sampling, seed, frequency_penalty, presence_penalty, min_p, num_beams
- **`/swap(model_name)`** - Switch to a different model while preserving conversation
  - Examples:
    - `/swap(llama-3.1-8b)` - Switch to Llama 3.1 8B model
    - `/swap(anthropic:claude-3-5-sonnet-20241022)` - Switch to Claude via API
  - Note: Requires infrastructure support for dynamic model loading
- **`/list_engines()`** - List available inference engines and their supported models
  - Shows local engines (NATIVE, VLLM, LLAMACPP) and API engines (ANTHROPIC, OPENAI, etc.)
  - Includes sample models and API key requirements for each engine

### Import/Export
- **`/save(path)`** - Save conversation to various formats
  - Formats: PDF, TXT, MD, JSON, CSV, HTML (auto-detected from extension)
  - Examples:
    - `/save(chat.pdf)` - PDF with formatting
    - `/save(chat.txt)` - Plain text
    - `/save(chat.md)` - Markdown format
    - `/save(chat.json)` - Structured JSON
    - `/save(chat.csv)` - CSV for data analysis
    - `/save(chat.html)` - HTML with styling
  - Force format: `/save(myfile, format=json)`
- **`/import(path)`** - Import conversation data from supported formats
  - Formats: JSON, CSV, Excel (.xlsx/.xls), Markdown (.md), Text (.txt)
  - Examples:
    - `/import(chat.json)` - Import from JSON format
    - `/import(data.csv)` - Import from CSV with role/content columns
    - `/import(conversation.md)` - Import from Markdown with ## User/Assistant headers
  - Automatically detects format from file extension

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

### Thinking Display
- **`/full_thoughts()`** - Toggle between compressed and full thinking view
  - Compressed (default): Shows brief summaries of thinking content
  - Full mode: Shows complete thinking chains and reasoning
  - Works with multiple thinking formats: GPT-OSS, <think>, <reasoning>, etc.
- **`/clear_thoughts()`** - Remove thinking content from conversation history
  - Preserves the final responses while removing all thinking/reasoning sections
  - Useful for cleaning up conversation history while keeping the actual answers
  - Works across all supported thinking formats

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
/import(data.json)         # Import conversation from JSON file
/full_thoughts()           # Toggle thinking display mode
/clear_thoughts()          # Remove thinking content from history
/compact()                 # Compress conversation history
/branch()                  # Create a new conversation branch
/list_engines()            # Show available inference engines
/swap(llama-3.1-8b)        # Switch to different model
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
    
    def _update_context_in_monitor(self):
        """Update the context usage in system monitor if available."""
        if self.system_monitor:
            # Update max context length in case model has changed
            max_context = getattr(self.config.model, "model_max_length", 4096)
            self.system_monitor.update_max_context_tokens(max_context)
            
            # Update current usage
            estimated_tokens = self._estimate_conversation_tokens()
            self.system_monitor.update_context_usage(estimated_tokens)

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

            # Update context usage in system monitor
            self._update_context_in_monitor()

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

            # Before creating branch, ensure current branch has proper copies of conversation AND model state
            current_branch = self.branch_manager.get_current_branch()
            if current_branch:
                # Save current conversation history (especially important for main branch with shared reference)
                if current_branch.id == "main":
                    current_branch.conversation_history = copy.deepcopy(self.conversation_history)
                
                # Always save current model state to current branch before branching
                self._save_current_model_state_to_branch(current_branch.id)

            # Create branch from current branch
            success, message, new_branch = self.branch_manager.create_branch(
                from_branch_id=self.branch_manager.current_branch_id, name=name
            )

            if success and new_branch:
                # Automatically switch to the newly created branch
                switch_success, switch_message, _ = self.branch_manager.switch_branch(new_branch.id)
                
                if switch_success:
                    # Update our conversation history reference to point to the new branch
                    self.conversation_history.clear()
                    self.conversation_history.extend(new_branch.conversation_history)
                    
                    # Update context usage in system monitor
                    self._update_context_in_monitor()
                
                # Display success message with branch info
                self._display_branch_created(new_branch)
                return CommandResult(
                    success=True,
                    message=f"Created and switched to branch '{new_branch.name}' (ID: {new_branch.id})",
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
                # Save current model state to outgoing branch before switching
                current_branch = self.branch_manager.get_current_branch() 
                if current_branch and current_branch.id != target_branch_id:
                    self._save_current_model_state_to_branch(current_branch.id)
                
                # Update our conversation history reference
                self.conversation_history.clear()
                self.conversation_history.extend(branch.conversation_history)
                
                # Restore model state from target branch
                self._restore_model_state_from_branch(branch)

                # Update context usage in system monitor
                self._update_context_in_monitor()

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

    def _handle_full_thoughts(self, command: ParsedCommand) -> CommandResult:
        """Handle the /full_thoughts() command to toggle thinking display mode."""
        try:
            # Toggle the thinking display mode
            self.show_full_thoughts = not self.show_full_thoughts
            
            # Get style attributes
            use_emoji = getattr(self._style, "use_emoji", True)
            title_style = getattr(self._style, "assistant_title_style", "bold cyan")
            border_style = getattr(self._style, "assistant_border_style", "cyan")
            expand = getattr(self._style, "expand_panels", False)
            
            # Create status message
            mode = "full" if self.show_full_thoughts else "compressed"
            emoji = "🧠 " if use_emoji else ""
            
            status_message = f"Thinking display mode: **{mode}**"
            if self.show_full_thoughts:
                status_message += "\nShowing complete thinking chains with full details"
            else:
                status_message += "\nShowing compressed thinking summaries (default)"
            
            # Display the toggle result
            self.console.print(
                Panel(
                    Markdown(status_message),
                    title=f"[{title_style}]{emoji}Thinking Display Mode[/{title_style}]",
                    border_style=border_style,
                    padding=(0, 1),
                    expand=expand,
                )
            )
            
            return CommandResult(
                success=True,
                message=f"Thinking display set to {mode} mode",
                should_continue=False,
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error toggling thinking display: {str(e)}",
                should_continue=False,
            )

    def _handle_clear_thoughts(self, command: ParsedCommand) -> CommandResult:
        """Handle the /clear_thoughts() command to remove thinking content from conversation history."""
        try:
            # Track how many messages we process and clean
            processed_count = 0
            cleaned_count = 0
            
            for msg in self.conversation_history:
                if msg.get("role") == "assistant":
                    processed_count += 1
                    original_content = msg.get("content", "")
                    
                    # Use thinking processor to extract and separate content
                    thinking_result = self.thinking_processor.extract_thinking(original_content)
                    
                    if thinking_result.has_thinking:
                        # Replace the message content with just the final content (no thinking)
                        msg["content"] = thinking_result.final_content
                        cleaned_count += 1
            
            # Update context usage in system monitor
            self._update_context_in_monitor()
            
            # Get style attributes
            use_emoji = getattr(self._style, "use_emoji", True)
            title_style = getattr(self._style, "assistant_title_style", "bold cyan")
            border_style = getattr(self._style, "assistant_border_style", "cyan")
            expand = getattr(self._style, "expand_panels", False)
            
            # Create status message
            emoji = "🧹 " if use_emoji else ""
            
            if cleaned_count > 0:
                status_message = f"Cleaned thinking content from **{cleaned_count}** assistant messages"
                if processed_count > cleaned_count:
                    status_message += f" (out of {processed_count} total messages)"
                status_message += "\nConversation responses preserved, only thinking content removed"
            else:
                status_message = f"No thinking content found in **{processed_count}** assistant messages"
                status_message += "\nConversation history unchanged"
            
            # Display the result
            self.console.print(
                Panel(
                    Markdown(status_message),
                    title=f"[{title_style}]{emoji}Thoughts Cleared[/{title_style}]",
                    border_style=border_style,
                    padding=(0, 1),
                    expand=expand,
                )
            )
            
            return CommandResult(
                success=True,
                message=f"Cleared thinking content from {cleaned_count} messages",
                should_continue=False,
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error clearing thinking content: {str(e)}",
                should_continue=False,
            )

    def _handle_clear(self, command: ParsedCommand) -> CommandResult:
        """Handle the /clear() command to clear entire conversation history."""
        try:
            # Count messages before clearing
            message_count = len(self.conversation_history)
            
            # Clear the conversation history
            self.conversation_history.clear()
            
            # Reset context usage in system monitor
            self._update_context_in_monitor()
            
            # Get style attributes
            use_emoji = getattr(self._style, "use_emoji", True)
            title_style = getattr(self._style, "assistant_title_style", "bold cyan")
            border_style = getattr(self._style, "assistant_border_style", "cyan")
            expand = getattr(self._style, "expand_panels", False)
            warning_style = getattr(self._style, "error_style", "yellow")
            
            # Create status message
            emoji = "🧹 " if use_emoji else ""
            
            status_message = f"Cleared **{message_count}** messages from conversation history"
            status_message += "\n[dim]Starting fresh conversation...[/dim]"
            
            # Display the result
            self.console.print(
                Panel(
                    Markdown(status_message),
                    title=f"[{title_style}]{emoji}Conversation Cleared[/{title_style}]",
                    border_style=border_style,
                    padding=(0, 1),
                    expand=expand,
                )
            )
            
            # Also update current branch if using branch manager
            if self.branch_manager:
                current_branch = self.branch_manager.get_current_branch()
                current_branch.conversation_history.clear()
            
            return CommandResult(
                success=True,
                message=f"Cleared {message_count} messages from conversation history",
                should_continue=False,
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error clearing conversation: {str(e)}",
                should_continue=False,
            )
    
    def _handle_import(self, command: ParsedCommand) -> CommandResult:
        """Handle the /import() command to import conversation data from supported formats."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="Import command requires a file path",
                    should_continue=False,
                )
            
            file_path = command.args[0].strip()
            
            # Expand user path
            file_path = os.path.expanduser(file_path)
            
            if not os.path.exists(file_path):
                return CommandResult(
                    success=False,
                    message=f"File not found: {file_path}",
                    should_continue=False,
                )
            
            # Detect file format from extension
            _, ext = os.path.splitext(file_path.lower())
            
            # Import based on file format
            success = False
            message = ""
            imported_messages = []
            
            if ext == '.json':
                success, message, imported_messages = self._import_from_json(file_path)
            elif ext == '.csv':
                success, message, imported_messages = self._import_from_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                success, message, imported_messages = self._import_from_excel(file_path)
            elif ext == '.md':
                success, message, imported_messages = self._import_from_markdown(file_path)
            elif ext == '.txt':
                success, message, imported_messages = self._import_from_text(file_path)
            else:
                return CommandResult(
                    success=False,
                    message=f"Unsupported file format: {ext}. Supported: JSON, CSV, Excel (.xlsx/.xls), Markdown (.md), Text (.txt)",
                    should_continue=False,
                )
            
            if success and imported_messages:
                # Add imported messages to conversation history
                self.conversation_history.extend(imported_messages)
                
                # Update context usage in system monitor
                self._update_context_in_monitor()
                
                # Get style attributes
                use_emoji = getattr(self._style, "use_emoji", True)
                title_style = getattr(self._style, "assistant_title_style", "bold cyan")
                border_style = getattr(self._style, "assistant_border_style", "cyan")
                expand = getattr(self._style, "expand_panels", False)
                
                # Create status message
                emoji = "📁 " if use_emoji else ""
                
                # Display the result
                self.console.print(
                    Panel(
                        Markdown(message),
                        title=f"[{title_style}]{emoji}Conversation Imported[/{title_style}]",
                        border_style=border_style,
                        expand=expand,
                    )
                )
            
            return CommandResult(
                success=success,
                message=message if not success else f"Imported {len(imported_messages)} messages from {os.path.basename(file_path)}",
                should_continue=False,
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error importing conversation: {str(e)}",
                should_continue=False,
            )
    
    def _handle_swap(self, command: ParsedCommand) -> CommandResult:
        """Handle the /swap(model_name) or /swap(config:path) command to switch models while preserving conversation."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="Swap command requires a model name or config path (e.g., model_name, engine:model_name, or config:path/to/config.yaml)",
                    should_continue=False,
                )
            
            model_input = command.args[0].strip()
            
            # Check if this is a config file reference
            if model_input.startswith("config:"):
                config_path = model_input[7:]  # Remove "config:" prefix
                return self._handle_config_swap(config_path)
            
            # Parse engine:model_name syntax
            if ":" in model_input:
                engine_name, model_name = model_input.split(":", 1)
                engine_name = engine_name.upper()
            else:
                # Default to current engine or infer from model name
                engine_name = None
                model_name = model_input
            
            # Get style attributes
            use_emoji = getattr(self._style, "use_emoji", True)
            title_style = getattr(self._style, "assistant_title_style", "bold cyan")
            border_style = getattr(self._style, "assistant_border_style", "yellow")
            expand = getattr(self._style, "expand_panels", False)
            
            # Check if this is the current model
            current_model = getattr(self.config.model, "model_name", "unknown")
            current_engine = getattr(self.config, "engine", "unknown")
            
            display_name = f"{engine_name}:{model_name}" if engine_name else model_name
            current_display = f"{current_engine}:{current_model}" if hasattr(self.config, 'engine') else current_model
            
            if model_name == current_model and (not engine_name or engine_name == str(current_engine)):
                return CommandResult(
                    success=False,
                    message=f"Already using model: {current_display}",
                    should_continue=False,
                )
            
            # Show swap status message
            emoji = "🔄 " if use_emoji else ""
            status_message = f"Requesting model swap to: **{display_name}**"
            status_message += f"\nCurrent: {current_display}"
            if engine_name:
                status_message += f"\nNew engine: {engine_name}"
            status_message += "\n[dim]Attempting to swap models...[/dim]"
            
            self.console.print(
                Panel(
                    Markdown(status_message),
                    title=f"[{title_style}]{emoji}Model Swap Requested[/{title_style}]",
                    border_style=border_style,
                    expand=expand,
                )
            )
            
            # Actually perform the model swap
            try:
                # Import the inference engine builder
                from oumi.builders.inference_engines import build_inference_engine
                
                # Update current config with new model/engine
                if engine_name:
                    # Map string engine names to enum values
                    from oumi.core.configs import InferenceEngineType
                    try:
                        new_engine_type = InferenceEngineType(engine_name)
                        self.config.engine = new_engine_type
                    except ValueError:
                        return CommandResult(
                            success=False,
                            message=f"Unknown engine type: {engine_name}. Use /list_engines() to see available engines.",
                            should_continue=False,
                        )
                
                # Update model name
                self.config.model.model_name = model_name
                
                # Create new inference engine
                new_inference_engine = build_inference_engine(
                    engine_type=self.config.engine,
                    model_params=self.config.model,
                    remote_params=getattr(self.config, 'remote_params', None),
                )
                
                # Update the command handler's inference engine reference
                self.inference_engine = new_inference_engine
                
                # Save the new model state to current branch
                current_branch = self.branch_manager.get_current_branch()
                if current_branch:
                    self._save_current_model_state_to_branch(current_branch.id)
                
                # Update context usage in system monitor since model may have different max_length
                self._update_context_in_monitor()
                
                # Show success message
                success_message = f"✅ **Model swap completed successfully!**\n\n"
                success_message += f"**Switched from:**\n"
                success_message += f"• Engine: `{current_engine}`\n"
                success_message += f"• Model: `{current_model}`\n\n"
                success_message += f"**Switched to:**\n"
                success_message += f"• Engine: `{self.config.engine}`\n"
                success_message += f"• Model: `{model_name}`\n\n"
                success_message += f"Conversation history preserved. You can now continue chatting with the new model."
                
                emoji = "✅ " if use_emoji else ""
                self.console.print(
                    Panel(
                        Markdown(success_message),
                        title=f"[{title_style}]{emoji}Model Swap Successful[/{title_style}]",
                        border_style="green",
                        expand=expand,
                    )
                )
                
                return CommandResult(
                    success=True,
                    message=f"Successfully swapped to {display_name}",
                    should_continue=False,
                )
                
            except Exception as e:
                error_message = f"Failed to swap models: {str(e)}"
                return CommandResult(
                    success=False,
                    message=error_message,
                    should_continue=False,
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error processing model swap request: {str(e)}",
                should_continue=False,
            )
    
    def _handle_config_swap(self, config_path: str) -> CommandResult:
        """Handle config-based model swapping by loading an Oumi YAML config."""
        try:
            import os
            from pathlib import Path
            
            # Convert to absolute path and validate
            if not config_path.startswith('/'):
                # Make relative to current working directory
                config_path = os.path.abspath(config_path)
            
            config_file = Path(config_path)
            if not config_file.exists():
                return CommandResult(
                    success=False,
                    message=f"Config file not found: {config_path}",
                    should_continue=False,
                )
            
            if not config_file.suffix.lower() in ['.yaml', '.yml']:
                return CommandResult(
                    success=False,
                    message=f"Config file must be YAML format (.yaml or .yml): {config_path}",
                    should_continue=False,
                )
            
            # Load the inference config
            try:
                new_config = InferenceConfig.from_yaml(str(config_file))
            except Exception as e:
                return CommandResult(
                    success=False,
                    message=f"Failed to load config from {config_path}: {str(e)}",
                    should_continue=False,
                )
            
            # Extract key information from the config
            new_model = getattr(new_config.model, "model_name", "Unknown")
            new_engine = getattr(new_config, "engine", "Unknown")
            
            # Get current info for comparison
            current_model = getattr(self.config.model, "model_name", "Unknown")
            current_engine = getattr(self.config, "engine", "Unknown")
            
            # Check if this would be the same configuration
            if new_model == current_model and str(new_engine) == str(current_engine):
                return CommandResult(
                    success=False,
                    message=f"Config would load the same model/engine: {current_engine}:{current_model}",
                    should_continue=False,
                )
            
            # Get style attributes
            use_emoji = getattr(self._style, "use_emoji", True)
            title_style = getattr(self._style, "assistant_title_style", "bold cyan")
            border_style = getattr(self._style, "assistant_border_style", "yellow")
            expand = getattr(self._style, "expand_panels", False)
            
            # Show detailed config swap information
            emoji = "📄 " if use_emoji else ""
            
            # Extract additional config details for display
            generation_params = []
            if hasattr(new_config, 'generation') and new_config.generation:
                gen_config = new_config.generation
                if hasattr(gen_config, 'max_new_tokens') and gen_config.max_new_tokens:
                    generation_params.append(f"max_new_tokens={gen_config.max_new_tokens}")
                if hasattr(gen_config, 'temperature') and gen_config.temperature is not None:
                    generation_params.append(f"temperature={gen_config.temperature}")
                if hasattr(gen_config, 'top_p') and gen_config.top_p is not None:
                    generation_params.append(f"top_p={gen_config.top_p}")
            
            model_params = []
            if hasattr(new_config.model, 'torch_dtype_str') and new_config.model.torch_dtype_str:
                model_params.append(f"dtype={new_config.model.torch_dtype_str}")
            if hasattr(new_config.model, 'model_max_length') and new_config.model.model_max_length:
                model_params.append(f"max_length={new_config.model.model_max_length}")
                
            # Create detailed status message
            status_message = f"**Config Swap Requested**\n"
            status_message += f"Config: `{config_file.name}`\n"
            status_message += f"Path: `{config_path}`\n\n"
            status_message += f"**Target Configuration:**\n"
            status_message += f"• Engine: `{new_engine}`\n"
            status_message += f"• Model: `{new_model}`\n"
            
            if generation_params:
                status_message += f"• Generation: {', '.join(generation_params)}\n"
            if model_params:
                status_message += f"• Model Settings: {', '.join(model_params)}\n"
            
            status_message += f"\n**Current Configuration:**\n"
            status_message += f"• Engine: `{current_engine}`\n"  
            status_message += f"• Model: `{current_model}`\n\n"
            status_message += "[dim]Attempting to swap models...[/dim]"
            
            self.console.print(
                Panel(
                    Markdown(status_message),
                    title=f"[{title_style}]{emoji}Config-Based Model Swap[/{title_style}]",
                    border_style=border_style,
                    expand=expand,
                )
            )
            
            # Actually perform the config-based swap
            try:
                # Import the inference engine builder
                from oumi.builders.inference_engines import build_inference_engine
                
                # Update current config with new settings
                self.config.engine = new_config.engine
                self.config.model = new_config.model
                if hasattr(new_config, 'generation') and new_config.generation:
                    self.config.generation = new_config.generation
                if hasattr(new_config, 'remote_params') and new_config.remote_params:
                    self.config.remote_params = new_config.remote_params
                
                # Create new inference engine
                new_inference_engine = build_inference_engine(
                    engine_type=new_config.engine,
                    model_params=new_config.model,
                    remote_params=getattr(new_config, 'remote_params', None),
                )
                
                # Update the command handler's inference engine reference
                self.inference_engine = new_inference_engine
                
                # Save the new model state to current branch
                current_branch = self.branch_manager.get_current_branch()
                if current_branch:
                    self._save_current_model_state_to_branch(current_branch.id)
                
                # Update context usage in system monitor since model may have different max_length
                self._update_context_in_monitor()
                
                # Show success message
                success_message = f"✅ **Model swap completed successfully!**\n\n"
                success_message += f"**Switched from:**\n"
                success_message += f"• Engine: `{current_engine}`\n"
                success_message += f"• Model: `{current_model}`\n\n"
                success_message += f"**Switched to:**\n"
                success_message += f"• Engine: `{new_engine}`\n"
                success_message += f"• Model: `{new_model}`\n"
                if generation_params:
                    success_message += f"• Generation: {', '.join(generation_params)}\n"
                if model_params:
                    success_message += f"• Model Settings: {', '.join(model_params)}\n"
                success_message += f"\nConversation history preserved. You can now continue chatting with the new model."
                
                emoji = "✅ " if use_emoji else ""
                self.console.print(
                    Panel(
                        Markdown(success_message),
                        title=f"[{title_style}]{emoji}Model Swap Successful[/{title_style}]",
                        border_style="green",
                        expand=expand,
                    )
                )
                
                return CommandResult(
                    success=True,
                    message=f"Successfully swapped to {new_engine}:{new_model}",
                    should_continue=False,
                )
                
            except Exception as e:
                error_message = f"Failed to swap models: {str(e)}"
                return CommandResult(
                    success=False,
                    message=error_message,
                    should_continue=False,
                )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error processing config swap request: {str(e)}",
                should_continue=False,
            )
    
    def _handle_list_engines(self, command: ParsedCommand) -> CommandResult:
        """Handle the /list_engines() command to list available inference engines and sample models."""
        try:
            # Get style attributes
            use_emoji = getattr(self._style, "use_emoji", True)
            title_style = getattr(self._style, "assistant_title_style", "bold cyan")
            border_style = getattr(self._style, "assistant_border_style", "cyan")
            expand = getattr(self._style, "expand_panels", False)
            
            # Create engines info
            engines_info = self._get_engines_info()
            
            # Create markdown content
            emoji = "🔧 " if use_emoji else ""
            content_lines = [
                "Available inference engines in Oumi:",
                "",
                "| Engine | Type | Sample Models | API Key Required |",
                "|--------|------|---------------|------------------|"
            ]
            
            for engine_info in engines_info:
                # Format sample models (limit to avoid overly wide table)
                sample_models = ", ".join(engine_info["sample_models"][:3])
                if len(engine_info["sample_models"]) > 3:
                    sample_models += f", +{len(engine_info['sample_models'])-3} more"
                
                api_key = "✅" if use_emoji and engine_info["requires_api_key"] else ("Yes" if engine_info["requires_api_key"] else "No")
                if not use_emoji:
                    api_key = "Yes" if engine_info["requires_api_key"] else "No"
                
                content_lines.append(
                    f"| **{engine_info['name']}** | {engine_info['type']} | {sample_models} | {api_key} |"
                )
            
            # Add usage instructions
            content_lines.extend([
                "",
                "### Usage for /swap() Command:",
                "",
                "**Local Engines** (NATIVE, VLLM, LLAMACPP):",
                "- `/swap(model_name)` - Uses HuggingFace model name",
                "- Example: `/swap(meta-llama/Llama-3.1-8B-Instruct)`",
                "",
                "**API Engines** (ANTHROPIC, OPENAI, TOGETHER, etc.):",
                "- `/swap(engine:model_name)` - Specifies both engine and model",
                "- Examples:",
                "  - `/swap(anthropic:claude-3-5-sonnet-20241022)`",
                "  - `/swap(openai:gpt-4o)`",
                "  - `/swap(together:meta-llama/Llama-3.1-70B-Instruct-Turbo)`",
                "",
                "**Setting up API Keys:**",
                "- Set environment variables in `~/.zshrc` or `~/.bashrc`",
                "- Example: `export ANTHROPIC_API_KEY=your_key_here`",
                "",
                "**Note:** Model swapping requires infrastructure support and is currently in development."
            ])
            
            content = "\n".join(content_lines)
            
            # Display the engines list
            self.console.print(
                Panel(
                    Markdown(content),
                    title=f"[{title_style}]{emoji}Oumi Inference Engines[/{title_style}]",
                    border_style=border_style,
                    padding=(1, 2),
                    expand=expand,
                )
            )
            
            return CommandResult(
                success=True,
                message=f"Listed {len(engines_info)} available inference engines",
                should_continue=False,
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error listing engines: {str(e)}",
                should_continue=False,
            )
    
    def _get_engines_info(self) -> list[dict]:
        """Get information about available inference engines and their sample models."""
        return [
            {
                "name": "NATIVE",
                "type": "Local",
                "description": "Native PyTorch inference with transformers library",
                "requires_api_key": False,
                "sample_models": [
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "Qwen/Qwen2.5-3B-Instruct", 
                    "microsoft/Phi-3.5-mini-instruct",
                    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
                ]
            },
            {
                "name": "VLLM",
                "type": "Local",
                "description": "High-performance local inference with vLLM",
                "requires_api_key": False,
                "sample_models": [
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "Qwen/Qwen2.5-32B-Instruct",
                    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                    "Qwen/QwQ-32B-Preview"
                ]
            },
            {
                "name": "LLAMACPP",
                "type": "Local",
                "description": "CPU/GPU optimized inference with llama.cpp (GGUF files)",
                "requires_api_key": False,
                "sample_models": [
                    "microsoft/Phi-3.5-mini-instruct (GGUF)",
                    "Qwen/Qwen2.5-3B-Instruct (GGUF)",
                    "meta-llama/Llama-4-Scout-17B-16E-Instruct (GGUF)",
                    "deepseek-ai/DeepSeek-R1-0528-GGUF",
                    "Qwen/Qwen3-30B-A3B-Instruct-GGUF"
                ]
            },
            {
                "name": "SGLANG",
                "type": "Local/Remote",
                "description": "SGLang inference engine for complex generation",
                "requires_api_key": False,
                "sample_models": [
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "Qwen/Qwen2-VL-2B-Instruct",
                    "meta-llama/Llama-3.2-11B-Vision-Instruct"
                ]
            },
            {
                "name": "REMOTE_VLLM",
                "type": "Remote",
                "description": "Connect to external vLLM server instance",
                "requires_api_key": False,
                "sample_models": [
                    "Custom models hosted on remote vLLM server",
                    "Configure via base_url in config"
                ]
            },
            {
                "name": "ANTHROPIC",
                "type": "API",
                "description": "Anthropic's Claude models via API",
                "requires_api_key": True,
                "sample_models": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022", 
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
            },
            {
                "name": "OPENAI",
                "type": "API",
                "description": "OpenAI's GPT models via API",
                "requires_api_key": True,
                "sample_models": [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                    "o1-preview",
                    "o1-mini"
                ]
            },
            {
                "name": "TOGETHER",
                "type": "API", 
                "description": "Together AI's hosted models",
                "requires_api_key": True,
                "sample_models": [
                    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    "meta-llama/Llama-4-Maverick-17B-Instruct",
                    "deepseek-ai/DeepSeek-R1",
                    "meta-llama/Llama-3.1-70B-Instruct-Turbo",
                    "Qwen/Qwen3-235B-A22B-Thinking"
                ]
            },
            {
                "name": "DEEPSEEK",
                "type": "API",
                "description": "DeepSeek's platform API",
                "requires_api_key": True,
                "sample_models": [
                    "deepseek-chat",
                    "deepseek-reasoner",
                    "deepseek-coder",
                    "deepseek-math"
                ]
            },
            {
                "name": "GOOGLE_VERTEX",
                "type": "API",
                "description": "Google Cloud Vertex AI models",
                "requires_api_key": True,
                "sample_models": [
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.0-pro",
                    "text-bison",
                    "chat-bison"
                ]
            },
            {
                "name": "GEMINI", 
                "type": "API",
                "description": "Google's Gemini models via API",
                "requires_api_key": True,
                "sample_models": [
                    "gemini-1.5-pro-latest",
                    "gemini-1.5-flash-latest", 
                    "gemini-1.0-pro-latest"
                ]
            },
            {
                "name": "LAMBDA",
                "type": "API",
                "description": "Lambda Labs cloud inference",
                "requires_api_key": True,
                "sample_models": [
                    "Custom models hosted on Lambda",
                    "Check Lambda Labs documentation"
                ]
            },
            {
                "name": "PARASAIL",
                "type": "API",
                "description": "Parasail inference platform",
                "requires_api_key": True,
                "sample_models": [
                    "Custom models on Parasail platform",
                    "Check Parasail documentation"
                ]
            },
            {
                "name": "SAMBANOVA",
                "type": "API",
                "description": "SambaNova's hosted models",
                "requires_api_key": True,
                "sample_models": [
                    "Meta-Llama-3.1-8B-Instruct",
                    "Meta-Llama-3.1-70B-Instruct", 
                    "Check SambaNova catalog"
                ]
            },
            {
                "name": "REMOTE",
                "type": "API",
                "description": "Generic OpenAI-compatible API endpoints",
                "requires_api_key": True,
                "sample_models": [
                    "Custom endpoints with OpenAI format",
                    "Configure via base_url in config"
                ]
            }
        ]

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
# Import format helpers to append to command_handler.py
    
    # Import format helpers
    def _import_from_json(self, file_path: str) -> tuple[bool, str, list]:
        """Import conversation from JSON format."""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = []
            
            # Handle both direct message arrays and wrapped formats
            if isinstance(data, list):
                # Direct message array
                raw_messages = data
            elif isinstance(data, dict) and "messages" in data:
                # Wrapped format with metadata
                raw_messages = data["messages"]
            else:
                return False, "Invalid JSON format: expected message array or object with 'messages' field", []
            
            # Convert to our message format
            for msg in raw_messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    # Filter to supported roles
                    role = msg["role"].lower()
                    if role in ["user", "assistant", "system"]:
                        messages.append({
                            "role": role,
                            "content": msg["content"]
                        })
            
            if not messages:
                return False, "No valid messages found in JSON file", []
            
            user_count = len([m for m in messages if m["role"] == "user"])
            assistant_count = len([m for m in messages if m["role"] == "assistant"])
            
            return True, f"Imported **{len(messages)}** messages from JSON ({user_count} user, {assistant_count} assistant)", messages
            
        except Exception as e:
            return False, f"Failed to parse JSON: {str(e)}", []
    
    def _import_from_csv(self, file_path: str) -> tuple[bool, str, list]:
        """Import conversation from CSV format."""
        try:
            import csv
            
            messages = []
            
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                # Try to detect CSV format
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(f, delimiter=delimiter)
                
                # Look for required columns
                fieldnames = reader.fieldnames or []
                
                # Find role and content columns (case-insensitive)
                role_col = None
                content_col = None
                
                for field in fieldnames:
                    field_lower = field.lower()
                    if field_lower in ['role', 'type', 'speaker']:
                        role_col = field
                    elif field_lower in ['content', 'message', 'text']:
                        content_col = field
                
                if not role_col or not content_col:
                    return False, f"CSV must have role column ({['role', 'type', 'speaker']}) and content column ({['content', 'message', 'text']})", []
                
                for row in reader:
                    role = row.get(role_col, "").lower().strip()
                    content = row.get(content_col, "").strip()
                    
                    if role in ["user", "assistant", "system"] and content:
                        messages.append({
                            "role": role,
                            "content": content
                        })
            
            if not messages:
                return False, "No valid messages found in CSV file", []
            
            user_count = len([m for m in messages if m["role"] == "user"])
            assistant_count = len([m for m in messages if m["role"] == "assistant"])
            
            return True, f"Imported **{len(messages)}** messages from CSV ({user_count} user, {assistant_count} assistant)", messages
            
        except Exception as e:
            return False, f"Failed to parse CSV: {str(e)}", []
    
    def _import_from_excel(self, file_path: str) -> tuple[bool, str, list]:
        """Import conversation from Excel format."""
        try:
            # Try pandas first
            try:
                import pandas as pd
                df = pd.read_excel(file_path)
            except ImportError:
                return False, "pandas is required to import Excel files. Install with: pip install pandas openpyxl", []
            
            messages = []
            
            # Find role and content columns (case-insensitive)
            columns = {col.lower(): col for col in df.columns}
            
            role_col = None
            content_col = None
            
            for role_key in ['role', 'type', 'speaker']:
                if role_key in columns:
                    role_col = columns[role_key]
                    break
            
            for content_key in ['content', 'message', 'text']:
                if content_key in columns:
                    content_col = columns[content_key]
                    break
            
            if not role_col or not content_col:
                return False, f"Excel file must have role column ({['role', 'type', 'speaker']}) and content column ({['content', 'message', 'text']})", []
            
            for _, row in df.iterrows():
                role = str(row[role_col]).lower().strip()
                content = str(row[content_col]).strip()
                
                # Skip NaN and empty values
                if role in ["user", "assistant", "system"] and content and content != "nan":
                    messages.append({
                        "role": role,
                        "content": content
                    })
            
            if not messages:
                return False, "No valid messages found in Excel file", []
            
            user_count = len([m for m in messages if m["role"] == "user"])
            assistant_count = len([m for m in messages if m["role"] == "assistant"])
            
            return True, f"Imported **{len(messages)}** messages from Excel ({user_count} user, {assistant_count} assistant)", messages
            
        except Exception as e:
            return False, f"Failed to parse Excel: {str(e)}", []
    
    def _import_from_markdown(self, file_path: str) -> tuple[bool, str, list]:
        """Import conversation from Markdown format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            messages = []
            current_role = None
            current_content = []
            
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Look for role markers
                if line.startswith('## User') or line.startswith('**User') or line.startswith('# User'):
                    # Save previous message
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "user"
                    current_content = []
                    
                elif line.startswith('## Assistant') or line.startswith('**Assistant') or line.startswith('# Assistant'):
                    # Save previous message
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "assistant"
                    current_content = []
                    
                elif line.startswith('## System') or line.startswith('**System') or line.startswith('# System'):
                    # Save previous message
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "system"
                    current_content = []
                    
                elif line.startswith('---') and len(line) >= 3 and all(c == '-' for c in line):
                    # Message separator - save current and reset
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                        current_role = None
                        current_content = []
                    
                else:
                    # Content line
                    if current_role:
                        current_content.append(line)
            
            # Save final message
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            
            if not messages:
                return False, "No messages found. Expected format with '## User', '## Assistant' headers or '---' separators", []
            
            user_count = len([m for m in messages if m["role"] == "user"])
            assistant_count = len([m for m in messages if m["role"] == "assistant"])
            
            return True, f"Imported **{len(messages)}** messages from Markdown ({user_count} user, {assistant_count} assistant)", messages
            
        except Exception as e:
            return False, f"Failed to parse Markdown: {str(e)}", []
    
    def _import_from_text(self, file_path: str) -> tuple[bool, str, list]:
        """Import conversation from plain text format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            messages = []
            current_role = None
            current_content = []
            
            lines = content.split('\n')
            
            for line in lines:
                line_stripped = line.strip()
                
                # Look for role indicators at start of lines
                line_lower = line_stripped.lower()
                
                if (line_lower.startswith('user:') or line_lower.startswith('human:') or 
                    line_lower.startswith('me:') or line_lower.startswith('q:')):
                    # Save previous message
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "user"
                    # Extract content after colon
                    content_part = line_stripped[line_stripped.find(':') + 1:].strip()
                    current_content = [content_part] if content_part else []
                    
                elif (line_lower.startswith('assistant:') or line_lower.startswith('ai:') or 
                      line_lower.startswith('bot:') or line_lower.startswith('a:')):
                    # Save previous message
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "assistant"
                    # Extract content after colon
                    content_part = line_stripped[line_stripped.find(':') + 1:].strip()
                    current_content = [content_part] if content_part else []
                    
                elif line_lower.startswith('system:'):
                    # Save previous message
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "system"
                    # Extract content after colon
                    content_part = line_stripped[line_stripped.find(':') + 1:].strip()
                    current_content = [content_part] if content_part else []
                    
                elif line_stripped == '' and current_role and current_content:
                    # Empty line might indicate message boundary
                    # But continue with current role unless we find a new one
                    current_content.append('')
                    
                else:
                    # Content line
                    if current_role:
                        current_content.append(line)
            
            # Save final message
            if current_role and current_content:
                content_text = '\n'.join(current_content).strip()
                if content_text:
                    messages.append({
                        "role": current_role,
                        "content": content_text
                    })
            
            if not messages:
                return False, "No messages found. Expected format with 'User:', 'Assistant:' prefixes or similar", []
            
            user_count = len([m for m in messages if m["role"] == "user"])
            assistant_count = len([m for m in messages if m["role"] == "assistant"])
            
            return True, f"Imported **{len(messages)}** messages from text ({user_count} user, {assistant_count} assistant)", messages
            
        except Exception as e:
            return False, f"Failed to parse text: {str(e)}", []