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

"""File operations command handler."""

import json
from datetime import datetime
from pathlib import Path

from rich.panel import Panel

from oumi.core.commands.base_handler import BaseCommandHandler, CommandResult
from oumi.core.commands.command_parser import ParsedCommand
from oumi.core.commands.utilities.export_utilities import ExportUtilities
from oumi.core.commands.utilities.import_utilities import ImportUtilities


class FileOperationsHandler(BaseCommandHandler):
    """Handles file-related commands: attach, save, import, load."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.export_utilities = ExportUtilities(self.context)
        self.import_utilities = ImportUtilities(self.context)

        # Auto-save functionality
        self._auto_save = True
        self._setup_chat_cache()

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return ["attach", "save", "import", "load"]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a file operations command."""
        if command.command == "attach":
            return self._handle_attach(command)
        elif command.command == "save":
            return self._handle_save(command)
        elif command.command == "import":
            return self._handle_import(command)
        elif command.command == "load":
            return self._handle_load(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unsupported command: {command.command}",
                should_continue=False,
            )

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
            attachment_result = self.context.file_handler.attach_file(
                file_path, conversation_tokens
            )

            if attachment_result.success:
                # Display attachment info
                self._display_attachment_result(attachment_result)

                # Add content to conversation (this will be used in the next inference)
                self._add_attachment_to_conversation(attachment_result)
                
                # Update context monitor to reflect the added attachment content
                self._update_context_in_monitor()

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
                message=f"Error attaching file: {str(e)}\\n\\nTraceback:\\n{error_details}",
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

        # Check for explicit format specification
        format_override = None
        if len(command.args) > 1:
            # Look for format=value in args
            for arg in command.args[1:]:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    if key.strip().lower() == "format":
                        format_override = value.strip().lower()

        # Determine format from file extension or override
        if format_override:
            export_format = format_override
        else:
            path_obj = Path(file_path)
            extension = path_obj.suffix.lower()
            format_map = {
                ".pdf": "pdf",
                ".txt": "text",
                ".md": "markdown",
                ".json": "json",
                ".csv": "csv",
                ".html": "html",
                ".htm": "html",
            }
            export_format = format_map.get(extension, "text")

        # Call appropriate export method
        success, message = self.export_utilities.export_conversation(
            file_path, export_format, self.conversation_history
        )

        return CommandResult(
            success=success,
            message=message,
            should_continue=False,
        )

    def _handle_import(self, command: ParsedCommand) -> CommandResult:
        """Handle the /import() command to import conversation data from supported formats."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="import command requires a file path argument",
                    should_continue=False,
                )

            file_path = command.args[0].strip()

            # Import the conversation
            success, message, imported_messages = (
                self.import_utilities.import_conversation(file_path)
            )

            if success and imported_messages:
                # Add imported messages to conversation history
                self.conversation_history.extend(imported_messages)
                self._update_context_in_monitor()

                return CommandResult(
                    success=True,
                    message=f"Imported {len(imported_messages)} messages from {file_path}",
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message=message or f"Failed to import from {file_path}",
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error importing conversation: {str(e)}",
                should_continue=False,
            )

    def _handle_load(self, command: ParsedCommand) -> CommandResult:
        """Handle the /load() command to load a chat from cache or browse recent chats."""
        try:
            # If no arguments, show recent chats for browsing
            if not command.args:
                from oumi.core.commands.chat_browser import ChatBrowser

                browser = ChatBrowser(self.console)
                selected_chat_id = browser.browse_recent_chats()

                if selected_chat_id:
                    return self._load_chat_by_id(selected_chat_id)
                else:
                    return CommandResult(
                        success=False,
                        message="No chat selected",
                        should_continue=False,
                    )

            # Load specific chat by ID
            chat_id = command.args[0].strip()
            return self._load_chat_by_id(chat_id)

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error loading chat: {str(e)}",
                should_continue=False,
            )

    def _display_attachment_result(self, result):
        """Display the result of a file attachment."""
        file_info = result.file_info

        # Create title with file name and basic info
        title = f"📎 Attached: {file_info.name}"

        # Build content info
        content_parts = []
        size_mb = (file_info.size_bytes or 0) / (1024 * 1024)
        content_parts.append(f"**Size:** {size_mb:.2f} MB")
        content_parts.append(f"**Type:** {file_info.file_type}")

        if hasattr(result, 'processing_info') and result.processing_info:
            content_parts.append(f"**Processing:** {result.processing_info}")

        if result.context_info:
            content_parts.append(f"**Context:** {result.context_info}")

        content = "\\n".join(content_parts)

        # Display in a panel
        panel = Panel(
            content,
            title=title,
            border_style=getattr(self._style, "attachment_border_style", "cyan"),
        )
        self.console.print(panel)

    def _add_attachment_to_conversation(self, result):
        """Add attachment content to conversation history."""
        # Add attachment message that will be processed by inference engine
        attachment_message = {
            "role": "attachment",
            "text_content": result.text_content,
            "file_info": {
                "name": result.file_info.name,
                "path": result.file_info.path,
                "size_bytes": result.file_info.size_bytes,
                "file_type": str(result.file_info.file_type)
            }
        }
        self.conversation_history.append(attachment_message)
        
        print(f"🔧 DEBUG: Added attachment to conversation history: {result.file_info.name}")
        print(f"🔧 DEBUG: Text content length: {len(result.text_content)} characters")

    def _load_chat_by_id(self, chat_id: str) -> CommandResult:
        """Load a specific chat by ID."""
        try:
            cache_file = self.chat_cache_dir / f"{chat_id}.json"

            if not cache_file.exists():
                return CommandResult(
                    success=False,
                    message=f"Chat '{chat_id}' not found in cache",
                    should_continue=False,
                )

            with open(cache_file, encoding="utf-8") as f:
                chat_data = json.load(f)

            # Load conversation history
            if "conversation_history" in chat_data:
                self.conversation_history.clear()
                self.conversation_history.extend(chat_data["conversation_history"])

                # Update context monitor
                self._update_context_in_monitor()

                # Display success message with chat info
                timestamp = chat_data.get("last_updated", "Unknown time")
                message_count = len(chat_data["conversation_history"])

                return CommandResult(
                    success=True,
                    message=f"Loaded chat '{chat_id}' ({message_count} messages, {timestamp})",
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message=f"Invalid chat data in '{chat_id}'",
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error loading chat '{chat_id}': {str(e)}",
                should_continue=False,
            )

    def _setup_chat_cache(self):
        """Set up the chat cache directory."""
        # Create cache directory
        cache_dir = Path.home() / ".oumi" / "chat_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.chat_cache_dir = cache_dir

        # Generate chat ID for this session
        model_name = getattr(self.config.model, "model_name", "default")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.chat_id = f"{model_name}_{timestamp}"
        self.chat_file = self.chat_cache_dir / f"{self.chat_id}.json"

    def auto_save_chat(self):
        """Auto-save the current chat if enabled."""
        if not self._auto_save:
            return

        try:
            # Prepare chat data
            chat_data = {
                "chat_id": self.chat_id,
                "model_name": getattr(self.config.model, "model_name", "default"),
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "conversation_history": self.conversation_history,
                "model_config": self._serialize_model_config(self.config.model),
                "generation_config": self._serialize_generation_config(
                    self.config.generation
                ),
            }

            # Save to cache
            with open(self.chat_file, "w", encoding="utf-8") as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)

        except Exception:
            # Silently fail for auto-save to avoid interrupting user experience
            pass

    def _serialize_model_config(self, model_config) -> dict:
        """Serialize model config to dictionary."""
        if model_config is None:
            return {}

        config_dict = {}
        for attr in [
            "model_name",
            "model_max_length",
            "torch_dtype_str",
            "attn_implementation",
        ]:
            if hasattr(model_config, attr):
                value = getattr(model_config, attr)
                if value is not None:
                    config_dict[attr] = str(value)

        return config_dict

    def _serialize_generation_config(self, generation_config) -> dict:
        """Serialize generation config to dictionary."""
        if generation_config is None:
            return {}

        config_dict = {}
        for attr in ["max_new_tokens", "temperature", "top_p", "top_k", "sampling"]:
            if hasattr(generation_config, attr):
                value = getattr(generation_config, attr)
                if value is not None:
                    config_dict[attr] = value

        return config_dict
