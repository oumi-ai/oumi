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
import os
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

    def _validate_and_sanitize_file_path(self, file_path: str) -> tuple[bool, str, str]:
        """Validate and sanitize a file path for security and safety using pathvalidate.

        Args:
            file_path: The file path to validate

        Returns:
            Tuple of (is_valid, sanitized_path, error_message)
        """
        try:
            from pathvalidate import (
                ValidationError,
                is_valid_filepath,
                sanitize_filepath,
            )
        except ImportError:
            return (
                False,
                "",
                "pathvalidate library is required for file path validation",
            )

        if not file_path:
            return False, "", "File path cannot be empty"

        # Check for unmatched quotes before sanitizing (pathvalidate doesn't handle this)
        stripped = file_path.strip()
        quote_chars = ["'", '"']
        for quote in quote_chars:
            if stripped.startswith(quote) and not stripped.endswith(quote):
                return False, "", f"Unmatched quote in file path: {quote}"
            if stripped.endswith(quote) and not stripped.startswith(quote):
                return False, "", f"Unmatched quote in file path: {quote}"

        # Strip whitespace and quotes
        cleaned_path = file_path.strip().strip("\"'")

        # Check if the cleaned path is effectively empty
        if not cleaned_path or cleaned_path.isspace():
            return False, "", "File path is empty or contains only whitespace"

        # Use pathvalidate to sanitize the file path
        try:
            sanitized = sanitize_filepath(
                cleaned_path,
                platform="universal",  # Works on all platforms
                max_len=255,  # Standard filesystem limit
            )
        except ValidationError as e:
            return False, "", f"Invalid file path: {str(e)}"

        # Verify the sanitized path is valid
        if not is_valid_filepath(sanitized, platform="universal"):
            return False, "", "File path contains invalid characters or format"

        # Additional security check - prevent path traversal
        if ".." in sanitized or sanitized.startswith("/"):
            return (
                False,
                "",
                "File path contains potential security risks (path traversal)",
            )

        # Check if the path would create a file with quotes in the name
        if any(quote in Path(sanitized).name for quote in ["'", '"']):
            return False, "", "File name cannot contain quote characters"

        return True, sanitized, ""

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return [
            "attach",
            "save",
            "import",
            "load",
            "save_history",
            "import_history",
            "fetch",
            "shell",
        ]

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
        elif command.command == "save_history":
            return self._handle_save_history(command)
        elif command.command == "import_history":
            return self._handle_import_history(command)
        elif command.command == "fetch":
            return self._handle_fetch(command)
        elif command.command == "shell":
            return self._handle_shell(command)
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
            # Get accurate conversation token count
            conversation_tokens = self._get_conversation_tokens()

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

        # Validate and sanitize the file path
        raw_path = command.args[0]
        is_valid, file_path, error_msg = self._validate_and_sanitize_file_path(raw_path)
        if not is_valid:
            return CommandResult(
                success=False,
                message=f"Invalid file path: {error_msg}",
                should_continue=False,
            )

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

        if hasattr(result, "processing_info") and result.processing_info:
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
                "file_type": str(result.file_info.file_type),
            },
        }
        self.conversation_history.append(attachment_message)

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
            # Check if we have branch manager for comprehensive format
            branch_manager = getattr(self.context, "branch_manager", None)

            if (
                branch_manager
                and hasattr(branch_manager, "branches")
                and len(branch_manager.branches) > 1
            ):
                # Use comprehensive format when multiple branches exist
                chat_data = self._build_comprehensive_history()
            else:
                # Use basic format for simple conversations
                chat_data = {
                    "chat_id": self.chat_id,
                    "model_name": getattr(self.config.model, "model_name", "default"),
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "conversation_history": self.conversation_history,
                    "model_config": self._serialize_model_config(self.config.model),
                    "generation_config": self._serialize_generation_config(
                        getattr(self.config, "generation", None)
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

    def _handle_save_history(self, command: ParsedCommand) -> CommandResult:
        """Handle the /save_history(path) command to save complete conversation state."""
        if not command.args:
            return CommandResult(
                success=False,
                message="save_history command requires a file path argument",
                should_continue=False,
            )

        # Validate and sanitize the file path
        raw_path = command.args[0]
        is_valid, file_path, error_msg = self._validate_and_sanitize_file_path(raw_path)
        if not is_valid:
            return CommandResult(
                success=False,
                message=f"Invalid file path: {error_msg}",
                should_continue=False,
            )

        try:
            # Build comprehensive history data
            history_data = self._build_comprehensive_history()

            # Ensure .json extension
            path_obj = Path(file_path)
            if path_obj.suffix.lower() != ".json":
                file_path = str(path_obj.with_suffix(".json"))

            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            # Count total elements saved
            branch_count = len(history_data.get("branches", {}))
            total_messages = sum(
                len(branch.get("conversation_history", []))
                for branch in history_data.get("branches", {}).values()
            )
            command_count = len(history_data.get("command_history", []))

            return CommandResult(
                success=True,
                message=(
                    f"Saved complete conversation history to {file_path}\n"
                    f"📊 Saved: {branch_count} branches, {total_messages} messages, "
                    f"{command_count} commands, full config & metadata"
                ),
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error saving history: {str(e)}",
                should_continue=False,
            )

    def _handle_import_history(self, command: ParsedCommand) -> CommandResult:
        """Handle the /import_history(path) command to restore complete conversation state."""
        if not command.args:
            return CommandResult(
                success=False,
                message="import_history command requires a file path argument",
                should_continue=False,
            )

        file_path = command.args[0].strip()

        try:
            # Load history data
            with open(file_path, encoding="utf-8") as f:
                history_data = json.load(f)

            # Validate schema
            if not self._validate_history_schema(history_data):
                return CommandResult(
                    success=False,
                    message="Invalid history file format - see Oumi history schema documentation",
                    should_continue=False,
                )

            # Restore conversation state
            success, message = self._restore_conversation_state(history_data)

            if success:
                # Count what was restored
                branch_count = len(history_data.get("branches", {}))
                total_messages = sum(
                    len(branch.get("conversation_history", []))
                    for branch in history_data.get("branches", {}).values()
                )
                command_count = len(history_data.get("command_history", []))

                return CommandResult(
                    success=True,
                    message=(
                        f"Restored conversation history from {file_path}\n"
                        f"📊 Restored: {branch_count} branches, {total_messages} messages, "
                        f"{command_count} commands, config & metadata"
                    ),
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message=f"Failed to restore history: {message}",
                    should_continue=False,
                )

        except FileNotFoundError:
            return CommandResult(
                success=False,
                message=f"History file not found: {file_path}",
                should_continue=False,
            )
        except json.JSONDecodeError as e:
            return CommandResult(
                success=False,
                message=f"Invalid JSON in history file: {str(e)}",
                should_continue=False,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error importing history: {str(e)}",
                should_continue=False,
            )

    def _build_comprehensive_history(self) -> dict:
        """Build comprehensive history data structure."""
        # Get current timestamp
        current_time = datetime.now().isoformat()

        # Get branch manager if available
        branch_manager = getattr(self.context, "branch_manager", None)

        # Build branches data
        branches = {}
        if branch_manager:
            for branch_id, branch in branch_manager.branches.items():
                branches[branch_id] = {
                    "id": branch.id,
                    "name": branch.name,
                    "created_at": branch.created_at.isoformat()
                    if branch.created_at
                    else current_time,
                    "last_active": branch.last_active.isoformat()
                    if branch.last_active
                    else current_time,
                    "parent_branch_id": branch.parent_branch_id,
                    "branch_point_index": branch.branch_point_index,
                    "conversation_history": branch.conversation_history,
                    "model_name": branch.model_name,
                    "engine_type": branch.engine_type,
                    "model_config": branch.model_config,
                    "generation_config": branch.generation_config,
                }
            current_branch_id = branch_manager.current_branch_id
        else:
            # No branch manager - create main branch from current conversation
            branches["main"] = {
                "id": "main",
                "name": "Main",
                "created_at": current_time,
                "last_active": current_time,
                "parent_branch_id": None,
                "branch_point_index": 0,
                "conversation_history": self.conversation_history,
                "model_name": getattr(self.config.model, "model_name", "default"),
                "engine_type": getattr(self.config, "engine", "unknown"),
                "model_config": self._serialize_model_config(self.config.model),
                "generation_config": self._serialize_generation_config(
                    getattr(self.config, "generation", None)
                ),
            }
            current_branch_id = "main"

        # Build comprehensive history structure
        history_data = {
            "schema_version": "1.0.0",
            "format": "oumi_conversation_history",
            "created_at": current_time,
            "source": "oumi_interactive_chat",
            # Session information
            "session": {
                "chat_id": getattr(
                    self,
                    "chat_id",
                    f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                ),
                "current_branch_id": current_branch_id,
                "total_session_time": None,  # Could be calculated if we track start time
                "oumi_version": "latest",  # Could get actual version
            },
            # Model and configuration
            "configuration": {
                "model": self._serialize_model_config(self.config.model),
                "generation": self._serialize_generation_config(
                    getattr(self.config, "generation", None)
                ),
                "engine": getattr(self.config, "engine", "unknown"),
                "style": self._serialize_style_config(
                    getattr(self.config, "style", None)
                ),
                "inference_params": self._get_current_inference_params(),
            },
            # All conversation branches
            "branches": branches,
            # Command history (placeholder - would need to be tracked)
            "command_history": self._get_command_history(),
            # Metadata about attachments and operations
            "attachments": self._get_attachment_metadata(),
            # Statistics
            "statistics": {
                "total_branches": len(branches),
                "total_messages": sum(
                    len(branch["conversation_history"]) for branch in branches.values()
                ),
                "total_user_messages": sum(
                    len(
                        [
                            msg
                            for msg in branch["conversation_history"]
                            if msg.get("role") == "user"
                        ]
                    )
                    for branch in branches.values()
                ),
                "total_assistant_messages": sum(
                    len(
                        [
                            msg
                            for msg in branch["conversation_history"]
                            if msg.get("role") == "assistant"
                        ]
                    )
                    for branch in branches.values()
                ),
                "estimated_tokens": self._get_conversation_tokens(),
                "created_at": current_time,
            },
        }

        return history_data

    def _serialize_style_config(self, style_config) -> dict:
        """Serialize style configuration."""
        if style_config is None:
            return {}

        style_dict = {}
        for attr in [
            "user_prompt_style",
            "assistant_title_style",
            "assistant_border_style",
            "analysis_text_style",
            "analysis_title_style",
            "analysis_border_style",
            "error_style",
            "success_style",
            "use_emoji",
            "expand_panels",
        ]:
            if hasattr(style_config, attr):
                value = getattr(style_config, attr)
                if value is not None:
                    style_dict[attr] = value

        return style_dict

    def _get_current_inference_params(self) -> dict:
        """Get current inference parameters."""
        # This would ideally get from the inference engine's current state
        return {
            "temperature": getattr(self.config.generation, "temperature", None)
            if hasattr(self.config, "generation")
            else None,
            "top_p": getattr(self.config.generation, "top_p", None)
            if hasattr(self.config, "generation")
            else None,
            "max_tokens": getattr(self.config.generation, "max_new_tokens", None)
            if hasattr(self.config, "generation")
            else None,
            "sampling": getattr(self.config.generation, "sampling", None)
            if hasattr(self.config, "generation")
            else None,
        }

    def _get_command_history(self) -> list:
        """Get command history - placeholder for future implementation."""
        # This would require tracking commands throughout the session
        # For now, return empty list with note
        return [
            {
                "note": "Command history tracking not yet implemented",
                "timestamp": datetime.now().isoformat(),
                "type": "system_note",
            }
        ]

    def _get_attachment_metadata(self) -> list:
        """Get attachment metadata from conversation history."""
        attachments = []
        for msg in self.conversation_history:
            if msg.get("role") == "attachment":
                attachments.append(
                    {
                        "filename": msg.get("filename", "unknown"),
                        "file_type": msg.get("file_type", "unknown"),
                        "size_bytes": msg.get("size_bytes", 0),
                        "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                        "content_preview": msg.get("text_content", "")[:200] + "..."
                        if len(msg.get("text_content", "")) > 200
                        else msg.get("text_content", ""),
                    }
                )
        return attachments

    def _validate_history_schema(self, history_data: dict) -> bool:
        """Validate the history data schema."""
        required_fields = ["schema_version", "format", "branches"]

        for field in required_fields:
            if field not in history_data:
                return False

        # Check format
        if history_data.get("format") != "oumi_conversation_history":
            return False

        # Check branches structure
        branches = history_data.get("branches", {})
        if not isinstance(branches, dict):
            return False

        for branch_id, branch in branches.items():
            required_branch_fields = ["id", "conversation_history"]
            for field in required_branch_fields:
                if field not in branch:
                    return False

            # Check conversation history is a list
            if not isinstance(branch.get("conversation_history"), list):
                return False

        return True

    def _restore_conversation_state(self, history_data: dict) -> tuple[bool, str]:
        """Restore conversation state from history data."""
        try:
            branches = history_data.get("branches", {})
            session = history_data.get("session", {})

            # Get or create branch manager
            branch_manager = getattr(self.context, "branch_manager", None)
            if not branch_manager:
                from oumi.core.commands.conversation_branches import (
                    ConversationBranchManager,
                )

                branch_manager = ConversationBranchManager()
                self.context._branch_manager = branch_manager

            # Clear existing branches
            branch_manager.branches.clear()

            # Restore branches
            from oumi.core.commands.conversation_branches import ConversationBranch

            for branch_id, branch_data in branches.items():
                branch = ConversationBranch(
                    id=branch_data["id"],
                    name=branch_data.get("name"),
                    created_at=datetime.fromisoformat(
                        branch_data.get("created_at", datetime.now().isoformat())
                    ),
                    last_active=datetime.fromisoformat(
                        branch_data.get("last_active", datetime.now().isoformat())
                    ),
                    parent_branch_id=branch_data.get("parent_branch_id"),
                    branch_point_index=branch_data.get("branch_point_index", 0),
                    conversation_history=branch_data.get("conversation_history", []),
                    model_name=branch_data.get("model_name"),
                    engine_type=branch_data.get("engine_type"),
                    model_config=branch_data.get("model_config"),
                    generation_config=branch_data.get("generation_config"),
                )
                branch_manager.branches[branch_id] = branch

            # Set current branch
            current_branch_id = session.get("current_branch_id", "main")
            if current_branch_id in branch_manager.branches:
                branch_manager.current_branch_id = current_branch_id
                # Update conversation history to match current branch
                current_branch = branch_manager.branches[current_branch_id]
                self.conversation_history.clear()
                self.conversation_history.extend(current_branch.conversation_history)
            else:
                # Default to main branch if available
                if "main" in branch_manager.branches:
                    branch_manager.current_branch_id = "main"
                    main_branch = branch_manager.branches["main"]
                    self.conversation_history.clear()
                    self.conversation_history.extend(main_branch.conversation_history)
                elif branches:
                    # Use first available branch
                    first_branch_id = list(branches.keys())[0]
                    branch_manager.current_branch_id = first_branch_id
                    first_branch = branch_manager.branches[first_branch_id]
                    self.conversation_history.clear()
                    self.conversation_history.extend(first_branch.conversation_history)

            # Update context monitor
            self._update_context_in_monitor()

            return True, "Successfully restored conversation state"

        except Exception as e:
            return False, f"Error restoring state: {str(e)}"

    def _handle_fetch(self, command: ParsedCommand) -> CommandResult:
        """Handle the /fetch(url) command to retrieve web content."""
        if not command.args:
            return CommandResult(
                success=False,
                message="fetch command requires a URL argument",
                should_continue=False,
            )

        url = command.args[0].strip()

        try:
            # Import requests and beautifulsoup4 with helpful error messages
            try:
                import requests
                from bs4 import BeautifulSoup
            except ImportError:
                return CommandResult(
                    success=False,
                    message="Web fetching requires additional dependencies. Install with: pip install 'oumi[interactive]'",
                    should_continue=False,
                )

            # Add protocol if missing
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"

            # Use context manager for accurate token estimation and budget calculation
            context_manager = self.context.context_window_manager

            # Build current conversation text for accurate token counting
            conversation_text = ""
            for msg in self.conversation_history:
                if isinstance(msg, dict):
                    # Handle regular messages
                    if "content" in msg:
                        conversation_text += str(msg["content"]) + "\n"
                    # Handle attachment messages
                    elif msg.get("role") == "attachment" and "text_content" in msg:
                        conversation_text += str(msg["text_content"]) + "\n"

            conversation_tokens = context_manager.estimate_tokens(conversation_text)
            budget = context_manager.calculate_budget(conversation_tokens)
            available_tokens = budget.available_for_content

            # Make the web request with timeout and user agent
            headers = {"User-Agent": "Oumi-AI-Assistant/1.0 (Interactive Chat Bot)"}

            self.console.print(f"🌐 Fetching content from {url}...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Parse HTML content
            if "text/html" in response.headers.get("content-type", ""):
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script, style, and other non-content elements
                for element in soup(
                    ["script", "style", "nav", "header", "footer", "aside"]
                ):
                    element.decompose()

                # Extract text content
                content = soup.get_text(separator=" ", strip=True)
                content_type = "HTML (parsed text)"
            else:
                content = response.text
                content_type = response.headers.get("content-type", "text/plain")

            # Check if content fits in available context using accurate tokenization
            content_tokens = context_manager.estimate_tokens(content)
            if content_tokens > available_tokens:
                # Truncate content to fit available space
                # Use binary search to find the right truncation point
                left, right = 0, len(content)
                truncated_content = content

                while left < right:
                    mid = (left + right + 1) // 2
                    test_content = content[:mid]
                    test_tokens = context_manager.estimate_tokens(test_content)

                    if test_tokens <= available_tokens:
                        left = mid
                        truncated_content = test_content
                    else:
                        right = mid - 1

                content = (
                    truncated_content
                    + f"\n\n[Content truncated from {content_tokens:,} to {context_manager.estimate_tokens(truncated_content):,} tokens due to context window limits]"
                )

            # Create attachment-style message for the conversation
            fetch_message = {
                "role": "attachment",
                "text_content": content,
                "attachment_info": {
                    "type": "web_fetch",
                    "url": url,
                    "content_type": content_type,
                    "size_chars": len(content),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            # Add to conversation history
            self.conversation_history.append(fetch_message)

            # Update context monitor
            self._update_context_in_monitor()

            # Display success with content info
            from rich.panel import Panel

            info_content = f"**URL:** {url}\n**Type:** {content_type}\n**Size:** {len(content):,} characters"

            if len(content) > 500:
                info_content += f"\n**Preview:** {content[:200]}..."
            else:
                info_content += f"\n**Content:** {content}"

            panel = Panel(
                info_content,
                title="🌐 Web Content Fetched",
                border_style="green",
                padding=(1, 2),
            )
            self.console.print(panel)

            return CommandResult(
                success=True,
                message=f"Fetched {len(content):,} characters from {url}",
                should_continue=False,
            )

        except requests.exceptions.Timeout:
            return CommandResult(
                success=False,
                message=f"Request timeout: {url} took longer than 30 seconds",
                should_continue=False,
            )
        except requests.exceptions.RequestException as e:
            return CommandResult(
                success=False,
                message=f"Failed to fetch {url}: {str(e)}",
                should_continue=False,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error fetching web content: {str(e)}",
                should_continue=False,
            )

    def _handle_shell(self, command: ParsedCommand) -> CommandResult:
        """Handle the /shell(command) command to execute local shell commands."""
        if not command.args:
            return CommandResult(
                success=False,
                message="shell command requires a command argument",
                should_continue=False,
            )

        shell_command = " ".join(command.args).strip()

        # Security restrictions - block dangerous commands
        dangerous_patterns = [
            "rm ",
            "del ",
            "format ",
            "fdisk",
            "sudo ",
            "su ",
            "chmod +x",
            "chown",
            "wget ",
            "curl ",
            "ssh ",
            "scp ",
            "nc ",
            "netcat",
            "nmap",
            "telnet",
            "python -c",
            "perl -e",
            "ruby -e",
            "bash -c",
            "sh -c",
            "eval",
            "&",
            "&&",
            "||",
            ";",
            "`",
            "$(",
            "mkfs",
            "mount",
            "umount",
            "kill",
            "killall",
            "pkill",
            "systemctl",
        ]

        shell_lower = shell_command.lower()
        for pattern in dangerous_patterns:
            if pattern in shell_lower:
                return CommandResult(
                    success=False,
                    message=f"Command blocked for security: contains '{pattern}'. Shell commands are restricted to safe, read-only operations.",
                    should_continue=False,
                )

        # Additional length restriction
        if len(shell_command) > 200:
            return CommandResult(
                success=False,
                message="Command too long. Maximum 200 characters allowed for security.",
                should_continue=False,
            )

        try:
            import subprocess

            # Use accurate context management like attach and fetch commands
            context_manager = self.context.context_window_manager

            # Build current conversation text for accurate token counting
            conversation_text = ""
            for msg in self.conversation_history:
                if isinstance(msg, dict):
                    # Handle regular messages
                    if "content" in msg:
                        conversation_text += str(msg["content"]) + "\n"
                    # Handle attachment messages
                    elif msg.get("role") == "attachment" and "text_content" in msg:
                        conversation_text += str(msg["text_content"]) + "\n"
                    elif msg.get("role") == "attachment" and "content" in msg:
                        conversation_text += str(msg["content"]) + "\n"

            conversation_tokens = context_manager.estimate_tokens(conversation_text)
            budget = context_manager.calculate_budget(conversation_tokens)
            available_tokens = budget.available_for_content

            self.console.print(f"🔧 Executing: {shell_command}")

            # Execute with timeout and capture both stdout and stderr
            result = subprocess.run(
                shell_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd(),  # Execute in current directory
            )

            # Combine stdout and stderr
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}"
            if result.stderr:
                if output:
                    output += "\n\n"
                output += f"STDERR:\n{result.stderr}"

            if not output:
                output = "[No output produced]"

            # Add return code info
            output += f"\n\nReturn code: {result.returncode}"

            # Check if output fits in available context using accurate tokenization
            output_tokens = context_manager.estimate_tokens(output)
            if output_tokens > available_tokens:
                # Truncate output to fit available space using binary search
                left, right = 0, len(output)
                truncated_output = output

                while left < right:
                    mid = (left + right + 1) // 2
                    test_output = output[:mid]
                    test_tokens = context_manager.estimate_tokens(test_output)

                    if test_tokens <= available_tokens:
                        left = mid
                        truncated_output = test_output
                    else:
                        right = mid - 1

                output = (
                    truncated_output
                    + f"\n\n[Output truncated from {output_tokens:,} to {context_manager.estimate_tokens(truncated_output):,} tokens due to context window limits]"
                )

            # Create attachment-style message for the conversation
            shell_message = {
                "role": "attachment",
                "text_content": output,
                "attachment_info": {
                    "type": "shell_command",
                    "command": shell_command,
                    "return_code": result.returncode,
                    "size_chars": len(output),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            # Add to conversation history
            self.conversation_history.append(shell_message)

            # Update context monitor
            self._update_context_in_monitor()

            # Display result
            from rich.panel import Panel

            status_color = "green" if result.returncode == 0 else "red"
            status_text = (
                "✅ Success"
                if result.returncode == 0
                else f"❌ Failed (code {result.returncode})"
            )

            info_content = f"**Command:** {shell_command}\n**Status:** {status_text}\n**Output Size:** {len(output):,} characters"

            # Show preview of output
            preview_length = 300
            if len(output) > preview_length:
                info_content += f"\n**Preview:**\n{output[:preview_length]}..."
            else:
                info_content += f"\n**Output:**\n{output}"

            panel = Panel(
                info_content,
                title="🔧 Shell Command Result",
                border_style=status_color,
                padding=(1, 2),
            )
            self.console.print(panel)

            return CommandResult(
                success=True,
                message=f"Executed command '{shell_command}' (exit code: {result.returncode})",
                should_continue=False,
            )

        except subprocess.TimeoutExpired:
            return CommandResult(
                success=False,
                message=f"Command timeout: '{shell_command}' took longer than 30 seconds",
                should_continue=False,
            )
        except subprocess.SubprocessError as e:
            return CommandResult(
                success=False,
                message=f"Shell command failed: {str(e)}",
                should_continue=False,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error executing shell command: {str(e)}",
                should_continue=False,
            )
