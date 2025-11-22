"""Import/export command handler for conversation data."""

import json
from datetime import datetime
from pathlib import Path

from oumi_chat.commands.base_handler import BaseCommandHandler, CommandResult
from oumi_chat.commands.command_parser import ParsedCommand
from oumi_chat.commands.utilities.export_utilities import ExportUtilities
from oumi_chat.commands.utilities.import_utilities import ImportUtilities
from oumi_chat.utils import validate_and_sanitize_file_path


class ImportExportHandler(BaseCommandHandler):
    """Handles import/export commands: save, import, load, save_history, import_history."""

    def __init__(self, *args, **kwargs):
        """Initialize the import/export handler."""
        super().__init__(*args, **kwargs)
        self.export_utilities = ExportUtilities(self.context)
        self.import_utilities = ImportUtilities(self.context)

        # Auto-save functionality
        self._auto_save = True
        self._setup_chat_cache()

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return ["save", "import", "load", "save_history", "import_history"]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle import/export commands."""
        if command.command == "save":
            return self._handle_save(command)
        elif command.command == "import":
            return self._handle_import(command)
        elif command.command == "load":
            return self._handle_load(command)
        elif command.command == "save_history":
            return self._handle_save_history(command)
        elif command.command == "import_history":
            return self._handle_import_history(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unsupported command: {command.command}",
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
        is_valid, file_path, error_msg = validate_and_sanitize_file_path(raw_path)
        if not is_valid:
            return CommandResult(
                success=False,
                message=f"Invalid file path: {error_msg}",
                should_continue=False,
            )

        # Check for explicit format specification
        format_override = None
        if len(command.args) > 1:
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

        # Call export method
        success, message = self.export_utilities.export_conversation(
            file_path, export_format, self.conversation_history
        )

        return CommandResult(
            success=success,
            message=message,
            should_continue=False,
        )

    def _handle_import(self, command: ParsedCommand) -> CommandResult:
        """Handle /import() command to import conversation data."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="import command requires a file path argument",
                    should_continue=False,
                )

            file_path = command.args[0].strip()

            # Import the conversation
            imported_messages, message = self.import_utilities.import_conversation(
                file_path
            )

            if imported_messages:
                # Add imported messages to conversation
                self.conversation_history.extend(imported_messages)

                # Update context monitor
                self._update_context_in_monitor()

                # Auto-save after import
                self._auto_save_if_enabled()

                return CommandResult(
                    success=True,
                    message=message or f"Imported {len(imported_messages)} messages",
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message=message or f"Failed to import from {file_path}",
                    should_continue=False,
                )

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return CommandResult(
                success=False,
                message=f"Error importing: {str(e)}\\n\\nTraceback:\\n{error_details}",
                should_continue=False,
            )

    def _handle_load(self, command: ParsedCommand) -> CommandResult:
        """Handle /load() command to load a previously saved chat from cache."""
        if not command.args:
            return CommandResult(
                success=False,
                message="load command requires a chat ID argument",
                should_continue=False,
            )

        chat_id = command.args[0].strip()

        try:
            return self._load_chat_by_id(chat_id)

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error loading chat: {str(e)}",
                should_continue=False,
            )

    def _handle_save_history(self, command: ParsedCommand) -> CommandResult:
        """Handle /save_history() to save complete conversation state."""
        # Implementation would include branch state, model config, etc.
        # Simplified for now
        if not command.args:
            return CommandResult(
                success=False,
                message="save_history command requires a file path argument",
                should_continue=False,
            )

        file_path = command.args[0].strip()

        try:
            # Save complete state including branches
            state = {
                "conversation_history": self.conversation_history,
                "timestamp": datetime.now().isoformat(),
                "model_config": self.config.model.__dict__ if self.config.model else {},
            }

            with open(file_path, "w") as f:
                json.dump(state, f, indent=2, default=str)

            return CommandResult(
                success=True,
                message=f"Saved complete history to {file_path}",
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error saving history: {str(e)}",
                should_continue=False,
            )

    def _handle_import_history(self, command: ParsedCommand) -> CommandResult:
        """Handle /import_history() to restore complete conversation state."""
        if not command.args:
            return CommandResult(
                success=False,
                message="import_history command requires a file path argument",
                should_continue=False,
            )

        file_path = command.args[0].strip()

        try:
            with open(file_path, "r") as f:
                state = json.load(f)

            # Restore conversation history
            self.conversation_history.clear()
            self.conversation_history.extend(state.get("conversation_history", []))

            return CommandResult(
                success=True,
                message=f"Restored history from {file_path}",
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error importing history: {str(e)}",
                should_continue=False,
            )

    def _setup_chat_cache(self):
        """Set up the chat cache directory."""
        # Simplified implementation
        pass

    def _load_chat_by_id(self, chat_id: str) -> CommandResult:
        """Load a chat by ID from cache."""
        # Simplified - would load from cache directory
        return CommandResult(
            success=False,
            message="Chat loading not yet implemented",
            should_continue=False,
        )

    def _auto_save_if_enabled(self):
        """Auto-save if enabled."""
        # Simplified
        pass
