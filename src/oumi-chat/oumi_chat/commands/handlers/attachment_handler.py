"""File attachment command handler."""

from rich.panel import Panel

from oumi_chat.commands.base_handler import BaseCommandHandler, CommandResult
from oumi_chat.commands.command_parser import ParsedCommand


class AttachmentHandler(BaseCommandHandler):
    """Handles file attachment command: attach."""

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return ["attach"]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle the attach command."""
        if command.command == "attach":
            return self._handle_attach(command)
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
                message=(
                    f"Error attaching file: {str(e)}\\n\\nTraceback:\\n{error_details}"
                ),
                should_continue=False,
            )

    def _display_attachment_result(self, result):
        """Display formatted attachment result to console."""
        # Import here to avoid circular imports
        from rich.text import Text

        # Format the attachment message
        message_text = Text()
        message_text.append("📎 ", style="bold")
        message_text.append(f"Attached: {result.file_info.name}\\n", style="bold cyan")
        message_text.append(f"Type: {result.file_info.type}\\n")
        message_text.append(f"Size: {result.file_info.size_display}\\n")

        if result.context_info:
            message_text.append(f"Context: {result.context_info}\\n", style="dim")

        if result.warning_message:
            message_text.append(f"⚠️  {result.warning_message}", style="yellow")

        # Display in a panel
        panel = Panel(
            message_text,
            title="File Attachment",
            border_style=getattr(self._style, "assistant_border_style", "cyan")
            if hasattr(self, "_style")
            else "cyan",
        )
        self.console.print(panel)

    def _add_attachment_to_conversation(self, result):
        """Add attachment content to conversation history."""
        # Create a special attachment message for the conversation
        attachment_msg = {
            "role": "attachment",
            "file_name": result.file_info.name,
            "file_type": result.file_info.type,
            "content": result.processed_content,
        }

        # If this is text-based content, also store it for token counting
        if result.file_info.type in ["text", "code", "markdown", "pdf", "document"]:
            attachment_msg["text_content"] = result.processed_content

        # Add to conversation history
        self.conversation_history.append(attachment_msg)
