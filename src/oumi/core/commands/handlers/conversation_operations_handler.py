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

"""Conversation operations command handler."""

from typing import Optional

from oumi.core.commands.base_handler import BaseCommandHandler, CommandResult
from oumi.core.commands.command_parser import ParsedCommand
from oumi.core.commands.conversation_renderer import ConversationRenderer


class ConversationOperationsHandler(BaseCommandHandler):
    """Handles conversation manipulation commands: delete, regen, clear, compact, full_thoughts, clear_thoughts."""

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return [
            "delete",
            "regen",
            "clear",
            "compact",
            "full_thoughts",
            "clear_thoughts",
            "show",
            "render",
        ]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a conversation operations command."""
        if command.command == "delete":
            return self._handle_delete(command)
        elif command.command == "regen":
            return self._handle_regen(command)
        elif command.command == "clear":
            return self._handle_clear(command)
        elif command.command == "compact":
            return self._handle_compact(command)
        elif command.command == "full_thoughts":
            return self._handle_full_thoughts(command)
        elif command.command == "clear_thoughts":
            return self._handle_clear_thoughts(command)
        elif command.command == "show":
            return self._handle_show(command)
        elif command.command == "render":
            return self._handle_render(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unsupported command: {command.command}",
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

            # Delete the last turn (could be user+assistant or just user)
            deleted_count = self._delete_last_turn()

            if deleted_count > 0:
                self._update_context_in_monitor()
                return CommandResult(
                    success=True,
                    message=f"Deleted {deleted_count} message(s) from conversation",
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message="No messages to delete",
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error deleting messages: {str(e)}",
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

            # Find the last user message
            last_user_input = self._get_last_user_input()

            if not last_user_input:
                return CommandResult(
                    success=False,
                    message="No user message found to regenerate response for",
                    should_continue=False,
                )

            # Remove the last assistant response if it exists
            removed_assistant = False
            if (
                self.conversation_history
                and self.conversation_history[-1].get("role") == "assistant"
            ):
                self.conversation_history.pop()
                removed_assistant = True

            # If we didn't remove an assistant response, and the last message is already
            # a user message, we need to remove it too to avoid duplicate user messages
            if (
                not removed_assistant
                and self.conversation_history
                and self.conversation_history[-1].get("role") == "user"
                and self.conversation_history[-1].get("content") == last_user_input
            ):
                self.conversation_history.pop()

            # Update context monitor
            self._update_context_in_monitor()

            # Return with user input override to regenerate
            return CommandResult(
                success=True,
                message="Regenerating last response...",
                should_continue=True,
                user_input_override=last_user_input,
                is_regeneration=True,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error regenerating response: {str(e)}",
                should_continue=False,
            )

    def _handle_clear(self, command: ParsedCommand) -> CommandResult:
        """Handle the /clear() command to clear entire conversation history."""
        try:
            # Count messages before clearing
            message_count = len(self.conversation_history)

            if message_count == 0:
                return CommandResult(
                    success=True,
                    message="Conversation history is already empty",
                    should_continue=False,
                )

            # Clear conversation history
            self.conversation_history.clear()

            # Update context monitor
            self._update_context_in_monitor()

            return CommandResult(
                success=True,
                message=f"Cleared {message_count} message(s) from conversation history",
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error clearing conversation: {str(e)}",
                should_continue=False,
            )

    def _handle_compact(self, command: ParsedCommand) -> CommandResult:
        """Handle the /compact() command to compress conversation history."""
        try:
            # Show compaction status
            original_tokens = self._estimate_conversation_tokens()
            original_count = len(self.conversation_history)

            # Allow compaction even for short conversations - let the user decide

            # Use compaction engine
            result = self.context.compaction_engine.compact_conversation(
                self.conversation_history
            )

            if result.success:
                # Update conversation history with compacted version
                self.conversation_history.clear()
                self.conversation_history.extend(result.compacted_conversation)

                # Update context monitor
                self._update_context_in_monitor()

                new_tokens = self._estimate_conversation_tokens()
                new_count = len(self.conversation_history)

                savings_tokens = original_tokens - new_tokens
                savings_messages = original_count - new_count

                return CommandResult(
                    success=True,
                    message=(
                        f"Compacted conversation: {original_count} → {new_count} messages, "
                        f"~{original_tokens} → ~{new_tokens} tokens (saved ~{savings_tokens} tokens)"
                    ),
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message=result.error_message or "Failed to compact conversation",
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error compacting conversation: {str(e)}",
                should_continue=False,
            )

    def _handle_full_thoughts(self, command: ParsedCommand) -> CommandResult:
        """Handle the /full_thoughts() command to toggle thinking display mode."""
        try:
            # Toggle the thinking display mode
            processor = self.context.thinking_processor
            current_mode = processor.get_display_mode()

            if current_mode == "compressed":
                processor.set_display_mode("full")
                new_mode = "full"
                description = "complete thinking chains and reasoning"
            else:
                processor.set_display_mode("compressed")
                new_mode = "compressed"
                description = "brief summaries of thinking content"

            return CommandResult(
                success=True,
                message=f"Thinking display mode set to '{new_mode}' - showing {description}",
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

            processor = self.context.thinking_processor

            for message in self.conversation_history:
                if message.get("role") == "assistant":
                    processed_count += 1
                    original_content = message.get("content", "")

                    # Clean thinking content
                    cleaned_content = processor.clean_thinking_content(original_content)

                    if cleaned_content != original_content:
                        message["content"] = cleaned_content
                        cleaned_count += 1

            # Update context monitor
            self._update_context_in_monitor()

            if cleaned_count > 0:
                return CommandResult(
                    success=True,
                    message=(
                        f"Removed thinking content from {cleaned_count} assistant message(s). "
                        "Responses preserved while reasoning sections cleaned up."
                    ),
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=True,
                    message="No thinking content found to remove",
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error clearing thinking content: {str(e)}",
                should_continue=False,
            )

    def _delete_last_turn(self) -> int:
        """Delete the last conversation turn and return number of messages deleted.

        Returns:
            Number of messages deleted.
        """
        if not self.conversation_history:
            return 0

        deleted_count = 0

        # If last message is assistant, delete it
        if self.conversation_history[-1].get("role") == "assistant":
            self.conversation_history.pop()
            deleted_count += 1

        # If there's still a user message, delete it too (complete turn)
        if (
            self.conversation_history
            and self.conversation_history[-1].get("role") == "user"
        ):
            self.conversation_history.pop()
            deleted_count += 1

        return deleted_count

    def _get_last_user_input(self) -> Optional[str]:
        """Get the content of the last user message.

        Returns:
            The last user input, or None if not found.
        """
        # Search backwards for the last user message
        for message in reversed(self.conversation_history):
            if message.get("role") == "user":
                return message.get("content", "")

        return None

    def _handle_show(self, command: ParsedCommand) -> CommandResult:
        """Handle the /show(pos) command to view a specific conversation position."""
        try:
            # Default to showing the most recent assistant message if no position specified
            position = None
            if command.args:
                try:
                    position = int(command.args[0].strip())
                except ValueError:
                    return CommandResult(
                        success=False,
                        message="Position must be a valid integer",
                        should_continue=False,
                    )

            # Collect all assistant messages with their positions and content
            assistant_messages = []
            for i, msg in enumerate(self.conversation_history):
                if msg.get("role") == "assistant":
                    assistant_messages.append((len(assistant_messages) + 1, i, msg))

            if not assistant_messages:
                return CommandResult(
                    success=False,
                    message="No assistant messages in conversation history",
                    should_continue=False,
                )

            # Determine which message to show
            if position is None:
                # Show most recent (last) assistant message
                display_position, msg_index, message = assistant_messages[-1]
                position_text = f"most recent (#{display_position})"
            else:
                # Validate position
                if position < 1 or position > len(assistant_messages):
                    return CommandResult(
                        success=False,
                        message=f"Position {position} is out of range (1-{len(assistant_messages)})",
                        should_continue=False,
                    )
                
                # Show specified position
                display_position, msg_index, message = assistant_messages[position - 1]
                position_text = f"#{position}"

            # Get the corresponding user message (if any)
            user_message = None
            if msg_index > 0 and self.conversation_history[msg_index - 1].get("role") == "user":
                user_message = self.conversation_history[msg_index - 1]

            # Display the conversation turn
            from rich.panel import Panel
            from rich.text import Text

            self.console.print()

            # Show user message if available
            if user_message:
                user_content = user_message.get("content", "")
                user_panel = Panel(
                    Text(user_content, style="white"),
                    title="[bold blue]You[/bold blue]",
                    border_style="blue",
                    padding=(1, 2),
                    expand=getattr(self.config.style, "expand_panels", False),
                )
                self.console.print(user_panel)

            # Show assistant message with thinking processing
            assistant_content = message.get("content", "")
            
            # Process thinking content if available
            thinking_processor = self.context.thinking_processor
            thinking_result = thinking_processor.extract_thinking(assistant_content)

            if thinking_result.has_thinking:
                # Show thinking content
                thinking_processor.render_thinking(
                    thinking_result, self.console, self.config.style, compressed=False
                )
                display_content = thinking_result.final_content
            else:
                display_content = assistant_content

            # Show assistant response
            assistant_panel = Panel(
                Text(display_content, style="white"),
                title=f"[bold cyan]Assistant ({position_text})[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
                expand=getattr(self.config.style, "expand_panels", False),
            )
            self.console.print(assistant_panel)

            return CommandResult(
                success=True,
                message=f"Displayed conversation position {position_text}",
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error showing conversation position: {str(e)}",
                should_continue=False,
            )

    def _handle_render(self, command: ParsedCommand) -> CommandResult:
        """Handle the /render(path) command to record conversation playback with asciinema."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="render command requires a file path argument (e.g., /render(conversation.cast))",
                    should_continue=False,
                )

            output_path = command.args[0].strip()
            
            # Ensure .cast extension
            from pathlib import Path
            path_obj = Path(output_path)
            if path_obj.suffix.lower() != ".cast":
                output_path = str(path_obj.with_suffix(".cast"))

            # Check if asciinema is available
            import subprocess
            import shutil
            
            if not shutil.which("asciinema"):
                return CommandResult(
                    success=False,
                    message="asciinema is not installed. Install with: pip install asciinema",
                    should_continue=False,
                )

            # Create the conversation renderer
            renderer = ConversationRenderer(
                conversation_history=self.conversation_history,
                console=self.console,
                config=self.config,
                thinking_processor=self.context.thinking_processor
            )

            # Start asciinema recording and play back conversation
            success, message = renderer.render_to_asciinema(output_path)

            return CommandResult(
                success=success,
                message=message,
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error rendering conversation: {str(e)}",
                should_continue=False,
            )
