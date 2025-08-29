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
from oumi.utils.logging import logger


class ConversationOperationsHandler(BaseCommandHandler):
    """Handles conversation manipulation commands.

    Supported commands: delete, regen, clear, compact, full_thoughts, clear_thoughts, edit.
    """

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
            "edit",
        ]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a conversation operations command."""
        logger.info(f"🎯 ConversationOperationsHandler: Received command '{command.command}' with args: {command.args}")
        logger.info(f"🎯 Current conversation length: {len(self.conversation_history) if hasattr(self, 'conversation_history') and self.conversation_history else 0}")
        
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
        elif command.command == "edit":
            return self._handle_edit(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unsupported command: {command.command}",
                should_continue=False,
            )

    def _handle_delete(self, command: ParsedCommand) -> CommandResult:
        """Handle the /delete([index]) command to remove conversation messages.
        
        Args:
            command: Parsed command, optionally with index argument
                    - No args: Delete last turn (original behavior)
                    - With index: Delete message at specified position
        """
        logger.info(f"🗑️  DELETE: Starting delete operation with args: {command.args}")
        try:
            if not self.conversation_history:
                return CommandResult(
                    success=False,
                    message="No conversation history to delete",
                    should_continue=False,
                )

            # Check if index is provided
            if command.args and len(command.args) > 0:
                try:
                    index = int(command.args[0])
                    logger.info(f"🗑️  DELETE: Deleting message at index {index}")
                    
                    # Validate index bounds (conversation might have changed since UI was rendered)
                    if index < 0 or index >= len(self.conversation_history):
                        logger.error(f"🗑️  DELETE: Index {index} out of bounds (conversation has {len(self.conversation_history)} messages)")
                        return CommandResult(
                            success=False,
                            message=f"Message index {index} is out of bounds. Conversation has {len(self.conversation_history)} messages (indices 0-{len(self.conversation_history)-1}). Please refresh the page to sync with current conversation state.",
                            should_continue=False,
                        )
                    
                    deleted_count = self._delete_at_index(index)
                    logger.info(f"🗑️  DELETE: Successfully deleted {deleted_count} message(s) at index {index}")
                        
                except (ValueError, IndexError):
                    logger.error(f"🗑️  DELETE: Error deleting at index {command.args[0]}")
                    return CommandResult(
                        success=False,
                        message=f"Invalid message index: {command.args[0]}",
                        should_continue=False,
                    )
            else:
                # Delete the last turn (could be user+assistant or just user)
                logger.info(f"🗑️  DELETE: Deleting last turn")
                deleted_count = self._delete_last_turn()
                logger.info(f"🗑️  DELETE: Successfully deleted {deleted_count} message(s) from last turn")

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
        """Handle the /regen([index]) command to regenerate assistant responses.
        
        Args:
            command: Parsed command, optionally with index argument
                    - No args: Regenerate last assistant response (original behavior)
                    - With index: Regenerate response at specified message position
        """
        logger.info(f"🔄 REGEN: Starting regeneration with args: {command.args}")
        try:
            if not self.conversation_history:
                return CommandResult(
                    success=False,
                    message="No conversation history to regenerate from",
                    should_continue=False,
                )

            # Check if index is provided
            if command.args and len(command.args) > 0:
                try:
                    index = int(command.args[0])
                    logger.info(f"🔄 REGEN: Regenerating message at index {index}")
                    result = self._regen_at_index(index)
                    logger.info(f"🔄 REGEN: Index-based regeneration result: {result.success} - {result.message}")
                    return result
                except (ValueError, IndexError) as e:
                    logger.error(f"🔄 REGEN: Error regenerating at index {command.args[0]}: {e}")
                    return CommandResult(
                        success=False,
                        message=f"Invalid message index: {command.args[0]} - {str(e)}",
                        should_continue=False,
                    )
            else:
                # Original behavior: regenerate last response
                logger.info(f"🔄 REGEN: Regenerating last response")
                result = self._regen_last_response()
                logger.info(f"🔄 REGEN: Last response regeneration result: {result.success} - {result.message}")
                return result

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error regenerating message: {str(e)}",
                should_continue=False,
            )

    def _regen_last_response(self) -> CommandResult:
        """Regenerate the last assistant response (original regen behavior)."""
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

    def _regen_at_index(self, index: int) -> CommandResult:
        """Regenerate response at specific index.

        Args:
            index: Zero-based index of assistant message to regenerate

        Returns:
            CommandResult with regeneration request
        """
        if not self.conversation_history:
            return CommandResult(
                success=False,
                message="No conversation history to regenerate from",
                should_continue=False,
            )

        if index < 0 or index >= len(self.conversation_history):
            raise IndexError(f"Message index {index} out of range (0-{len(self.conversation_history)-1})")

        # Check if the message at index is an assistant message
        target_message = self.conversation_history[index]
        if target_message.get("role") != "assistant":
            return CommandResult(
                success=False,
                message=f"Message at index {index} is not an assistant response",
                should_continue=False,
            )

        # Find the preceding user message to regenerate from
        user_input = None
        for i in range(index - 1, -1, -1):
            if self.conversation_history[i].get("role") == "user":
                user_input = self.conversation_history[i].get("content")
                logger.info(f"🔄 REGEN: Found user message at index {i} for regeneration")
                logger.info(f"🔄 REGEN: User input content preview: {user_input[:100] if user_input else 'None'}...")
                break

        if not user_input:
            return CommandResult(
                success=False,
                message=f"No user message found before assistant message at index {index}",
                should_continue=False,
            )

        # Remove the target assistant message and all messages after it
        # This ensures a clean regeneration from the target point
        logger.info(f"🔄 REGEN: Truncating conversation from {len(self.conversation_history)} to {index} messages")
        self.conversation_history = self.conversation_history[:index]
        logger.info(f"🔄 REGEN: Conversation now has {len(self.conversation_history)} messages")

        # Update context monitor
        self._update_context_in_monitor()

        # Return with user input override to regenerate
        logger.info(f"🔄 REGEN: Returning user_input_override for regeneration: {user_input[:100] if user_input else 'None'}...")
        return CommandResult(
            success=True,
            message=f"Regenerating response at position {index}...",
            should_continue=True,
            user_input_override=user_input,
            is_regeneration=True,
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
            original_tokens = self._get_conversation_tokens()
            original_count = len(self.conversation_history)

            # Allow compaction even for short conversations - let the user decide

            # Use compaction engine
            compacted_history, summary_text = (
                self.context.compaction_engine.compact_conversation(
                    self.conversation_history
                )
            )

            if compacted_history != self.conversation_history:
                # Update conversation history with compacted version
                self.conversation_history.clear()
                self.conversation_history.extend(compacted_history)

                # Update context monitor
                self._update_context_in_monitor()

                new_tokens = self._get_conversation_tokens()
                new_count = len(self.conversation_history)

                savings_tokens = original_tokens - new_tokens

                return CommandResult(
                    success=True,
                    message=(
                        f"Compacted conversation: {original_count} → {new_count} msgs,"
                        f"~{original_tokens} → ~{new_tokens} tokens "
                        f"(saved ~{savings_tokens} tokens)"
                    ),
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=True,
                    message=(
                        "Conversation is already compact (too few messages to compress)"
                    ),
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
                message=(
                    f"Thinking display mode set to '{new_mode}' - showing {description}"
                ),
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error toggling thinking display: {str(e)}",
                should_continue=False,
            )

    def _handle_clear_thoughts(self, command: ParsedCommand) -> CommandResult:
        """Handle the /clear_thoughts() command.

        Removes thinking content from conversation history.
        """
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
                        f"Removed thinking content from {cleaned_count} "
                        "assistant message(s). Responses preserved while "
                        "reasoning sections cleaned up."
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

    def _delete_at_index(self, index: int) -> int:
        """Delete message at specific index and return number of messages deleted.

        Args:
            index: Zero-based index of message to delete

        Returns:
            Number of messages deleted (1 if successful, 0 if index invalid)

        Raises:
            IndexError: If index is out of range
        """
        if not self.conversation_history:
            return 0

        if index < 0 or index >= len(self.conversation_history):
            raise IndexError(f"Message index {index} out of range (0-{len(self.conversation_history)-1})")

        # Remove the message at the specified index
        self.conversation_history.pop(index)
        return 1

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
            # Default to showing the most recent assistant message if no position
            # specified
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
                        message=(
                            f"Position {position} is out of range "
                            f"(1-{len(assistant_messages)})"
                        ),
                        should_continue=False,
                    )

                # Show specified position
                display_position, msg_index, message = assistant_messages[position - 1]
                position_text = f"#{position}"

            # Get the corresponding user message (if any)
            user_message = None
            if (
                msg_index > 0
                and self.conversation_history[msg_index - 1].get("role") == "user"
            ):
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

            # Get model name for assistant title
            model_name = getattr(self.config.model, "model_name", "Assistant")

            # Clean up model name for display
            if "/" in model_name:
                display_model_name = model_name.split("/")[-1]  # Get last part after /
            else:
                display_model_name = model_name

            # Show assistant response with markdown rendering
            from rich.markdown import Markdown

            try:
                # Try to render as markdown
                markdown_content = Markdown(display_content)
                assistant_panel = Panel(
                    markdown_content,
                    title=(
                        f"[bold cyan]{display_model_name} ({position_text})[/bold cyan]"
                    ),
                    border_style="cyan",
                    padding=(1, 2),
                    expand=getattr(self.config.style, "expand_panels", False),
                )
            except Exception:
                # Fallback to plain text if markdown fails
                assistant_panel = Panel(
                    Text(display_content, style="white"),
                    title=(
                        f"[bold cyan]{display_model_name} ({position_text})[/bold cyan]"
                    ),
                    border_style="cyan",
                    padding=(1, 2),
                    expand=getattr(self.config.style, "expand_panels", False),
                )

            self.console.print(assistant_panel)

            return CommandResult(
                success=True,
                message=f"Message {position}: {display_content}",
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error showing conversation position: {str(e)}",
                should_continue=False,
            )

    def _handle_render(self, command: ParsedCommand) -> CommandResult:
        """Handle the /render(path) command.

        Records conversation playback with asciinema.
        """
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message=(
                        "render command requires a file path argument "
                        "(e.g., /render(conversation.cast))"
                    ),
                    should_continue=False,
                )

            output_path = command.args[0].strip()

            # Ensure .cast extension
            from pathlib import Path

            path_obj = Path(output_path)
            if path_obj.suffix.lower() != ".cast":
                output_path = str(path_obj.with_suffix(".cast"))

            # Check if asciinema is available
            import shutil

            if not shutil.which("asciinema"):
                return CommandResult(
                    success=False,
                    message=(
                        "asciinema is not installed. "
                        "Install with: pip install asciinema"
                    ),
                    should_continue=False,
                )

            # Create the conversation renderer
            renderer = ConversationRenderer(
                conversation_history=self.conversation_history,
                console=self.console,
                config=self.config,
                thinking_processor=self.context.thinking_processor,
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

    def _handle_edit(self, command: ParsedCommand) -> CommandResult:
        """Handle the /edit(index, new_content) command to edit a message.
        
        Args:
            command: Parsed command with index and new content arguments
                    - First arg: Message index (required)
                    - Second arg: New content (required)
        """
        logger.info(f"✏️  EDIT: Starting edit operation with args: {command.args}")
        logger.info(f"✏️  EDIT: Current conversation length: {len(self.conversation_history) if self.conversation_history else 0}")
        
        try:
            if not self.conversation_history:
                return CommandResult(
                    success=False,
                    message="No conversation history to edit",
                    should_continue=False,
                )

            # Validate arguments
            if not command.args or len(command.args) < 2:
                return CommandResult(
                    success=False,
                    message="Edit command requires index and new content: /edit(index, 'new content')",
                    should_continue=False,
                )

            try:
                index = int(command.args[0])
                new_content = command.args[1]
                logger.info(f"✏️  EDIT: Editing message at index {index}")
                logger.info(f"✏️  EDIT: New content preview: {new_content[:100]}...")
            except (ValueError, IndexError):
                return CommandResult(
                    success=False,
                    message=f"Invalid arguments: index must be a number, content must be provided",
                    should_continue=False,
                )

            # Validate index
            if index < 0 or index >= len(self.conversation_history):
                logger.error(f"✏️  EDIT: Index {index} out of bounds (conversation has {len(self.conversation_history)} messages)")
                return CommandResult(
                    success=False,
                    message=f"Message index {index} out of range (0-{len(self.conversation_history)-1})",
                    should_continue=False,
                )

            # Update the message content
            old_content = self.conversation_history[index].get("content", "")
            old_role = self.conversation_history[index].get("role", "unknown")
            logger.info(f"✏️  EDIT: Updating {old_role} message at index {index}")
            logger.info(f"✏️  EDIT: Old content preview: {old_content[:100]}...")
            
            self.conversation_history[index]["content"] = new_content
            logger.info(f"✏️  EDIT: Successfully updated message content at index {index}")

            # Update context monitor
            self._update_context_in_monitor()
            logger.info(f"✏️  EDIT: Context monitor updated")

            return CommandResult(
                success=True,
                message=f"Updated {old_role} message at index {index}",
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error editing message: {str(e)}",
                should_continue=False,
            )
