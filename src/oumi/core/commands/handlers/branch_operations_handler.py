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

"""Branch operations command handler."""

import copy
from datetime import datetime

from rich.table import Table

from oumi.core.commands.base_handler import BaseCommandHandler, CommandResult
from oumi.core.commands.command_parser import ParsedCommand


class BranchOperationsHandler(BaseCommandHandler):
    """Handles branch-related commands: branch, switch, branches, branch_delete."""

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return ["branch", "branch_from", "switch", "branches", "branch_delete"]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a branch operations command."""
        if command.command == "branch":
            return self._handle_branch(command)
        elif command.command == "branch_from":
            return self._handle_branch_from(command)
        elif command.command == "switch":
            return self._handle_switch(command)
        elif command.command == "branches":
            return self._handle_branches(command)
        elif command.command == "branch_delete":
            return self._handle_branch_delete(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unsupported command: {command.command}",
                should_continue=False,
            )

    def _handle_branch(self, command: ParsedCommand) -> CommandResult:
        """Handle the /branch() command to create a new conversation branch."""
        try:
            branch_manager = self.context.branch_manager

            # Save current conversation history before creating new branch
            # Use defensive copying to prevent shared reference issues
            current_history_snapshot = copy.deepcopy(self.conversation_history)
            branch_manager.sync_conversation_history(current_history_snapshot)

            # Get branch name from arguments
            branch_name = None
            if command.args:
                branch_name = command.args[0].strip()

            # Create the branch from current branch
            current_branch_id = branch_manager.current_branch_id
            success, message, new_branch = branch_manager.create_branch(
                from_branch_id=current_branch_id, name=branch_name
            )

            if success and new_branch:
                # Automatically switch to the newly created branch
                # This ensures the user is working on the new branch immediately
                branch_manager.current_branch_id = new_branch.id

                # Update conversation history to match the new branch
                # (which should be a copy of the current conversation)
                self.conversation_history.clear()
                self.conversation_history.extend(copy.deepcopy(new_branch.conversation_history))

                # Update context monitor
                self._update_context_in_monitor()

                return CommandResult(
                    success=True,
                    message=message,
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message=message,
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error creating branch: {str(e)}",
                should_continue=False,
            )

    def _handle_switch(self, command: ParsedCommand) -> CommandResult:
        """Handle the /switch(branch_name) command to switch conversation branches."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="switch command requires a branch name argument",
                    should_continue=False,
                )

            branch_name = command.args[0].strip()
            branch_manager = self.context.branch_manager

            # Save current conversation history to the current branch before switching
            # Use copy to ensure we don't get affected by subsequent list modifications
            current_history_snapshot = copy.deepcopy(self.conversation_history)
            branch_manager.sync_conversation_history(current_history_snapshot)

            # Switch to the branch
            success, message, branch = branch_manager.switch_branch(branch_name)

            if success and branch:
                # Update conversation history with branch content
                # Clear and replace with deep copy to ensure isolation
                self.conversation_history.clear()
                self.conversation_history.extend(copy.deepcopy(branch.conversation_history))

                # Update context monitor
                self._update_context_in_monitor()

                return CommandResult(
                    success=True,
                    message=message,
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message=message,
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error switching branch: {str(e)}",
                should_continue=False,
            )

    def _handle_branches(self, command: ParsedCommand) -> CommandResult:
        """Handle the /branches() command to list all conversation branches."""
        try:
            branch_manager = self.context.branch_manager
            branches = branch_manager.list_branches()

            if not branches:
                return CommandResult(
                    success=True,
                    message="No conversation branches found",
                    should_continue=False,
                )

            # Create a table to display branches
            table = Table(
                title="🌿 Conversation Branches"
                if getattr(self._style, "use_emoji", True)
                else "Conversation Branches"
            )
            table.add_column("Branch", style="cyan", no_wrap=True)
            table.add_column("Messages", style="magenta")
            table.add_column("Created", style="green")
            table.add_column("Preview", style="white")

            current_branch = branch_manager.get_current_branch()
            current_branch_id = current_branch.id if current_branch else None

            for branch in branches:
                branch_name = branch.get("name", branch.get("id", "unknown"))
                message_count = branch.get(
                    "message_count", 0
                )  # Use pre-calculated count
                created_time = branch.get("created_at", "unknown")

                # Get preview from the branch info
                preview = branch.get("preview", "Empty")

                # Format created time
                if created_time != "unknown":
                    try:
                        if isinstance(created_time, str):
                            dt = datetime.fromisoformat(
                                created_time.replace("Z", "+00:00")
                            )
                            created_display = dt.strftime("%m/%d %H:%M")
                        else:
                            created_display = str(created_time)
                    except Exception:
                        created_display = str(created_time)
                else:
                    created_display = "unknown"

                # Mark current branch
                if branch.get("id") == current_branch_id:
                    branch_name = f"→ {branch_name}"

                table.add_row(branch_name, str(message_count), created_display, preview)

            self.console.print(table)

            # Create a message containing all branch names and IDs for test verification
            branch_names = []
            for branch in branches:
                name = branch.get("name", branch.get("id", "unknown"))
                id = branch.get("id", "unknown")
                # Include both name and ID to support both test patterns
                if name != id:
                    branch_names.append(f"{name} ({id})")
                else:
                    branch_names.append(name)
            message = f"Found {len(branches)} branch(es): {', '.join(branch_names)}"

            return CommandResult(success=True, message=message, should_continue=False)

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error listing branches: {str(e)}",
                should_continue=False,
            )

    def _handle_branch_delete(self, command: ParsedCommand) -> CommandResult:
        """Handle the /branch_delete(branch_name) command to delete a branch."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="branch_delete command requires a branch name argument",
                    should_continue=False,
                )

            branch_name = command.args[0].strip()
            branch_manager = self.context.branch_manager

            # Delete the branch
            success, message = branch_manager.delete_branch(branch_name)

            return CommandResult(
                success=success,
                message=message,
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error deleting branch: {str(e)}",
                should_continue=False,
            )

    def _handle_branch_from(self, command: ParsedCommand) -> CommandResult:
        """Handle the /branch_from(name,pos) command to branch from position."""
        try:
            if len(command.args) < 2:
                return CommandResult(
                    success=False,
                    message="branch_from command requires two arguments: branch_name and position",
                    should_continue=False,
                )

            # Save current conversation history to the current branch before branching
            branch_manager = self.context.branch_manager
            branch_manager.sync_conversation_history(self.conversation_history)

            branch_name = command.args[0].strip()
            try:
                position = int(command.args[1].strip())
            except ValueError:
                return CommandResult(
                    success=False,
                    message="Position must be a valid integer",
                    should_continue=False,
                )

            # Validate position is positive and refers to an assistant message
            if position < 1:
                return CommandResult(
                    success=False,
                    message="Position must be 1 or greater (1-indexed)",
                    should_continue=False,
                )

            # Count assistant messages to validate position
            assistant_messages = []
            for i, msg in enumerate(self.conversation_history):
                if msg.get("role") == "assistant":
                    assistant_messages.append((i, msg))

            if position > len(assistant_messages):
                return CommandResult(
                    success=False,
                    message=f"Position {position} exceeds number of assistant messages ({len(assistant_messages)})",
                    should_continue=False,
                )

            # Get the actual index in conversation history for the specified assistant message
            branch_point_index = (
                assistant_messages[position - 1][0] + 1
            )  # +1 to branch after the assistant message

            # Create conversation history up to the branch point
            branched_history = self.conversation_history[:branch_point_index]

            # Create the branch from specific position
            success, message = branch_manager.create_branch_from_position(
                branch_name, branched_history, branch_point_index
            )

            if success:
                # Switch to the new branch
                switch_success, switch_message, branch = branch_manager.switch_branch(
                    branch_name.lower()
                )
                # Update conversation history to the new branch
                self.conversation_history.clear()
                self.conversation_history.extend(branched_history)
                self._update_context_in_monitor()

                return CommandResult(
                    success=True,
                    message=f"Created and switched to branch '{branch_name}' from assistant message {position}",
                    should_continue=False,
                )
            else:
                return CommandResult(
                    success=False,
                    message=message,
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error creating branch from position: {str(e)}",
                should_continue=False,
            )
