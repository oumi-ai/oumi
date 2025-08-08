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

"""Conversation branching system for managing multiple conversation paths."""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class ConversationBranch:
    """Represents a single conversation branch."""

    id: str
    name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    parent_branch_id: Optional[str] = None
    branch_point_index: int = 0  # Index in parent where branch was created
    conversation_history: list[dict] = field(default_factory=list)
    
    # Model configuration state for this branch
    model_name: Optional[str] = None
    engine_type: Optional[str] = None
    model_config: Optional[dict] = None  # Serialized model config
    generation_config: Optional[dict] = None  # Serialized generation config

    def get_preview(self, max_length: int = 50) -> str:
        """Get a preview of the last messages in this branch.

        Args:
            max_length: Maximum length of preview text.

        Returns:
            Preview string showing last user/assistant exchange.
        """
        if not self.conversation_history:
            return "(empty branch)"

        # Find last user message
        last_user = None
        last_assistant = None

        for msg in reversed(self.conversation_history):
            if msg.get("role") == "user" and not last_user:
                last_user = msg.get("content", "")
            elif msg.get("role") == "assistant" and not last_assistant:
                last_assistant = msg.get("content", "")

            if last_user and last_assistant:
                break

        preview_parts = []
        if last_user:
            user_preview = last_user[:max_length]
            if len(last_user) > max_length:
                user_preview += "..."
            preview_parts.append(f"User: {user_preview}")

        if last_assistant:
            assistant_preview = last_assistant[:max_length]
            if len(last_assistant) > max_length:
                assistant_preview += "..."
            preview_parts.append(f"Assistant: {assistant_preview}")

        return " | ".join(preview_parts) if preview_parts else "(no messages)"


class ConversationBranchManager:
    """Manages multiple conversation branches."""

    MAX_BRANCHES = 5

    def __init__(self):
        """Initialize the branch manager."""
        # Create main branch
        self.branches: dict[str, ConversationBranch] = {}
        self.current_branch_id = "main"
        self._branch_counter = 0

        # Initialize with main branch
        main_branch = ConversationBranch(
            id="main", name="Main", conversation_history=[]
        )
        self.branches["main"] = main_branch

    def create_branch(
        self,
        from_branch_id: str,
        branch_point: Optional[int] = None,
        name: Optional[str] = None,
    ) -> tuple[bool, str, Optional[ConversationBranch]]:
        """Create a new branch from an existing one.

        Args:
            from_branch_id: ID of the branch to fork from.
            branch_point: Index to branch from (None = current end).
            name: Optional name for the branch.

        Returns:
            Tuple of (success, message, new_branch).
        """
        # Check branch limit
        if len(self.branches) >= self.MAX_BRANCHES:
            return (
                False,
                f"Maximum number of branches ({self.MAX_BRANCHES}) reached",
                None,
            )

        # Check source branch exists
        if from_branch_id not in self.branches:
            return False, f"Branch '{from_branch_id}' not found", None

        source_branch = self.branches[from_branch_id]

        # Determine branch point
        if branch_point is None:
            branch_point = len(source_branch.conversation_history)
        elif branch_point < 0 or branch_point > len(source_branch.conversation_history):
            return False, "Invalid branch point", None

        # Generate new branch ID
        self._branch_counter += 1
        new_id = f"branch_{self._branch_counter}"

        # Create new branch with copied history and model state up to branch point
        new_branch = ConversationBranch(
            id=new_id,
            name=name or f"Branch {self._branch_counter}",
            parent_branch_id=from_branch_id,
            branch_point_index=branch_point,
            conversation_history=copy.deepcopy(
                source_branch.conversation_history[:branch_point]
            ),
            # Copy model state from source branch
            model_name=source_branch.model_name,
            engine_type=source_branch.engine_type,
            model_config=copy.deepcopy(source_branch.model_config) if source_branch.model_config else None,
            generation_config=copy.deepcopy(source_branch.generation_config) if source_branch.generation_config else None,
        )

        self.branches[new_id] = new_branch
        return (
            True,
            f"Created branch '{new_branch.name}' from '{source_branch.name}'",
            new_branch,
        )

    def switch_branch(
        self, branch_id: str
    ) -> tuple[bool, str, Optional[ConversationBranch]]:
        """Switch to a different branch.

        Args:
            branch_id: ID of the branch to switch to.

        Returns:
            Tuple of (success, message, branch).
        """
        if branch_id not in self.branches:
            return False, f"Branch '{branch_id}' not found", None

        self.current_branch_id = branch_id
        branch = self.branches[branch_id]
        branch.last_active = datetime.now()

        return True, f"Switched to branch '{branch.name}'", branch

    def delete_branch(self, branch_id: str) -> tuple[bool, str]:
        """Delete a branch.

        Args:
            branch_id: ID of the branch to delete.

        Returns:
            Tuple of (success, message).
        """
        if branch_id == "main":
            return False, "Cannot delete the main branch"

        if branch_id not in self.branches:
            return False, f"Branch '{branch_id}' not found"

        if branch_id == self.current_branch_id:
            # Switch to main if deleting current branch
            self.current_branch_id = "main"

        branch_name = self.branches[branch_id].name
        del self.branches[branch_id]

        return True, f"Deleted branch '{branch_name}'"

    def get_current_branch(self) -> ConversationBranch:
        """Get the current active branch.

        Returns:
            The current ConversationBranch.
        """
        return self.branches[self.current_branch_id]

    def list_branches(self) -> list[dict[str, any]]:
        """List all branches with their information.

        Returns:
            List of branch information dictionaries.
        """
        branches_info = []

        for branch_id, branch in self.branches.items():
            info = {
                "id": branch_id,
                "name": branch.name,
                "is_current": branch_id == self.current_branch_id,
                "created_at": branch.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "last_active": branch.last_active.strftime("%Y-%m-%d %H:%M:%S"),
                "message_count": len(branch.conversation_history),
                "preview": branch.get_preview(),
                "parent": branch.parent_branch_id,
            }
            branches_info.append(info)

        # Sort by last active, with current branch first
        branches_info.sort(
            key=lambda x: (not x["is_current"], x["last_active"]), reverse=True
        )

        return branches_info

    def get_branch_by_name(self, name: str) -> Optional[ConversationBranch]:
        """Find a branch by name (case-insensitive).

        Args:
            name: Branch name to search for.

        Returns:
            The branch if found, None otherwise.
        """
        name_lower = name.lower()
        for branch in self.branches.values():
            if branch.name and branch.name.lower() == name_lower:
                return branch
        return None
