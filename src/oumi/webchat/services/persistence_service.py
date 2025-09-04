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

"""Persistence service for WebChat sessions and conversations."""

import time
from typing import Dict, List, Optional, Any, Tuple

from oumi.utils.logging import logger
from oumi.webchat.persistence import WebchatDB


class PersistenceService:
    """Manages persistence for WebChat sessions, conversations, and branches."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize persistence service.
        
        Args:
            db_path: Optional path to SQLite database file.
        """
        try:
            self.db = WebchatDB(db_path)
            self._enabled = True
            logger.info("🗄️  PersistenceService initialized with WebchatDB")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize WebchatDB: {e}")
            self.db = None
            self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if persistence is enabled.
        
        Returns:
            True if persistence is enabled, False otherwise.
        """
        return self._enabled and self.db is not None
    
    def ensure_session(self, session_id: str) -> bool:
        """Ensure a session exists in the database.
        
        Args:
            session_id: Session ID to ensure exists.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.is_enabled:
            return False
        
        try:
            self.db.ensure_session(session_id)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to ensure session {session_id}: {e}")
            return False
    
    def ensure_conversation(self, session_id: str) -> Optional[str]:
        """Ensure a conversation exists for the session and is linked to it.
        
        Args:
            session_id: Session ID to ensure a conversation for.
            
        Returns:
            Conversation ID if successful, None otherwise.
        """
        if not self.is_enabled:
            return None
        
        try:
            return self.db.ensure_conversation(session_id)
        except Exception as e:
            logger.warning(f"⚠️ Failed to ensure conversation for session {session_id}: {e}")
            return None
    
    def ensure_branch(
        self, 
        conversation_id: str, 
        branch_id: str, 
        name: Optional[str] = None, 
        parent_branch_id: Optional[str] = None
    ) -> bool:
        """Ensure a branch exists in the database.
        
        Args:
            conversation_id: Conversation ID that the branch belongs to.
            branch_id: Branch ID to ensure exists.
            name: Optional name for the branch.
            parent_branch_id: Optional parent branch ID.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.is_enabled or not conversation_id:
            return False
        
        try:
            self.db.ensure_branch(conversation_id, branch_id, name, parent_branch_id)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to ensure branch {branch_id}: {e}")
            return False
    
    def set_session_current_branch(
        self, 
        session_id: str, 
        conversation_id: str, 
        branch_id: str
    ) -> bool:
        """Set the current branch for a session.
        
        Args:
            session_id: Session ID to update.
            conversation_id: Conversation ID for the session.
            branch_id: Branch ID to set as current.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.is_enabled:
            return False
        
        try:
            self.db.set_session_current_branch(session_id, conversation_id, branch_id)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to set current branch for session {session_id}: {e}")
            return False
    
    def append_message(
        self, 
        session_id: str, 
        branch_id: str, 
        role: str, 
        content: str, 
        timestamp: Optional[float] = None
    ) -> bool:
        """Append a message to a branch.
        
        Args:
            session_id: Session ID that owns the branch.
            branch_id: Branch ID to append the message to.
            role: Role of the message sender (user, assistant, system).
            content: Content of the message.
            timestamp: Optional timestamp for the message.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.is_enabled:
            return False
        
        timestamp = timestamp or time.time()
        
        try:
            # Ensure session and conversation exist
            conv_id = self.db.ensure_conversation(session_id)
            if not conv_id:
                return False
            
            # Ensure branch exists
            self.db.ensure_branch(conv_id, branch_id, name=branch_id)
            
            # Append message to branch
            self.db.append_message_to_branch(
                conv_id,
                branch_id,
                role=role,
                content=str(content),
                created_at=float(timestamp)
            )
            
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to append message to branch {branch_id}: {e}")
            return False
    
    def append_conversation_pair(
        self, 
        session_id: str, 
        branch_id: str, 
        user_message: str, 
        assistant_message: str,
        user_timestamp: Optional[float] = None,
        assistant_timestamp: Optional[float] = None
    ) -> bool:
        """Append a user-assistant message pair to a branch.
        
        Args:
            session_id: Session ID that owns the branch.
            branch_id: Branch ID to append the messages to.
            user_message: Content of the user message.
            assistant_message: Content of the assistant message.
            user_timestamp: Optional timestamp for the user message.
            assistant_timestamp: Optional timestamp for the assistant message.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.is_enabled:
            return False
        
        now = time.time()
        user_timestamp = user_timestamp or now - 1  # Slightly earlier if not provided
        assistant_timestamp = assistant_timestamp or now
        
        try:
            # Ensure session and conversation exist
            conv_id = self.db.ensure_conversation(session_id)
            if not conv_id:
                return False
            
            # Ensure branch exists
            self.db.ensure_branch(conv_id, branch_id, name=branch_id)
            
            # Append user message
            self.db.append_message_to_branch(
                conv_id,
                branch_id,
                role="user",
                content=str(user_message),
                created_at=float(user_timestamp)
            )
            
            # Append assistant message
            self.db.append_message_to_branch(
                conv_id,
                branch_id,
                role="assistant",
                content=str(assistant_message),
                created_at=float(assistant_timestamp)
            )
            
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to append conversation pair to branch {branch_id}: {e}")
            return False
    
    def save_branch_history(
        self, 
        session_id: str, 
        branch_id: str, 
        messages: List[Dict[str, Any]]
    ) -> bool:
        """Save a branch's entire conversation history.
        
        Args:
            session_id: Session ID that owns the branch.
            branch_id: Branch ID to save the history for.
            messages: List of message dictionaries with role, content, and timestamp.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.is_enabled:
            return False
        
        try:
            # Ensure session and conversation exist
            conv_id = self.db.ensure_conversation(session_id)
            if not conv_id:
                return False
            
            # Ensure branch exists
            self.db.ensure_branch(conv_id, branch_id, name=branch_id)
            
            # Bulk add messages to branch
            self.db.bulk_add_branch_history(conv_id, branch_id, messages)
            
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to save branch history for branch {branch_id}: {e}")
            return False
    
    def get_branch_messages(self, branch_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a branch in sequence order.
        
        Args:
            branch_id: Branch ID to get messages for.
            
        Returns:
            List of message dictionaries.
        """
        if not self.is_enabled:
            return []
        
        try:
            return self.db.get_branch_messages(branch_id)
        except Exception as e:
            logger.warning(f"⚠️ Failed to get messages for branch {branch_id}: {e}")
            return []
    
    def get_session_branches(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all branches for a session's current conversation.
        
        Args:
            session_id: Session ID to get branches for.
            
        Returns:
            List of branch dictionaries.
        """
        if not self.is_enabled:
            return []
        
        try:
            return self.db.get_session_branches(session_id)
        except Exception as e:
            logger.warning(f"⚠️ Failed to get branches for session {session_id}: {e}")
            return []
    
    def hydrate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load complete session state from database for hydrating in-memory structures.
        
        Args:
            session_id: Session ID to hydrate.
            
        Returns:
            Dictionary with session info, branches, and current branch messages,
            or None if session not found or error occurs.
        """
        if not self.is_enabled:
            return None
        
        try:
            return self.db.hydrate_session(session_id)
        except Exception as e:
            logger.warning(f"⚠️ Failed to hydrate session {session_id}: {e}")
            return None
    
    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update the title of a conversation.
        
        Args:
            conversation_id: Conversation ID to update.
            title: New title for the conversation.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.is_enabled:
            return False
        
        try:
            self.db.update_conversation_title(conversation_id, title)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to update conversation title: {e}")
            return False