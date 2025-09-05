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

"""Branch management service for WebChat server."""

import copy
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from oumi.core.commands.conversation_branches import ConversationBranchManager, ConversationBranch
from oumi.utils.logging import logger
from oumi.webchat.services.persistence_service import PersistenceService


class BranchService:
    """Manages conversation branches for WebChat."""
    
    def __init__(self, persistence_service: Optional[PersistenceService] = None):
        """Initialize branch service.
        
        Args:
            persistence_service: Optional persistence service for storing branches.
        """
        self.persistence = persistence_service
    
    def create_branch(
        self,
        branch_manager: ConversationBranchManager,
        from_branch_id: Optional[str] = None,
        name: Optional[str] = None,
        switch_to: bool = True,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Tuple[bool, str, Optional[ConversationBranch]]:
        """Create a new branch.
        
        Args:
            branch_manager: Branch manager to create the branch in.
            from_branch_id: Source branch ID to fork from, or None for current.
            name: Optional name for the new branch.
            switch_to: Whether to switch to the new branch after creation.
            session_id: Optional session ID for persistence.
            conversation_id: Optional conversation ID for persistence.
            
        Returns:
            Tuple of (success, message, branch) where branch is the new branch if created.
        """
        # If from_branch_id is None, use current branch
        if from_branch_id is None:
            from_branch_id = branch_manager.current_branch_id
        
        # Log branch creation attempt
        logger.debug(f"🌿 Creating branch from '{from_branch_id}' with name '{name}'")
        logger.debug(f"🌿 Branch manager has branches: {list(branch_manager.branches.keys())}")
        
        # CRITICAL FIX: Sync source branch conversation if it's the current branch
        if from_branch_id == branch_manager.current_branch_id:
            branch_manager.sync_conversation_history(branch_manager.conversation_history)
            logger.debug(f"🔄 Synced current branch '{from_branch_id}' before creating new branch")
        
        # Create the branch
        success, message, new_branch = branch_manager.create_branch(
            from_branch_id=from_branch_id, 
            name=name, 
            switch_to=switch_to
        )
        
        if success and new_branch:
            logger.debug(f"✅ Created branch '{new_branch.id}' successfully")
            
            # Save to persistence if available
            if self.persistence and session_id and conversation_id:
                try:
                    # Ensure branch exists in DB
                    self.persistence.ensure_branch(
                        conversation_id, 
                        new_branch.id, 
                        name=new_branch.name, 
                        parent_branch_id=from_branch_id
                    )
                    
                    # Save branch history
                    self.persistence.save_branch_history(
                        session_id,
                        new_branch.id,
                        new_branch.conversation_history
                    )
                    
                    # Update current branch pointer if switched
                    if switch_to:
                        self.persistence.set_session_current_branch(
                            session_id,
                            conversation_id,
                            branch_manager.current_branch_id
                        )
                        
                    logger.debug(f"🗄️ Saved branch '{new_branch.id}' to persistence")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save branch to persistence: {e}")
        else:
            logger.warning(f"⚠️ Failed to create branch: {message}")
        
        return success, message, new_branch
    
    def switch_branch(
        self,
        branch_manager: ConversationBranchManager,
        branch_id: str,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str, Optional[ConversationBranch]]:
        """Switch to a different branch.
        
        Args:
            branch_manager: Branch manager to switch branch in.
            branch_id: Branch ID to switch to.
            session_id: Optional session ID for persistence.
            conversation_id: Optional conversation ID for persistence.
            conversation_history: Optional conversation history to sync to current branch.
            
        Returns:
            Tuple of (success, message, branch) where branch is the branch switched to if successful.
        """
        # Log branch switch attempt
        logger.debug(f"🔀 Switching branch from '{branch_manager.current_branch_id}' to '{branch_id}'")
        
        # Sync current conversation to the current branch before switching
        if conversation_history is not None:
            current_branch_id = branch_manager.current_branch_id
            if current_branch_id in branch_manager.branches:
                current_branch = branch_manager.branches[current_branch_id]
                current_branch.conversation_history = copy.deepcopy(conversation_history)
                current_branch.last_active = datetime.now()
                logger.debug(f"🔄 Saved {len(conversation_history)} messages to current branch '{current_branch_id}'")
        
        # Switch branch
        success, message, branch = branch_manager.switch_branch(branch_id)
        
        if success and branch:
            logger.debug(f"✅ Switched to branch '{branch_id}' successfully")
            
            # Save to persistence if available
            if self.persistence and session_id and conversation_id:
                try:
                    self.persistence.set_session_current_branch(
                        session_id,
                        conversation_id,
                        branch_manager.current_branch_id
                    )
                    logger.debug(f"🗄️ Updated current branch to '{branch_id}' in persistence")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update current branch in persistence: {e}")
        else:
            logger.warning(f"⚠️ Failed to switch branch: {message}")
        
        return success, message, branch
    
    def delete_branch(
        self,
        branch_manager: ConversationBranchManager,
        branch_id: str,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Delete a branch with safety checks and persistent storage.
        
        Args:
            branch_manager: Branch manager to delete the branch from.
            branch_id: Branch ID to delete.
            session_id: Optional session ID for persistence and checking if branch is current.
            conversation_id: Optional conversation ID for persistence.
            
        Returns:
            Tuple of (success, message, result_details).
        """
        # Log branch deletion attempt
        logger.debug(f"🗑️ Deleting branch '{branch_id}'")
        
        # Safety check: Cannot delete main branch
        if branch_id == "main":
            logger.warning("⚠️ Cannot delete the main branch")
            return False, "Cannot delete the main branch", None
        
        # Safety check: Cannot delete current branch
        if branch_id == branch_manager.current_branch_id:
            logger.warning(f"⚠️ Cannot delete the current branch '{branch_id}' (switch first)")
            return False, "Cannot delete the current branch (switch to a different branch first)", None
        
        # Check if branch exists in memory
        if branch_id not in branch_manager.branches:
            logger.warning(f"⚠️ Branch '{branch_id}' not found")
            return False, f"Branch '{branch_id}' not found", None
        
        # Perform additional checks in persistence if available
        db_result = None
        if self.persistence and session_id and conversation_id:
            try:
                # Check if branch has children in DB
                if self.persistence.branch_has_children(conversation_id, branch_id):
                    logger.warning(f"⚠️ Branch '{branch_id}' has children; delete descendants first")
                    return False, "Branch has children; delete descendants first", None
                
                # Check if branch is current in persistence (double-check)
                if self.persistence.branch_is_current(session_id, branch_id):
                    logger.warning(f"⚠️ Branch '{branch_id}' is current in DB; switch first")
                    return False, "Branch is current in DB; switch first", None
                
                # Delete in persistence first (maintains atomicity - if DB fails, memory state is unchanged)
                db_result = self.persistence.delete_branch(conversation_id, branch_id, session_id)
                
                # If DB deletion failed, abort
                if not db_result["success"]:
                    logger.warning(f"⚠️ DB deletion failed: {db_result['reason']}")
                    return False, f"Database error: {db_result['reason']}", db_result
                    
                logger.debug(f"✅ Deleted branch '{branch_id}' from persistence successfully")
            
            except Exception as e:
                logger.error(f"❌ Error during persistence checks/deletion: {e}")
                return False, f"Persistence error: {str(e)}", None
        
        # Now delete the branch in memory
        try:
            success, message = branch_manager.delete_branch(branch_id)
            
            if success:
                logger.debug(f"✅ Deleted branch '{branch_id}' from memory successfully")
                # Return details from both memory and DB operations
                return True, f"Branch '{branch_id}' deleted successfully", db_result
            else:
                logger.warning(f"⚠️ Failed to delete branch from memory: {message}")
                return False, message, db_result
        except Exception as e:
            logger.error(f"❌ Error during in-memory branch deletion: {e}")
            return False, f"Memory deletion error: {str(e)}", db_result
    
    def get_branch_info(
        self,
        branch_manager: ConversationBranchManager,
        persistence_service: Optional[PersistenceService] = None,
        session_id: Optional[str] = None,
        is_hydrated_from_db: bool = False,
        current_conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get information about all branches with optional enhanced info from persistence.
        
        Args:
            branch_manager: Branch manager to get branch info from.
            persistence_service: Optional persistence service for enhanced info.
            session_id: Optional session ID for persistence queries.
            is_hydrated_from_db: Whether the session is hydrated from DB.
            current_conversation_id: Optional current conversation ID.
            
        Returns:
            List of branch information dictionaries.
        """
        # Get basic branch info from branch manager
        branches_info = branch_manager.list_branches()
        
        # Enhance with DB data if available
        if persistence_service and is_hydrated_from_db and current_conversation_id and session_id:
            try:
                # Get DB branch information
                db_branches = persistence_service.get_session_branches(session_id)
                db_branch_lookup = {b["id"]: b for b in db_branches}
                
                # Update each branch with DB-backed information
                for branch_info in branches_info:
                    branch_id = branch_info["id"]
                    if branch_id in db_branch_lookup:
                        db_branch = db_branch_lookup[branch_id]
                        # Use DB count as authoritative
                        branch_info["message_count"] = db_branch["message_count"]
                        
                        # Enhanced preview from DB if needed
                        if db_branch["message_count"] > 0 and not branch_info["preview"]:
                            try:
                                recent_messages = persistence_service.get_branch_messages(branch_id)
                                if recent_messages:
                                    last_msg = recent_messages[-1]
                                    content = last_msg["content"][:50]
                                    branch_info["preview"] = f"[{last_msg['role']}] {content}..." if content else "(empty message)"
                            except Exception as preview_error:
                                logger.debug(f"⚠️ Failed to generate DB preview for branch {branch_id}: {preview_error}")
                
                logger.debug(f"🗄️ Enhanced {len(branches_info)} branches with DB-backed counts")
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to enhance branch info with DB data: {e}")
        
        return branches_info