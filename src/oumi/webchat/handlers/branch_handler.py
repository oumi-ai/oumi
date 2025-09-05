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

"""Branch operations handler for Oumi WebChat server."""

import copy
import time
from datetime import datetime
from typing import Dict, Optional, Any

from aiohttp import web

from oumi.utils.logging import logger
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.protocol import normalize_branch_action, extract_session_id, get_valid_branch_actions


class BranchHandler:
    """Handles branch operations for Oumi WebChat."""
    
    def __init__(
        self, 
        session_manager: SessionManager,
        db = None
    ):
        """Initialize branch handler.
        
        Args:
            session_manager: Session manager for WebChat sessions
            db: Optional WebchatDB instance for persistence
        """
        self.session_manager = session_manager
        self.db = db
    
    async def handle_branches_api(self, request: web.Request) -> web.Response:
        """Handle branch operations via REST API.
        
        Args:
            request: Web request with branch operation parameters
            
        Returns:
            JSON response with branch operation result
        """
        if request.method == "GET":
            # For GET, extract session_id from query params only
            try:
                session_id = extract_session_id(request)
                logger.debug(f"🌐 DEBUG: Branch API (GET) called with session_id: '{session_id}'")
                session = await self.session_manager.get_or_create_session_safe(session_id, self.db)
            except ValueError as e:
                logger.warning(f"⚠️ Branch API error: {e}")
                return web.json_response({"error": str(e)}, status=400)
            # DEBUG: Check raw branch storage
            logger.debug(f"📋 DEBUG: GET branches request - session_id: '{session_id}'")
            logger.debug(f"📋 DEBUG: Session object ID: {id(session)}")
            logger.debug(f"📋 DEBUG: Branch manager object ID: {id(session.branch_manager)}")
            logger.debug(f"📋 DEBUG: Raw branches dict: {list(session.branch_manager.branches.keys())}")
            logger.debug(f"📋 DEBUG: Branch counter: {session.branch_manager._branch_counter}")
            
            branches = session.get_enhanced_branch_info(self.db)
            current_branch = session.branch_manager.current_branch_id
            logger.debug(f"📋 DEBUG: Get branches HTTP request - current: '{current_branch}', available: {[b['id'] for b in branches]}")
            logger.debug(f"📋 DEBUG: Branch details: {[(b['id'], b['message_count'], b['created_at']) for b in branches]}")
            return web.json_response(
                {
                    "branches": branches,
                    "current_branch": current_branch,
                    "persistence": {
                        "is_persistent": bool(self.db),
                        "is_hydrated_from_db": getattr(session, 'is_hydrated_from_db', False),
                        "current_conversation_id": getattr(session, 'current_conversation_id', None),
                    },
                }
            )
        
        elif request.method == "POST":
            try:
                data = await request.json()
                
                # Normalize and validate action
                action = normalize_branch_action(data.get("action", ""))
                
                # Check if session_id is also in POST data (for consistency)
                try:
                    # This will prefer the query parameter but fall back to body
                    session_id = extract_session_id(request, data)
                    session = await self.session_manager.get_or_create_session_safe(session_id, self.db)
                except ValueError as e:
                    logger.warning(f"⚠️ Branch API error: {e}")
                    return web.json_response({"error": str(e)}, status=400)
                
                if action == "switch":
                    return await self._handle_switch_branch(session_id, data)
                elif action == "create":
                    return await self._handle_create_branch(session_id, data)
                elif action == "delete":
                    return await self._handle_delete_branch(session_id, data)
                else:
                    valid_actions = get_valid_branch_actions()
                    return web.json_response(
                        {
                            "error": f"Unknown action: '{action}' (expected: {valid_actions})",
                            "valid_actions": valid_actions.split(", ")
                        },
                        status=400
                    )
            
            except Exception as e:
                logger.error(f"Branch API error: {e}")
                return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_switch_branch(self, session_id: str, data: Dict[str, Any]) -> web.Response:
        """Handle branch switching operation.
        
        Args:
            session_id: WebChat session ID
            data: Request data with branch_id to switch to
            
        Returns:
            JSON response with switch operation result
        """
        branch_id = data.get("branch_id")
        
        async def switch_branch_atomically(session):
            """Switch branches atomically within session lock."""
            logger.debug(f"🔀 DEBUG: Branch switch requested - from '{session.branch_manager.current_branch_id}' to '{branch_id}'")
            logger.debug(f"🔀 DEBUG: Current conversation length before switch: {len(session.conversation_history)}")
            
            # Log current conversation state
            for i, msg in enumerate(session.conversation_history):
                role = msg.get('role', 'unknown')
                content = str(msg.get('content', ''))[:50]
                logger.debug(f"🔀 DEBUG: Pre-switch Message {i}: [{role}] {content}...")
            
            # CRITICAL FIX: Save current conversation to current branch before switching
            current_branch_id = session.branch_manager.current_branch_id
            if current_branch_id in session.branch_manager.branches:
                current_branch = session.branch_manager.branches[current_branch_id]
                # Save current conversation history to current branch
                current_branch.conversation_history = copy.deepcopy(session.conversation_history)
                current_branch.last_active = datetime.now()
                logger.debug(f"🔀 DEBUG: Saved {len(session.conversation_history)} messages to current branch '{current_branch_id}'")
            
            success, message, branch = session.branch_manager.switch_branch(
                branch_id
            )
            logger.debug(f"🔀 DEBUG: Branch switch result - success: {success}, message: '{message}'")
            
            if success and branch:
                # PHASE 1B: Try to hydrate branch from DB if needed
                if session.is_hydrated_from_db and self.db:
                    try:
                        # Only hydrate if branch appears empty (DB is authoritative)
                        if len(branch.conversation_history) == 0:
                            logger.debug(f"🗄️ Attempting to hydrate empty branch '{branch_id}' from DB")
                            session.hydrate_branch_from_db(branch_id, self.db)
                    except Exception as hydration_error:
                        logger.warning(f"⚠️ Branch hydration failed for {branch_id}: {hydration_error}")
                
                logger.debug(f"🔀 DEBUG: Branch '{branch_id}' conversation length: {len(branch.conversation_history)}")
                # Log branch conversation before clearing current history
                for i, msg in enumerate(branch.conversation_history):
                    role = msg.get('role', 'unknown')
                    content = str(msg.get('content', ''))[:50]
                    logger.debug(f"🔀 DEBUG: Branch Message {i}: [{role}] {content}...")
                        
                # Update conversation history
                logger.debug(f"🔀 DEBUG: Clearing current conversation ({len(session.conversation_history)} messages) and loading branch conversation ({len(branch.conversation_history)} messages)")
                session.conversation_history.clear()
                session.conversation_history.extend(branch.conversation_history)
                logger.debug(f"🔀 DEBUG: Post-switch conversation length: {len(session.conversation_history)}")
            
            return success, message, session
        
        # Execute branch switch atomically
        success, message, session = await self.session_manager.execute_session_operation(
            session_id, 
            switch_branch_atomically
        )
        
        # Dual-write persistence: update session's current branch pointer
        try:
            if self.db and success:
                conv_id = self.db.ensure_conversation(session_id)
                # Mark session as persistent and record conversation id
                session.current_conversation_id = conv_id
                if not getattr(session, 'is_hydrated_from_db', False):
                    session.is_hydrated_from_db = True
                self.db.ensure_branch(
                    conv_id, 
                    session.branch_manager.current_branch_id, 
                    name=session.branch_manager.current_branch_id
                )
                self.db.set_session_current_branch(
                    session_id, 
                    conv_id, 
                    session.branch_manager.current_branch_id
                )
        except Exception as pe:
            logger.warning(f"⚠️ Dual-write persistence (branch switch) failed: {pe}")
        
        return web.json_response(
            {
                "success": success,
                "message": message,
                "conversation": session.serialize_conversation(),
                "current_branch": session.branch_manager.current_branch_id,
            }
        )
    
    async def _handle_create_branch(self, session_id: str, data: Dict[str, Any]) -> web.Response:
        """Handle branch creation operation.
        
        Args:
            session_id: WebChat session ID
            data: Request data with branch creation parameters
            
        Returns:
            JSON response with branch creation result
        """
        # Get branch parameters from request data
        session = await self.session_manager.get_or_create_session_safe(session_id, self.db)
        from_branch = data.get(
            "from_branch", session.branch_manager.current_branch_id
        )
        name = data.get("name")
        
        async def create_branch_atomically(session):
            """Create branch atomically within session lock."""
            logger.debug(f"🌿 DEBUG: Branch create requested - name: '{name}', from_branch: '{from_branch}'")
            logger.debug(f"🌿 DEBUG: Session object ID: {id(session)}")
            logger.debug(f"🌿 DEBUG: Branch manager object ID: {id(session.branch_manager)}")
            logger.debug(f"🌿 DEBUG: Current conversation length at branch point: {len(session.conversation_history)}")
            logger.debug(f"🌿 DEBUG: Branches before create: {list(session.branch_manager.branches.keys())}")
            logger.debug(f"🌿 DEBUG: Branch counter before create: {session.branch_manager._branch_counter}")
            
            # Log conversation at branch point
            for i, msg in enumerate(session.conversation_history):
                role = msg.get('role', 'unknown')
                content = str(msg.get('content', ''))[:50]
                logger.debug(f"🌿 DEBUG: Branch-point Message {i}: [{role}] {content}...")
            
            # CRITICAL DEBUG: Check source branch conversation before creating new branch
            source_branch = session.branch_manager.branches.get(from_branch)
            if source_branch:
                logger.debug(f"🌿 DEBUG: Source branch '{from_branch}' conversation length: {len(source_branch.conversation_history)}")
                for i, msg in enumerate(source_branch.conversation_history):
                    role = msg.get('role', 'unknown')
                    content = str(msg.get('content', ''))[:50]
                    logger.debug(f"🌿 DEBUG: Source branch Message {i}: [{role}] {content}...")
            else:
                logger.error(f"🚨 Source branch '{from_branch}' not found in branches!")
            
            # CRITICAL FIX: Sync source branch conversation before creating new branch
            # This ensures ANY current branch (not just main) gets synced before branching
            if from_branch == session.branch_manager.current_branch_id:
                logger.debug(f"🔄 DEBUG: Syncing current branch '{from_branch}' conversation history before branch creation")
                logger.debug(f"🔄 DEBUG: Session conversation has {len(session.conversation_history)} messages")
                session.branch_manager.sync_conversation_history(session.conversation_history)
                # Re-check source branch after sync
                source_branch_after_sync = session.branch_manager.branches.get(from_branch)
                if source_branch_after_sync:
                    logger.debug(f"🔄 DEBUG: Source branch '{from_branch}' conversation after sync: {len(source_branch_after_sync.conversation_history)} messages")
            
            success, message, new_branch = session.branch_manager.create_branch(
                from_branch_id=from_branch, name=name
            )
            
            # DEBUG: Verify new branch conversation inheritance
            if success and new_branch:
                logger.debug(f"🌿 DEBUG: New branch '{new_branch.id}' conversation length: {len(new_branch.conversation_history)}")
                for i, msg in enumerate(new_branch.conversation_history):
                    role = msg.get('role', 'unknown')
                    content = str(msg.get('content', ''))[:50]
                    logger.debug(f"🌿 DEBUG: New branch Message {i}: [{role}] {content}...")
            logger.debug(f"🌿 DEBUG: Branch create result - success: {success}, message: '{message}', new_branch_id: '{new_branch.id if new_branch else None}'")
            logger.debug(f"🌿 DEBUG: Branches after create: {list(session.branch_manager.branches.keys())}")
            logger.debug(f"🌿 DEBUG: Branch counter after create: {session.branch_manager._branch_counter}")
            
            # CRITICAL FIX: Validate branch was actually created and stored
            if success and new_branch:
                if new_branch.id not in session.branch_manager.branches:
                    logger.error(f"🚨 CRITICAL: Branch {new_branch.id} was created but not found in storage!")
                    logger.error(f"🚨 Branch manager state: {vars(session.branch_manager)}")
                else:
                    logger.debug(f"✅ Branch {new_branch.id} successfully stored and verified")
            
            # Dual-write persistence (best-effort)
            try:
                if success and new_branch and self.db:
                    self.db.ensure_session(session_id)
                    conv_id = self.db.ensure_conversation(session_id)
                    # Mark session as persistent and record conversation id
                    session.current_conversation_id = conv_id
                    if not getattr(session, 'is_hydrated_from_db', False):
                        session.is_hydrated_from_db = True
                    # Ensure both source and new branch exist in DB
                    self.db.ensure_branch(conv_id, from_branch, name=from_branch)
                    self.db.ensure_branch(conv_id, new_branch.id, name=new_branch.name, parent_branch_id=from_branch)
                    # Populate new branch history from in-memory copy
                    self.db.bulk_add_branch_history(conv_id, new_branch.id, new_branch.conversation_history)
                    # Update session's current branch pointer if switched
                    self.db.set_session_current_branch(session_id, conv_id, session.branch_manager.current_branch_id)
            except Exception as pe:
                logger.warning(f"⚠️ Dual-write persistence (branch create) failed: {pe}")
            
            return success, message, new_branch
        
        # Execute branch creation atomically
        success, message, new_branch = await self.session_manager.execute_session_operation(
            session_id, 
            create_branch_atomically
        )
        
        if success and new_branch:
            # Return the created branch in the expected format
            branch_data = {
                "id": new_branch.id,
                "name": new_branch.name,
                "message_count": len(new_branch.conversation_history),
                "created_at": new_branch.created_at.isoformat() if hasattr(new_branch.created_at, 'isoformat') else str(new_branch.created_at),
                "last_active": new_branch.last_active.isoformat() if hasattr(new_branch.last_active, 'isoformat') else str(new_branch.last_active),
                "is_active": new_branch.id == session.branch_manager.current_branch_id
            }
            logger.debug(f"🌿 DEBUG: Returning created branch: {branch_data}")
            
            return web.json_response(
                {
                    "success": success,
                    "message": message,
                    "branch": branch_data,
                }
            )
        else:
            return web.json_response(
                {
                    "success": success,
                    "message": message,
                }
            )
    
    async def _handle_delete_branch(self, session_id: str, data: Dict[str, Any]) -> web.Response:
        """Handle branch deletion operation.
        
        Args:
            session_id: WebChat session ID
            data: Request data with branch_id to delete
            
        Returns:
            JSON response with deletion operation result
        """
        branch_id = data.get("branch_id")
        
        async def delete_branch_atomically(session):
            """Delete branch atomically within session lock."""
            logger.debug(f"🗑️  DEBUG: Branch delete requested - branch_id: '{branch_id}'")
            logger.debug(f"🗑️  DEBUG: Available branches before delete: {[b['id'] for b in session.get_enhanced_branch_info(self.db)]}")
            success, message = session.branch_manager.delete_branch(branch_id)
            logger.debug(f"🗑️  DEBUG: Branch delete result - success: {success}, message: '{message}'")
            logger.debug(f"🗑️  DEBUG: Available branches after delete: {[b['id'] for b in session.get_enhanced_branch_info(self.db)]}")
            return success, message, session
        
        # Execute branch deletion atomically
        success, message, session = await self.session_manager.execute_session_operation(
            session_id, 
            delete_branch_atomically
        )
        
        return web.json_response(
            {
                "success": success,
                "message": message,
                "branches": session.get_enhanced_branch_info(self.db),
                "current_branch": session.branch_manager.current_branch_id,
            }
        )
    
    async def handle_sync_conversation_api(self, request: web.Request) -> web.Response:
        """Sync conversation from the frontend to a specific session/branch.

        Requires a valid `session_id` in the POST body. Previously this endpoint
        implicitly defaulted to the "default" session when `session_id` was
        omitted; that behavior is no longer supported to avoid cross-session
        state leaks. If you intend to target a specific session, pass it
        explicitly (e.g., {"session_id": "session_..."}).
        
        Args:
            request: Web request with conversation data
            
        Returns:
            JSON response with sync operation result
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        try:
            session_id = extract_session_id(None, data)
        except ValueError as e:
            logger.warning(
                "sync_conversation called without session_id; rejecting with 400 (legacy implicit 'default' removed)"
            )
            return web.json_response(
                {
                    "error": str(e),
                    "hint": "Pass the intended session_id explicitly. Previous implicit 'default' fallback has been removed.",
                },
                status=400,
            )
        conversation = data.get("conversation", [])
        branch_id = data.get("branch_id")  # Optional target branch ID
        
        async def sync_conversation_with_branch(session):
            """Sync conversation to session and current/target branch."""
            # Update the session's conversation history
            session.conversation_history.clear()
            session.conversation_history.extend(conversation)
            session.update_activity()
            
            # Determine which branch to sync to
            target_branch_id = branch_id if branch_id else session.branch_manager.current_branch_id
            
            # Sync to the target branch (current by default)
            if target_branch_id in session.branch_manager.branches:
                target_branch = session.branch_manager.branches[target_branch_id]
                target_branch.conversation_history = copy.deepcopy(conversation)
                target_branch.last_active = datetime.now()
                logger.debug(f"🔄 Synced {len(conversation)} messages to branch '{target_branch_id}'")
                
                # Dual-write persistence (best-effort)
                if self.db:
                    try:
                        self.db.ensure_session(session_id)
                        conv_id = self.db.ensure_conversation(session_id)
                        # Mark session as persistent and record conversation id
                        session.current_conversation_id = conv_id
                        if not getattr(session, 'is_hydrated_from_db', False):
                            session.is_hydrated_from_db = True
                        # Ensure branch exists
                        self.db.ensure_branch(
                            conv_id, 
                            target_branch_id, 
                            name=target_branch.name
                        )
                        # Replace branch history with new conversation
                        self.db.bulk_add_branch_history(conv_id, target_branch_id, conversation)
                    except Exception as pe:
                        logger.warning(f"⚠️ Dual-write persistence (sync conversation) failed: {pe}")
            else:
                logger.warning(f"⚠️ Target branch '{target_branch_id}' not found, only updated session")
            
            return session
        
        # Use atomic operation for conversation sync
        session = await self.session_manager.execute_session_operation(
            session_id, 
            sync_conversation_with_branch
        )
        
        return web.json_response({"success": True})
    
    async def handle_get_conversation_api(self, request: web.Request) -> web.Response:
        """Handle getting conversation from backend session, optionally for a specific branch.
        
        Args:
            request: Web request with session and branch parameters
            
        Returns:
            JSON response with conversation data
        """
        try:
            session_id = extract_session_id(request)
            branch_id = request.query.get("branch_id")  # Optional branch ID
            
            session = await self.session_manager.get_or_create_session_safe(session_id, self.db)
        except ValueError as e:
            logger.warning(f"\u26a0\ufe0f get_conversation called without session_id: {e}")
            return web.json_response({"error": str(e)}, status=400)
        logger.debug(f"GET /conversation for session={session_id}, branch_id={branch_id}")
        
        # If branch_id is provided, return that branch's conversation
        if branch_id and branch_id in session.branch_manager.branches:
            target_branch = session.branch_manager.branches[branch_id]
            logger.debug(
                f"GET /conversation branch exists: current_branch={session.branch_manager.current_branch_id}, target_branch_len={len(target_branch.conversation_history)}"
            )
            conversation = []
            for msg in target_branch.conversation_history:
                if isinstance(msg, dict):
                    conversation.append({
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "timestamp": msg.get("timestamp", time.time()),
                    })
                else:
                    conversation.append({
                        "role": "unknown", 
                        "content": str(msg), 
                        "timestamp": time.time()
                    })
            logger.debug(f"🔍 Returning conversation for branch '{branch_id}': {len(conversation)} messages")
            return web.json_response({"conversation": conversation})
        elif branch_id:
            logger.warning(f"⚠️ Requested branch '{branch_id}' not found, returning current branch conversation")
        
        # Default: return current session conversation (current branch)
        logger.debug(f"🔍 Returning conversation for current branch '{session.branch_manager.current_branch_id}': {len(session.conversation_history)} messages")
        return web.json_response({"conversation": session.serialize_conversation()})
