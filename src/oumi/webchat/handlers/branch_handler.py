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
        """Handle branch deletion operation with persistence and safety checks.
        
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
            
            # Get current conversation ID if session has persistence
            conversation_id = getattr(session, 'current_conversation_id', None)
            
            # Create branch service instance if not already available
            from oumi.webchat.services.branch_service import BranchService
            from oumi.webchat.services.persistence_service import PersistenceService
            
            # Create persistence service (pass through to DB)
            persistence = None
            if self.db:
                try:
                    persistence = PersistenceService(self.db.db_path)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create persistence service: {e}")
            
            # Create branch service
            branch_service = BranchService(persistence)
            
            # Use enhanced branch service to delete branch with full safety checks
            success, message, db_result = branch_service.delete_branch(
                branch_manager=session.branch_manager,
                branch_id=branch_id,
                session_id=session_id,
                conversation_id=conversation_id
            )
            
            # Enhanced debug logging
            logger.debug(f"🗑️  DEBUG: Branch delete result - success: {success}, message: '{message}'")
            if db_result:
                logger.debug(f"🗑️  DEBUG: DB deleted {db_result.get('deleted_message_count', 0)} messages")
                if 'deleted_edges_count' in db_result:
                    logger.debug(f"🗑️  DEBUG: DB deleted {db_result.get('deleted_edges_count', 0)} graph edges")
            logger.debug(f"🗑️  DEBUG: Available branches after delete: {[b['id'] for b in session.get_enhanced_branch_info(self.db)]}")
            
            # Build detailed result with full information
            result = {
                "success": success,
                "message": message,
                "branches": session.get_enhanced_branch_info(self.db),
                "current_branch": session.branch_manager.current_branch_id,
            }
            
            # Add DB operation details if available
            if db_result:
                result["persistence"] = {
                    "deleted_message_count": db_result.get("deleted_message_count", 0),
                    "deleted_edges_count": db_result.get("deleted_edges_count", 0) if "deleted_edges_count" in db_result else 0
                }
            
            return success, message, session, result
        
        # Execute branch deletion atomically
        success, message, session, result = await self.session_manager.execute_session_operation(
            session_id, 
            delete_branch_atomically
        )
        
        # Determine appropriate status code based on result
        status_code = 200 if success else 400
        
        # If operation succeeded, broadcast update to all WebSocket clients
        if success:
            try:
                await session.broadcast_to_websockets(
                    {
                        "type": "conversation_update",
                        "branches": session.get_enhanced_branch_info(self.db),
                        "current_branch": session.branch_manager.current_branch_id,
                        "timestamp": time.time(),
                    }
                )
                logger.debug(f"✅ Broadcast branch update after deletion")
            except Exception as e:
                logger.warning(f"⚠️ Failed to broadcast branch update: {e}")
        
        return web.json_response(result, status=status_code)
    
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
    
    async def handle_reset_history_api(self, request: web.Request) -> web.Response:
        """Handle resetting chat history via REST API.
        
        This is a destructive operation that clears all conversations and branches.
        It can either reset a specific session or all sessions.
        
        Args:
            request: Web request with reset operation parameters
            
        Returns:
            JSON response with reset operation result
        """
        try:
            data = await request.json()
            
            # Extract session_id and optional parameters
            try:
                session_id = extract_session_id(request, data)
                session = await self.session_manager.get_or_create_session_safe(session_id, self.db)
            except ValueError as e:
                logger.warning(f"⚠️ Reset history API error: {e}")
                return web.json_response({"error": str(e)}, status=400)
                
            # Get optional parameters
            scope = data.get("scope", "current")  # "current" or "all"
            confirm_phrase = data.get("confirm_phrase", "")
            
            # Validation
            if confirm_phrase != "RESET":
                return web.json_response(
                    {
                        "error": "Invalid confirmation phrase. Must be 'RESET'.",
                        "success": False,
                    },
                    status=400
                )
            
            # Count of items to be deleted for telemetry
            delete_counts = {
                "branches": 0,
                "messages": 0,
                "sessions": 0,
                "embeddings": 0,
            }
            
            if scope == "all" and not data.get("admin_override"):
                return web.json_response(
                    {
                        "error": "Resetting all sessions requires admin_override=true parameter.",
                        "success": False,
                    },
                    status=403
                )
            
            # Generate a unique operation ID for this reset
            import uuid
            operation_id = f"reset_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"🧹 Starting chat history reset (scope={scope}, operation_id={operation_id})")
            
            # Single session reset function
            async def reset_session_atomically(session):
                """Reset a single session atomically within session lock."""
                # Count items before deletion for telemetry
                branch_count = len(session.branch_manager.branches)
                message_count = 0
                for branch_id, branch in session.branch_manager.branches.items():
                    message_count += len(branch.conversation_history)
                
                logger.info(f"🧹 Resetting session {session_id}: {branch_count} branches, {message_count} messages")
                delete_counts["branches"] += branch_count
                delete_counts["messages"] += message_count
                
                # Get current branch ID for later use
                current_branch_id = session.branch_manager.current_branch_id
                
                # Reset in-memory branch manager state
                for branch_id in list(session.branch_manager.branches.keys()):
                    if branch_id != "main":
                        # Delete all branches except main
                        del session.branch_manager.branches[branch_id]
                
                # Reset main branch to empty state but keep it
                if "main" in session.branch_manager.branches:
                    session.branch_manager.branches["main"].conversation_history = []
                    session.branch_manager.branches["main"].last_active = datetime.now()
                
                # Reset branch counter
                session.branch_manager._branch_counter = 1
                
                # Switch to main branch
                session.branch_manager.current_branch_id = "main"
                
                # Reset session conversation
                session.conversation_history.clear()
                
                # Reset system monitor if exists
                if hasattr(session, "system_monitor"):
                    max_context_tokens = getattr(session.config.model, "model_max_length", None) or 4096
                    session.system_monitor.reset(max_context_tokens=max_context_tokens)
                
                # Update activity timestamp
                session.update_activity()
                
                # Database cleanup for this session if available
                if self.db and hasattr(session, 'current_conversation_id'):
                    try:
                        conv_id = session.current_conversation_id
                        
                        # Get all branch IDs in the conversation
                        branch_ids = self.db.list_branches(conv_id)
                        
                        # Delete all branches except main
                        for branch_id in branch_ids:
                            if branch_id != "main":
                                self.db.delete_branch(conv_id, branch_id)
                        
                        # Clear main branch
                        if "main" in branch_ids:
                            self.db.bulk_add_branch_history(conv_id, "main", [])
                            
                        # Reset session current branch to main
                        self.db.set_session_current_branch(session_id, conv_id, "main")
                        
                        logger.info(f"🧹 Database cleanup complete for session {session_id}")
                    except Exception as e:
                        logger.error(f"❌ Database cleanup failed for session {session_id}: {e}")
                
                logger.info(f"🧹 Session {session_id} reset complete")
                
                return True, "Session reset successful", session
            
            # Execute reset operation atomically
            if scope == "current":
                # Reset only the current session
                success, message, session = await self.session_manager.execute_session_operation(
                    session_id, 
                    reset_session_atomically
                )
                
                delete_counts["sessions"] = 1
                
                # If operation succeeded, broadcast update to all WebSocket clients
                if success:
                    try:
                        await session.broadcast_to_websockets(
                            {
                                "type": "conversation_update",
                                "branches": session.get_enhanced_branch_info(self.db),
                                "current_branch": session.branch_manager.current_branch_id,
                                "conversation": session.serialize_conversation(),
                                "timestamp": time.time(),
                                "reset": True,
                            }
                        )
                        logger.debug(f"✅ Broadcast conversation update after reset")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to broadcast conversation update: {e}")
            
            else:  # scope == "all"
                # Reset all sessions (admin only)
                success = True
                message = "All sessions reset successful"
                session_count = 0
                
                # Get all session IDs
                all_session_ids = await self.session_manager.list_sessions()
                
                # Reset each session
                for sid in all_session_ids:
                    try:
                        s_success, s_message, _ = await self.session_manager.execute_session_operation(
                            sid, 
                            reset_session_atomically
                        )
                        
                        if s_success:
                            session_count += 1
                        else:
                            logger.error(f"❌ Failed to reset session {sid}: {s_message}")
                            # Don't fail completely if one session fails
                            message = f"Partial reset: {session_count}/{len(all_session_ids)} sessions reset"
                    except Exception as e:
                        logger.error(f"❌ Exception during reset of session {sid}: {e}")
                        message = f"Partial reset: {session_count}/{len(all_session_ids)} sessions reset"
                
                delete_counts["sessions"] = session_count
                
                # Clean up vector indexes and other resources
                # This is a placeholder for any additional cleanup needed
                try:
                    # Example: clean embedding indexes
                    # if hasattr(self.db, 'clean_embedding_indexes'):
                    #     deleted_count = self.db.clean_embedding_indexes()
                    #     delete_counts["embeddings"] = deleted_count
                    pass
                except Exception as e:
                    logger.error(f"❌ Failed to clean resource indexes: {e}")
            
            # Log telemetry for the reset operation
            try:
                import json
                logger.info(
                    f"🧹 Reset operation complete: operation_id={operation_id}, "
                    f"scope={scope}, counts={json.dumps(delete_counts)}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to log telemetry: {e}")
            
            # Return success response with counts
            return web.json_response(
                {
                    "success": success,
                    "message": message,
                    "operation_id": operation_id,
                    "timestamp": time.time(),
                    "scope": scope,
                    "deleted": delete_counts,
                }
            )
            
        except Exception as e:
            logger.error(f"Reset history API error: {e}")
            return web.json_response({"error": str(e), "success": False}, status=500)
