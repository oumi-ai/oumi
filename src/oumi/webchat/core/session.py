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

"""WebChat session management for Oumi server."""

import asyncio
import copy
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any

from aiohttp import web
from rich.console import Console

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.commands.command_context import CommandContext
from oumi.core.commands.command_parser import CommandParser
from oumi.core.commands.command_router import CommandRouter
from oumi.core.commands.conversation_branches import ConversationBranchManager
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.monitoring import SystemMonitor
from oumi.core.thinking import ThinkingProcessor
from oumi.utils.logging import logger


class WebChatSession:
    """Manages a single webchat session with full command support."""

    def __init__(self, session_id: str, config: InferenceConfig):
        """Initialize a webchat session.

        Args:
            session_id: Unique session identifier.
            config: Inference configuration.
        """
        self.session_id = session_id
        self.config = config
        self.conversation_history = []
        
        # Add creation debugging
        from oumi.utils.logging import logger
        logger.debug(f"📝 Created new conversation history for session {session_id}, object id: {id(self.conversation_history)}")

        # Initialize core components
        self.console = Console()
        self.thinking_processor = ThinkingProcessor()
        self.branch_manager = ConversationBranchManager(self.conversation_history)
        
        # CRITICAL FIX: Ensure main branch shares the same conversation history reference
        # This fixes the core issue where branch manager's main branch gets out of sync
        if "main" in self.branch_manager.branches:
            self.branch_manager.branches["main"].conversation_history = self.conversation_history

        # Initialize SystemMonitor with proper context length (same as oumi chat)
        max_context_tokens = getattr(config.model, "model_max_length", None) or 4096
        self.system_monitor = SystemMonitor(max_context_tokens=max_context_tokens)

        # Initialize inference engine upfront (same as oumi chat)
        self.inference_engine = self._build_inference_engine(config)

        # Initialize command system
        self.command_context = CommandContext(
            console=self.console,
            config=config,
            conversation_history=self.conversation_history,
            inference_engine=self.inference_engine,
            system_monitor=self.system_monitor,
        )

        # Add additional components directly to the context using private attributes
        self.command_context._thinking_processor = self.thinking_processor
        self.command_context._branch_manager = self.branch_manager

        self.command_parser = CommandParser()
        self.command_router = CommandRouter(self.command_context)

        # WebSocket connections for this session
        self.websockets: Set[web.WebSocketResponse] = set()

        # Last activity timestamp for cleanup (using monotonic time)
        self.last_activity = time.monotonic()
        
        # Persistence tracking
        self.is_hydrated_from_db = False
        self.current_conversation_id: Optional[str] = None

    async def add_websocket(self, ws: web.WebSocketResponse):
        """Add a WebSocket connection to this session."""
        self.websockets.add(ws)

    async def remove_websocket(self, ws: web.WebSocketResponse):
        """Remove a WebSocket connection from this session."""
        self.websockets.discard(ws)

    async def broadcast_to_websockets(self, message: dict):
        """Broadcast a message to all WebSocket connections in this session."""
        if not self.websockets:
            return

        # Snapshot websockets to avoid races during iteration
        websockets_snapshot = set(self.websockets)
        message_str = json.dumps(message)
        closed_sockets = set()

        for ws in websockets_snapshot:
            try:
                await ws.send_str(message_str)
            except ConnectionResetError:
                closed_sockets.add(ws)

        # Clean up closed sockets
        for ws in closed_sockets:
            self.websockets.discard(ws)

    def _build_inference_engine(self, config: InferenceConfig):
        """Build the inference engine for this session (same as oumi chat)."""
        try:
            return build_inference_engine(
                engine_type=config.engine or InferenceEngineType.NATIVE,
                model_params=config.model,
                remote_params=config.remote_params,
                generation_params=config.generation,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize inference engine: {e}")
            return None

    def update_activity(self):
        """Update last activity timestamp using monotonic time."""
        self.last_activity = time.monotonic()

    def hydrate_from_db(self, db_data: dict) -> None:
        """Hydrate session state from database.
        
        Args:
            db_data: Result from WebchatDB.hydrate_session()
        """
        if not db_data:
            logger.debug(f"🗄️ No DB data to hydrate for session {self.session_id}")
            return
        
        session_info = db_data["session_info"]
        branches = db_data["branches"]
        current_messages = db_data["current_messages"]
        current_branch_id = db_data["current_branch_id"]
        
        # Set conversation ID
        self.current_conversation_id = session_info["current_conversation_id"]
        
        # Hydrate conversation history from current branch
        self.conversation_history.clear()
        for msg in current_messages:
            self.conversation_history.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "metadata": msg.get("metadata", {})
            })
        
        # Recreate branch manager with hydrated data
        self.branch_manager = ConversationBranchManager(self.conversation_history)
        
        # Set current branch
        if current_branch_id and current_branch_id in self.branch_manager.branches:
            self.branch_manager.current_branch_id = current_branch_id
            
        # Update branch manager's branches with DB data
        for branch_data in branches:
            branch_id = branch_data["id"]
            if branch_id in self.branch_manager.branches:
                branch = self.branch_manager.branches[branch_id]
                branch.name = branch_data.get("name", branch_id)
                branch.parent_branch_id = branch_data.get("parent_branch_id")
                # Note: We only load current branch messages; other branches load on-demand
                if branch_id == current_branch_id:
                    branch.conversation_history = self.conversation_history
        
        self.is_hydrated_from_db = True
        logger.info(f"🗄️ Session {self.session_id} hydrated from DB: {len(current_messages)} messages, {len(branches)} branches")
        
        # Re-sync the command context
        self.command_context._conversation_history = self.conversation_history
        self.command_context._branch_manager = self.branch_manager
        
        # Additional logging for debugging hydration
        from oumi.utils.logging import logger
        logger.debug(f"🗄️ After hydration: conversation history length={len(self.conversation_history)}, object id: {id(self.conversation_history)}")
        logger.debug(f"🗄️ After hydration: branch manager main history length={len(self.branch_manager.branches['main'].conversation_history) if 'main' in self.branch_manager.branches else 0}")
        # Check if they're the same object
        if 'main' in self.branch_manager.branches:
            main_branch = self.branch_manager.branches['main']
            is_same_object = id(main_branch.conversation_history) == id(self.conversation_history)
            logger.debug(f"🗄️ After hydration: main branch history is same object as session history: {is_same_object}")
            if not is_same_object:
                logger.error(f"🚨 CRITICAL ERROR: Branch manager main history is not same object as session history!")

    def hydrate_branch_from_db(self, branch_id: str, db) -> bool:
        """Hydrate a specific branch's messages from database on-demand.
        
        Args:
            branch_id: Branch ID to hydrate
            db: WebchatDB instance
            
        Returns:
            True if hydrated successfully, False otherwise
        """
        if not db or not self.current_conversation_id:
            return False
            
        try:
            # Get messages for this specific branch
            branch_messages = db.get_branch_messages(branch_id)
            
            # Find the branch object
            if branch_id not in self.branch_manager.branches:
                logger.warning(f"🗄️ Branch {branch_id} not found in branch manager")
                return False
                
            branch = self.branch_manager.branches[branch_id]
            
            # Update branch conversation history
            branch.conversation_history.clear()
            for msg in branch_messages:
                branch.conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"],
                    "metadata": msg.get("metadata", {})
                })
            
            logger.debug(f"🗄️ Hydrated branch {branch_id} with {len(branch_messages)} messages from DB")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to hydrate branch {branch_id} from DB: {e}")
            return False

    def get_enhanced_branch_info(self, db) -> List[Dict[str, Any]]:
        """Get branch information enhanced with database-backed counts and previews.
        
        Falls back to in-memory calculation if DB unavailable.
        """
        # Start with standard branch info
        branches_info = self.branch_manager.list_branches()
        
        # Enhance with DB data if available and session is persistent
        if db and self.is_hydrated_from_db and self.current_conversation_id:
            try:
                # Get DB branch information
                db_branches = db.get_session_branches(self.session_id)
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
                                recent_messages = db.get_branch_messages(branch_id)
                                if recent_messages:
                                    last_msg = recent_messages[-1]
                                    content = last_msg["content"][:50]
                                    branch_info["preview"] = f"[{last_msg['role']}] {content}..." if content else "(empty message)"
                            except Exception as preview_error:
                                logger.debug(f"⚠️ Failed to generate DB preview for branch {branch_id}: {preview_error}")
                
                logger.debug(f"🗄️ Enhanced {len(branches_info)} branches with DB-backed counts")
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to enhance branch info with DB data: {e}")
                # Fall back to in-memory data (already populated)
        
        return branches_info

    def serialize_conversation(self) -> List[Dict[str, Any]]:
        """Serialize conversation history for web interface."""
        result = []
        for msg in self.conversation_history:
            if isinstance(msg, dict):
                result.append(
                    {
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "timestamp": msg.get("timestamp", time.time()),
                    }
                )
            else:
                # Handle other message formats if needed
                result.append(
                    {"role": "unknown", "content": str(msg), "timestamp": time.time()}
                )
        return result

    async def _perform_regeneration_inference(self, user_input_override: str):
        """Perform inference for regeneration commands using the user_input_override.
        
        Args:
            user_input_override: The edited user input to use for regeneration
        """
        logger.info(f"🔄 Starting regeneration inference with updated user input")
        logger.info(f"🔄 User input override: {user_input_override[:100]}...")
        
        # Don't add user_input_override as new message - the edit command already updated the original message
        # Just use the user_input_override for inference to ensure we use the edited content
        
        # Broadcast thinking indicator for regeneration
        await self.broadcast_to_websockets(
            {"type": "assistant_thinking", "timestamp": time.time()}
        )
        
        # Build conversation using the existing conversation history
        # The conversation was already truncated by the regen command
        try:
            from oumi.core.types.conversation import Conversation, Message, Role
            
            # Convert conversation history to Message objects
            conversation_messages = []
            
            # Add system prompt if configured
            if hasattr(self, 'system_prompt') and self.system_prompt:
                conversation_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
            
            # Add existing conversation history, but use user_input_override for the final user message
            for i, msg in enumerate(self.conversation_history):
                if msg.get("role") == "user":
                    # For the last user message, use user_input_override to ensure we have the edited content
                    if i == len(self.conversation_history) - 1 and msg.get("role") == "user":
                        conversation_messages.append(Message(role=Role.USER, content=user_input_override))
                    else:
                        conversation_messages.append(Message(role=Role.USER, content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    conversation_messages.append(Message(role=Role.ASSISTANT, content=msg.get("content", "")))
            
            # Create conversation object
            full_conversation = Conversation(messages=conversation_messages)
            
            logger.info(f"🔄 Built conversation with {len(conversation_messages)} messages for regeneration")
            logger.info(f"🔄 Final user message content: {conversation_messages[-1].content[:100] if conversation_messages and conversation_messages[-1].role == Role.USER else 'None'}...")
            
            # Perform inference using the same logic as OpenAI API handler
            logger.info(f"🚀 Calling inference_engine.infer() for regeneration")
            start_time = time.time()
            
            model_response = self.inference_engine.infer(
                input=[full_conversation],
                inference_config=self.config,
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Regeneration inference completed in {elapsed:.2f} seconds")
            
            # Extract response content
            response_content = ""
            if model_response:
                last_conversation = model_response[-1] if isinstance(model_response, list) else model_response
                for message in reversed(last_conversation.messages):
                    if message.role == Role.ASSISTANT and isinstance(message.content, str):
                        response_content = message.content
                        break
            
            if not response_content:
                response_content = "No response generated"
            
            # Check if there's an unexpected assistant response at the end that regen should have removed
            if self.conversation_history and self.conversation_history[-1].get("role") == "assistant":
                logger.warning(f"🔄 Found unexpected assistant response at end - regen should have cleaned this up")
                logger.warning(f"🔄 Removing 1 assistant message: {self.conversation_history[-1].get('content', '')[:50]}...")
                self.conversation_history.pop()
                # Only remove ONE message - if there are more, something else is wrong
                if self.conversation_history and self.conversation_history[-1].get("role") == "assistant":
                    logger.error(f"🔄 Multiple assistant messages found - this indicates a deeper issue!")
                    # Don't remove more - let it be visible so we can debug
            
            # Add the new assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_content,
                "timestamp": time.time()
            })
            
            logger.info(f"🔄 Added regenerated response to conversation (total messages: {len(self.conversation_history)})")
            
            # Broadcast assistant response to WebSockets
            await self.broadcast_to_websockets(
                {
                    "type": "assistant_message",
                    "content": response_content,
                    "timestamp": time.time()
                }
            )
            
            logger.info(f"🔄 Regeneration completed successfully - {len(response_content)} chars generated")
            
        except Exception as e:
            logger.error(f"❌ Regeneration inference failed: {e}")
            error_message = f"Regeneration failed: {str(e)}"
            
            # Add error message to conversation
            self.conversation_history.append({
                "role": "assistant", 
                "content": error_message,
                "timestamp": time.time()
            })
            
            # Broadcast error
            await self.broadcast_to_websockets(
                {
                    "type": "assistant_message",
                    "content": error_message,
                    "timestamp": time.time()
                }
            )