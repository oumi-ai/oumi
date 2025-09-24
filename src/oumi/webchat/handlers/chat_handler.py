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

"""Chat completion endpoints handler for Oumi WebChat server."""

import copy
import time
from typing import Any, Dict, List, Optional

from aiohttp import web

from oumi.core.types.conversation import Conversation, ContentItem, Message, Role, Type
from oumi.utils.logging import logger
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.chatgraph_migration.graph_store import GraphStore
from oumi.webchat.protocol import extract_session_id, extract_branch_id
from oumi.webchat.utils.id_utils import generate_message_id
from oumi.webchat.utils.fallbacks import model_name_fallback


class ChatHandler:
    """Handles chat completion requests for Oumi WebChat."""
    
    def __init__(
        self, 
        session_manager: SessionManager,
        system_prompt: Optional[str] = None,
        db = None,
        enhanced_features_available: bool = False
    ):
        """Initialize chat handler.
        
        Args:
            session_manager: Session manager for WebChat sessions
            system_prompt: Optional system prompt for conversations
            db: Optional WebchatDB instance for persistence
            enhanced_features_available: Whether enhanced API features are available
        """
        self.session_manager = session_manager
        self.system_prompt = system_prompt
        self.db = db
        self.enhanced_features_available = enhanced_features_available
        self.response_formatter = None
        self.request_validator = None
        
        # Initialize enhanced components if available
        if enhanced_features_available:
            try:
                from oumi.webchat.api_responses import ResponseFormatter, RequestValidator
                self.response_formatter = ResponseFormatter()
                self.request_validator = RequestValidator()
            except ImportError:
                logger.warning("Enhanced API components could not be imported")

    def _convert_client_content(self, content: Any) -> Any:
        """Normalizes client-provided message content into Oumi types."""

        if isinstance(content, list):
            items: list[ContentItem] = []

            for part in content:
                if not isinstance(part, dict):
                    continue

                part_type = str(part.get("type", "")).lower()

                def _extract_value(*keys: str) -> Optional[str]:
                    for key in keys:
                        value = part.get(key)
                        if isinstance(value, str) and value:
                            return value
                        if isinstance(value, dict):
                            url = value.get("url")
                            if isinstance(url, str) and url:
                                return url
                    return None

                if part_type == "text":
                    value = _extract_value("content", "text") or ""
                    items.append(ContentItem(type=Type.TEXT, content=value))
                elif part_type == "image_url":
                    value = _extract_value("content", "image_url", "url")
                    if value:
                        items.append(ContentItem(type=Type.IMAGE_URL, content=value))
                elif part_type == "audio_url":
                    value = _extract_value("content", "audio_url", "url")
                    if value:
                        items.append(ContentItem(type=Type.AUDIO_URL, content=value))
                elif part_type == "video_url":
                    value = _extract_value("content", "video_url", "url")
                    if value:
                        items.append(ContentItem(type=Type.VIDEO_URL, content=value))
                elif part_type == "image_path":
                    value = _extract_value("content", "path")
                    if value:
                        items.append(ContentItem(type=Type.IMAGE_PATH, content=value))
                elif part_type == "audio_path":
                    value = _extract_value("content", "path")
                    if value:
                        items.append(ContentItem(type=Type.AUDIO_PATH, content=value))
                elif part_type == "video_path":
                    value = _extract_value("content", "path")
                    if value:
                        items.append(ContentItem(type=Type.VIDEO_PATH, content=value))

            if items:
                return items

            # Fallback: stringify any remaining payload to avoid empty content errors
            return "\n".join(
                str(part) for part in content if part is not None
            )

        if isinstance(content, str):
            return content

        if content is None:
            return ""

        return str(content)

    def _content_items_to_text(self, content: list[ContentItem]) -> str:
        segments: list[str] = []
        for item in content:
            if item.type == Type.TEXT:
                segments.append(item.content or "")
            else:
                segments.append(f"<{item.type.value}>")
        return " ".join(segment for segment in segments if segment)

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """Handle chat completions requests in OpenAI format.
        
        Args:
            request: Web request with chat completion parameters
            
        Returns:
            JSON response with completion result
        """
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {
                    "error": {
                        "message": f"Invalid JSON: {str(e)}",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )
        
        # Extract required fields
        messages = data.get("messages", [])
        if not messages:
            return web.json_response(
                {
                    "error": {
                        "message": "messages field is required",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )
        
        # Extract optional fields
        model = data.get("model", model_name_fallback("request.model"))
        temperature = data.get("temperature", 1.0)
        max_tokens = data.get("max_tokens", 100)
        stream = data.get("stream", False)
        
        # Extract session_id and branch_id with consistent handling
        # We don't require these parameters as this endpoint can work without a session
        session_id = extract_session_id(None, data, required=False)  # WebChat session ID
        branch_id = data.get("branch_id")  # Target branch ID for this chat request
        
        try:
            # Convert OpenAI format messages to Oumi conversation format
            oumi_messages = []
            
            # Add system prompt if provided
            if self.system_prompt:
                oumi_messages.append(
                    Message(role=Role.SYSTEM, content=self.system_prompt)
                )
            
            # Convert messages
            for msg in messages:
                role_mapping = {
                    "system": Role.SYSTEM,
                    "user": Role.USER,
                    "assistant": Role.ASSISTANT,
                }
                role = role_mapping.get(msg.get("role"), Role.USER)
                raw_content = msg.get("content", "")
                normalized_content = self._convert_client_content(raw_content)
                oumi_messages.append(Message(role=role, content=normalized_content))
            
            # Get the latest user message for inference
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                return web.json_response(
                    {
                        "error": {
                            "message": "No user message found",
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )
            
            latest_user_raw = user_messages[-1].get("content", "")
            latest_user_content = self._convert_client_content(latest_user_raw)
            latest_user_text = (
                self._content_items_to_text(latest_user_content)
                if isinstance(latest_user_content, list)
                else str(latest_user_content)
            )
            
            # CRITICAL: If we have a session_id, use the session for branch-aware chat
            if session_id:
                async def handle_branch_aware_chat(session):
                    """Handle branch-aware chat completion within session lock."""
                    # Use local variable to avoid UnboundLocalError
                    effective_branch_id = branch_id
                    
                    # Handle branch switching if branch_id provided
                    if effective_branch_id and effective_branch_id != session.branch_manager.current_branch_id:
                        logger.debug(f"🌿 Chat request targeting branch '{effective_branch_id}', current: '{session.branch_manager.current_branch_id}'")
                        
                        # Save current conversation to current branch
                        current_branch = session.branch_manager.get_current_branch()
                        current_branch.conversation_history = copy.deepcopy(session.conversation_history)
                        current_branch.last_active = time.time()
                        logger.debug(f"🔄 Saved {len(session.conversation_history)} messages to branch '{session.branch_manager.current_branch_id}'")
                        
                        # Switch to target branch and load its conversation
                        if effective_branch_id in session.branch_manager.branches:
                            target_branch = session.branch_manager.branches[effective_branch_id]
                            session.branch_manager.current_branch_id = effective_branch_id
                            
                            # Load target branch conversation into session
                            session.conversation_history.clear()
                            session.conversation_history.extend(target_branch.conversation_history)
                            target_branch.last_active = time.time()
                            logger.debug(f"🔄 Loaded {len(session.conversation_history)} messages from branch '{effective_branch_id}'")
                        else:
                            logger.error(f"🚨 Target branch '{effective_branch_id}' not found, staying on current branch")
                            effective_branch_id = session.branch_manager.current_branch_id  # Reset to current
                    elif effective_branch_id:
                        logger.debug(f"🌿 Chat request already on target branch '{effective_branch_id}'")
                    else:
                        # Use current branch if no branch_id specified
                        effective_branch_id = session.branch_manager.current_branch_id
                        logger.debug(f"🌿 Chat request using current branch '{effective_branch_id}' (no branch_id specified)")
                    
                    # Create a snapshot of conversation history for consistency
                    conversation_snapshot = copy.deepcopy(session.conversation_history)
                    
                    return session, effective_branch_id, conversation_snapshot
                
                # Use atomic operation for branch switching
                session, effective_branch_id, conversation_snapshot = await self.session_manager.execute_session_operation(
                    session_id, 
                    handle_branch_aware_chat
                )
                
                # Update branch_id to the one actually used
                branch_id = effective_branch_id 
                
                # CRITICAL: Prefer client-provided messages as the authoritative
                # conversation context to avoid leaking history from previous
                # UI conversations within the same backend session.
                conversation_messages = []
                if self.system_prompt:
                    conversation_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
                # Convert request messages directly for inference context
                for m in messages:
                    role_mapping = {
                        "system": Role.SYSTEM,
                        "user": Role.USER,
                        "assistant": Role.ASSISTANT,
                    }
                    r = role_mapping.get(m.get("role"), Role.USER)
                    c = m.get("content", "")
                    conversation_messages.append(Message(role=r, content=c))
                logger.debug(f"🧠 Using client-provided conversation with {len(messages)} messages for context")
                
                # Create the conversation object
                full_conversation = Conversation(messages=conversation_messages)
                
                logger.debug(f"🧠 Using full conversation with {len(conversation_messages)} total messages for context")
                
                # Get the current inference engine and config from the session
                session_config = session.config
                session_engine = session.inference_engine
                
                # Use swapped engine/config if available
                if hasattr(session.command_context, 'config') and session.command_context.config:
                    session_config = session.command_context.config
                    logger.debug(f"🔄 Using session's swapped config: {getattr(session_config.model, 'model_name', 'Unknown')}")
                
                if hasattr(session.command_context, 'inference_engine') and session.command_context.inference_engine:
                    session_engine = session.command_context.inference_engine
                    logger.debug(f"🔄 Using session's swapped inference engine")
                
            else:
                # Fallback: create a simple conversation with just the latest message
                conversation_messages = []
                
                # Add system prompt if provided
                if self.system_prompt:
                    conversation_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
                
                conversation_messages.append(Message(role=Role.USER, content=latest_user_content))
                full_conversation = Conversation(messages=conversation_messages)
                logger.debug(f"🧠 No session context, using single message conversation")
                
                # Use the session manager's default config and create a temporary engine
                from oumi.infer import get_engine
                session_config = self.session_manager.default_config
                session_engine = get_engine(session_config)
            
            # Determine the effective model name actually used for this request
            try:
                effective_model_name = getattr(session_config.model, 'model_name', None)
            except Exception:
                effective_model_name = None
            if not effective_model_name:
                effective_model_name = model_name_fallback("session_config.model.model_name")

            # CRITICAL FIX: Use the SAME proven logic as "oumi chat" command
            # This is the battle-tested approach that works with all engine types
            try:
                logger.info(f"🧠 Starting inference with {session_config.engine} engine")
                logger.info(f"🧠 Conversation has {len(full_conversation.messages)} messages")
                logger.info(f"🧠 Model: {getattr(session_config.model, 'model_name', 'Unknown')}")
                
                # Log conversation content for debugging
                for i, msg in enumerate(full_conversation.messages):
                    logger.debug(f"Message {i}: {msg.role} - {str(msg.content)[:100]}...")
                
                logger.info(f"🚀 Calling inference_engine.infer() with {session_config.engine} engine")
                for idx, cmsg in enumerate(full_conversation.messages):
                    logger.debug(
                        "[infer] Conversation message %d role=%s type=%s", 
                        idx,
                        cmsg.role,
                        type(cmsg.content),
                    )
                    if isinstance(cmsg.content, list):
                        for part_idx, part in enumerate(cmsg.content):
                            logger.debug(
                                "[infer]   part %d -> type=%s content=%s",
                                part_idx,
                                getattr(part, 'type', None),
                                getattr(part, 'content', None)[:60] if getattr(part, 'content', None) else None,
                            )
                start_time = time.time()
                
                # Use the same inference engine interface as oumi chat command
                model_response = session_engine.infer(
                    input=[full_conversation],  # List containing one conversation object
                    inference_config=session_config,
                )
                
                elapsed = time.time() - start_time
                logger.info(f"✅ inference_engine.infer() completed in {elapsed:.2f} seconds")
                
                # Extract the response using the same logic as oumi chat
                response_content = ""
                if model_response:
                    # Get the last conversation from the response
                    last_conversation = model_response[-1] if isinstance(model_response, list) else model_response
                    
                    # Find the assistant's response in reverse order (most recent first)
                    for message in reversed(last_conversation.messages):
                        if message.role == Role.ASSISTANT and isinstance(message.content, str):
                            response_content = message.content
                            break
                
                if not response_content:
                    response_content = "No response generated"
                    
                logger.debug(f"✅ Got response from inference engine: {len(response_content)} chars")
                
            except Exception as e:
                logger.error("❌ Inference engine call failed: %s", e)
                logger.exception(e)
                raise
            
            # Check if we have a valid response
            if not response_content or response_content.startswith("Inference failed:"):
                msg = response_content or "No response generated"
                return web.json_response(
                    {
                        "message": msg,
                        "error": {
                            "message": msg,
                            "type": "server_error",
                        }
                    },
                    status=500,
                )
            
            # Format response in OpenAI format
            response_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": effective_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(latest_user_text.split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(latest_user_text.split()) + len(response_content.split()),
                },
            }
            
            # Update WebChat session if session_id provided
            if session_id:
                logger.info(
                    f"🔍 Updating WebChat session {session_id} from OpenAI API (branch: {branch_id})"
                )
                
                async def update_conversation_with_branch(session):
                    """Update conversation for the active branch without trusting stale client history.

                    Rules:
                    - If this is a fresh session (no history yet), seed from client messages once.
                    - Otherwise, append only the latest user message from the request, then the assistant reply.
                    - Keep the active branch sharing the SAME list object as the session for integrity.
                    """
                    is_fresh_session = len(session.conversation_history) == 0

                    if is_fresh_session:
                        # Seed the new session with client-provided messages (first request only)
                        session.conversation_history.clear()
                        for m in messages:
                            session.conversation_history.append(
                                {
                                    "id": m.get("id") or generate_message_id(),
                                    "role": m.get("role", "user"),
                                    "content": m.get("content", ""),
                                    "timestamp": time.time(),
                                }
                            )
                    else:
                        # Append only the latest user message to server-authoritative history
                        # Identify the latest user message in the request
                        last_user = None
                        for m in reversed(messages):
                            if m.get("role") == "user":
                                last_user = m
                                break
                        if last_user is not None:
                            session.conversation_history.append(
                                {
                                    "id": last_user.get("id") or generate_message_id(),
                                    "role": "user",
                                    "content": last_user.get("content", ""),
                                    "timestamp": time.time(),
                                }
                            )

                    # Append assistant response with model metadata
                    try:
                        model_name = getattr(session_config.model, 'model_name', None)
                    except Exception:
                        model_name = None
                    try:
                        engine_name = str(session_config.engine) if getattr(session_config, 'engine', None) else None
                    except Exception:
                        engine_name = None
                    session.conversation_history.append(
                        {
                            "id": generate_message_id(),
                            "role": "assistant",
                            "content": response_content,
                            "timestamp": time.time(),
                            "metadata": {
                                "model_name": model_name,
                                "engine": engine_name,
                                "duration_ms": int(max(0.0, (elapsed if 'elapsed' in locals() else 0.0)) * 1000),
                            },
                        }
                    )

                    # Sync conversation to the active branch: share SAME object for integrity
                    current_branch = session.branch_manager.get_current_branch()
                    current_branch.conversation_history = session.conversation_history
                    # Use datetime for branch timestamps to match ConversationBranch
                    from datetime import datetime as _dt
                    current_branch.last_active = _dt.now()
                    
                    logger.info(
                        f"🔄 Synced conversation to branch '{session.branch_manager.current_branch_id}' - {len(session.conversation_history)} messages"
                    )
                    
                    # Dual-write persistence (best-effort)
                    try:
                        if self.db:
                            self.db.ensure_session(session_id)
                            conv_id = self.db.ensure_conversation(session_id)
                            # Mark session as persistent and record conversation id
                            session.current_conversation_id = conv_id
                            if not getattr(session, 'is_hydrated_from_db', False):
                                session.is_hydrated_from_db = True
                            # Ensure branch exists
                            self.db.ensure_branch(
                                conv_id, 
                                session.branch_manager.current_branch_id, 
                                name=session.branch_manager.current_branch_id
                            )
                            # Append the last two messages (user + assistant) and align IDs with DB ids
                            if len(session.conversation_history) >= 2:
                                last_two = session.conversation_history[-2:]
                                import json as _json
                                assistant_db_id = None
                                for m in last_two:
                                    db_id = self.db.append_message_to_branch(
                                        conv_id,
                                        session.branch_manager.current_branch_id,
                                        role=m.get("role", "user"),
                                        content=str(m.get("content", "")),
                                        created_at=float(m.get("timestamp", time.time())),
                                        metadata=_json.dumps(m.get("metadata", {})) if m.get("metadata") else None,
                                    )
                                    try:
                                        m["id"] = db_id
                                    except Exception:
                                        pass
                                    if m.get("role") == "assistant":
                                        assistant_db_id = db_id
                                # Graph dual-write using message id if message-based schema is present
                                try:
                                    gs_msg = GraphStore(self.db.db_path)
                                    if assistant_db_id:
                                        gs_msg.add_edge_for_message_tail(conv_id, assistant_db_id)
                                except Exception as _gge:
                                    logger.warning(f"[Graph] message-tail dual-write failed: {_gge}")
                            # Update session's current branch record
                            self.db.set_session_current_branch(
                                session_id, 
                                conv_id, 
                                session.branch_manager.current_branch_id
                            )
                            
                            # Graph dual-write
                            try:
                                # Pass exact DB path to GraphStore
                                gs = GraphStore(self.db.db_path)
                                logger.debug(f"[Graph] Using GraphStore at {self.db.db_path}")
                                gs.add_edge_for_branch_tail(
                                    conv_id, session.branch_manager.current_branch_id
                                )
                                logger.debug("[Graph] add_edge_for_branch_tail succeeded")
                            except Exception as ge:
                                logger.warning(f"Graph dual-write failed: {type(ge).__name__}: {ge}")
                                # Extra diagnostics: list graph_edges columns if possible
                                try:
                                    import sqlite3
                                    conn = sqlite3.connect(self.db.db_path)
                                    cur = conn.cursor()
                                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_edges'")
                                    has_table = bool(cur.fetchone())
                                    if has_table:
                                        cur.execute("PRAGMA table_info('graph_edges')")
                                        cols = [r[1] for r in cur.fetchall()]
                                        logger.warning(f"[Graph] graph_edges columns: {cols}")
                                    else:
                                        logger.warning("[Graph] graph_edges table not found")
                                    conn.close()
                                except Exception as diag_err:
                                    logger.warning(f"[Graph] column inspect failed: {diag_err}")
                    except Exception as pe:
                        logger.warning(f"⚠️ Dual-write persistence failed: {pe}")
                    
                    # Update context usage
                    self.session_manager.update_context_usage(session_id)
                    logger.info(
                        f"🔍 WebChat session updated, conversation length: {len(session.conversation_history)}"
                    )
                    return session
                
                # Use atomic operation for conversation update
                session = await self.session_manager.execute_session_operation(session_id, update_conversation_with_branch)
            
            # Handle streaming vs non-streaming
            if stream:
                # For now, just return non-streaming response
                # TODO: Implement proper streaming with branch support
                logger.warning(f"⚠️ Streaming mode doesn't support branch_id parameter yet, using current branch")
                return web.json_response(response_data)
            else:
                return web.json_response(response_data)
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            msg = f"Inference failed: {str(e)}"
            return web.json_response(
                {
                    "message": msg,
                    "error": {
                        "message": msg,
                        "type": "server_error",
                    }
                },
                status=500,
            )
    
    async def handle_enhanced_chat_completions(self, request: web.Request) -> web.Response:
        """Enhanced chat completions with better validation and error handling.
        
        Args:
            request: Web request with chat completion parameters
            
        Returns:
            JSON response with completion result using enhanced format
        """
        if not self.enhanced_features_available or not self.request_validator or not self.response_formatter:
            from oumi.webchat.api_responses import create_json_response
            return create_json_response(
                {"error": "Enhanced features not available"}, 
                status=501
            )
        
        from oumi.webchat.api_responses import create_json_response, ErrorType
        
        try:
            # Validate request
            chat_request = await self.request_validator.validate_chat_request(request)
            
            # Extract required fields
            messages = chat_request.messages
            session_id = chat_request.session_id
            temperature = chat_request.temperature
            max_tokens = chat_request.max_tokens
            stream = chat_request.stream
            
            # Get or create session
            session = await self.session_manager.get_or_create_session_safe(session_id, self.db)
            
            # Convert OpenAI format messages to Oumi conversation format
            oumi_messages = []
            
            # Add system prompt if provided
            if self.system_prompt:
                oumi_messages.append(
                    Message(role=Role.SYSTEM, content=self.system_prompt)
                )
            
            # Convert messages
            for msg in messages:
                role_mapping = {
                    "system": Role.SYSTEM,
                    "user": Role.USER,
                    "assistant": Role.ASSISTANT,
                }
                role = role_mapping.get(msg.get("role"), Role.USER)
                content = msg.get("content", "")
                oumi_messages.append(Message(role=role, content=content))
            
            # Get the latest user message for inference
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                response_data, status_code = self.response_formatter.error(
                    ErrorType.VALIDATION_ERROR,
                    "No user message found in request"
                )
                return create_json_response(response_data, status_code)
            
            latest_user_raw = user_messages[-1].get("content", "")
            latest_user_content = self._convert_client_content(latest_user_raw)
            latest_user_text = (
                self._content_items_to_text(latest_user_content)
                if isinstance(latest_user_content, list)
                else str(latest_user_content)
            )
            
            # Run inference
            from oumi.infer import infer
            
            results = infer(
                config=session.config,
                inputs=[latest_user_text],
                system_prompt=self.system_prompt,
                inference_engine=session.inference_engine,
            )
            
            if not results:
                response_data, status_code = self.response_formatter.error(
                    ErrorType.INTERNAL_ERROR,
                    "No response generated from inference engine"
                )
                return create_json_response(response_data, status_code)
            
            # Extract response content
            response_content = ""
            conversation = results[0]
            for message in conversation.messages:
                if message.role not in [Role.USER, Role.SYSTEM]:
                    if isinstance(message.content, str):
                        response_content = message.content
                        break
                    elif isinstance(message.content, list):
                        for item in message.content:
                            if hasattr(item, "content") and item.content:
                                response_content = str(item.content)
                                break
            
            if not response_content:
                response_content = str(conversation)
            
            # Format response in OpenAI format
            response_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": getattr(session.config.model, "model_name", model_name_fallback("session.config.model.model_name")),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(latest_user_text.split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(latest_user_text.split()) + len(response_content.split()),
                },
            }
            
            # Update WebChat session
            session.conversation_history.extend([
                {
                    "role": "user",
                    "content": latest_user_text,
                    "timestamp": time.time(),
                },
                {
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": time.time(),
                }
            ])
            
            # Update context usage
            self.session_manager.update_context_usage(session_id)
            
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
                        session.branch_manager.current_branch_id, 
                        name=session.branch_manager.current_branch_id
                    )
                    # Append the last two messages (user + assistant)
                    if len(session.conversation_history) >= 2:
                        last_two = session.conversation_history[-2:]
                        for m in last_two:
                            self.db.append_message_to_branch(
                                conv_id,
                                session.branch_manager.current_branch_id,
                                role=m.get("role", "user"),
                                content=str(m.get("content", "")),
                                created_at=float(m.get("timestamp", time.time())),
                            )
                    # Update session's current branch record
                    self.db.set_session_current_branch(
                        session_id, 
                        conv_id, 
                        session.branch_manager.current_branch_id
                    )
                    
                    # Graph dual-write
                    try:
                        # Pass exact DB path to GraphStore
                        GraphStore(self.db.db_path).add_edge_for_branch_tail(
                            conv_id, session.branch_manager.current_branch_id
                        )
                    except Exception as ge:
                        logger.warning(f"Graph dual-write failed: {ge}")
                except Exception as pe:
                    logger.warning(f"⚠️ Dual-write persistence failed: {pe}")
            
            return create_json_response(response_data)
            
        except Exception as e:
            logger.error(f"Enhanced chat completion error: {e}")
            response_data, status_code = self.response_formatter.internal_error(
                message=f"Chat completion failed: {str(e)}"
            )
            return create_json_response(response_data, status_code)
