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

"""WebSocket handler for real-time communication in Oumi WebChat."""

import asyncio
import json
import time
import uuid
from typing import Dict, Optional, Any
from datetime import datetime

from aiohttp import WSMsgType, web

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger
from oumi.webchat.core.session_manager import SessionManager
from oumi.webchat.chatgraph_migration.graph_store import GraphStore
from oumi.webchat.protocol import normalize_msg_type, extract_session_id, get_valid_message_types


class WebSocketHandler:
    """Manages WebSocket connections and message handling for webchat."""
    
    def __init__(
        self, 
        session_manager: SessionManager,
        system_prompt: Optional[str] = None,
        db = None
    ):
        """Initialize WebSocket handler.
        
        Args:
            session_manager: Session manager for managing WebChat sessions
            system_prompt: Optional system prompt to include in conversations
            db: Optional WebchatDB instance for persistence
        """
        self.session_manager = session_manager
        self.system_prompt = system_prompt
        self.db = db
    
    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time communication.
        
        Args:
            request: Web request containing the WebSocket connection
            
        Returns:
            WebSocketResponse for the connection
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Extract or generate session ID
        session_id = request.query.get("session_id", str(uuid.uuid4()))
        
        # Get or create session safely with proper locks
        session = await self.session_manager.get_or_create_session_safe(session_id, self.db)
        
        # Add WebSocket to session
        await session.add_websocket(ws)
        
        # Send initial session state
        await ws.send_str(
            json.dumps(
                {
                    "type": "session_init",
                    "session_id": session_id,
                    "conversation": session.serialize_conversation(),
                    "branches": session.get_enhanced_branch_info(self.db),
                    "current_branch": session.branch_manager.current_branch_id,
                    "persistence": {
                        "is_persistent": bool(self.db),
                        "is_hydrated_from_db": getattr(session, 'is_hydrated_from_db', False),
                        "current_conversation_id": getattr(session, 'current_conversation_id', None),
                    },
                    "model_info": {
                        "name": getattr(session.config.model, "model_name", "Unknown"),
                        "engine": str(session.config.engine),
                    },
                }
            )
        )
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_websocket_message(session, data, ws)
                    except json.JSONDecodeError:
                        await ws.send_str(
                            json.dumps(
                                {"type": "error", "message": "Invalid JSON format"}
                            )
                        )
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        
        except asyncio.CancelledError:
            pass
        finally:
            await session.remove_websocket(ws)
        
        return ws
    
    async def handle_websocket_message(
        self, session, data: Dict[str, Any], ws: web.WebSocketResponse
    ):
        """Handle individual WebSocket messages.
        
        Args:
            session: WebChatSession instance for this connection
            data: Message data from the WebSocket
            ws: WebSocketResponse to send responses to
        """
        msg_type = normalize_msg_type(data.get("type", ""))
        
        if msg_type == "ping":
            await ws.send_str(json.dumps({"type": "pong"}))
        
        elif msg_type == "chat":
            await self.handle_chat_message(session, data, ws)
        
        elif msg_type == "command":
            await self.handle_command_message(session, data, ws)
        
        elif msg_type == "get_branches":
            branches = session.get_enhanced_branch_info(self.db)
            current_branch = session.branch_manager.current_branch_id
            logger.debug(f"📋 DEBUG: Get branches WS request - current: '{current_branch}', available: {[b['id'] for b in branches]}")
            logger.debug(f"📋 DEBUG: Branch details: {[(b['id'], b['message_count'], b['created_at']) for b in branches]}")
            await ws.send_str(
                json.dumps(
                    {
                        "type": "branches_update",
                        "branches": branches,
                        "current_branch": current_branch,
                    }
                )
            )
        
        elif msg_type == "system_monitor":
            monitor_stats = session.system_monitor.get_stats()
            await ws.send_str(
                json.dumps(
                    {
                        "type": "system_update",
                        "data": {
                            "cpu_percent": monitor_stats.cpu_percent,
                            "ram_used_gb": monitor_stats.ram_used_gb,
                            "ram_total_gb": monitor_stats.ram_total_gb,
                            "ram_percent": monitor_stats.ram_percent,
                            "gpu_vram_used_gb": monitor_stats.gpu_vram_used_gb,
                            "gpu_vram_total_gb": monitor_stats.gpu_vram_total_gb,
                            "gpu_vram_percent": monitor_stats.gpu_vram_percent,
                            "context_used_tokens": monitor_stats.context_used_tokens,
                            "context_max_tokens": monitor_stats.context_max_tokens,
                            "context_percent": monitor_stats.context_percent,
                            "conversation_turns": monitor_stats.conversation_turns,
                        },
                    }
                )
            )
        
        else:
            valid_types = get_valid_message_types()
            await ws.send_str(
                json.dumps(
                    {
                        "type": "error", 
                        "message": f"Unknown message type: '{msg_type}' (expected: {valid_types})"
                    }
                )
            )
    
    async def handle_chat_message(
        self, session, data: Dict[str, Any], ws: web.WebSocketResponse
    ):
        """Handle regular chat messages.
        
        Args:
            session: WebChatSession instance for this connection
            data: Message data from the WebSocket
            ws: WebSocketResponse to send responses to
        """
        user_message = data.get("message", "")
        # Extract branch_id with consistent handling
        target_branch_id = data.get("branch_id")
        
        # Note: We don't use extract_session_id here since the session is already provided,
        # but we consistently accept branch_id as an optional parameter
        try:
            logger.debug(
                f"WS chat start: session={session.session_id}, current_branch={session.branch_manager.current_branch_id}, target_branch={target_branch_id}"
            )
        except Exception:
            pass

        async def _apply_and_sync(s):
            # If requested, switch to the target branch first
            if target_branch_id and target_branch_id != s.branch_manager.current_branch_id:
                try:
                    s.branch_manager.sync_conversation_history(s.conversation_history)
                    success, msg, branch = s.branch_manager.switch_branch(target_branch_id)
                    if success and branch:
                        s.conversation_history.clear()
                        s.conversation_history.extend(branch.conversation_history)
                        logger.debug(f"WS chat: switched to branch {target_branch_id}")
                except Exception as e:
                    logger.warning(f"WS branch switch failed: {e}")

            # Append user message
            s.conversation_history.append(
                {"role": "user", "content": user_message, "timestamp": time.time()}
            )

            # Sync the active branch immediately
            try:
                current_branch = s.branch_manager.get_current_branch()
                current_branch.conversation_history = s.conversation_history.copy()
                current_branch.last_active = datetime.now()
                logger.debug(
                    f"WS chat post-user: current_branch={s.branch_manager.current_branch_id}, conv_len={len(s.conversation_history)}, branch_len={len(current_branch.conversation_history)}"
                )
            except Exception as sync_err:
                logger.debug(f"Branch pre-sync after user message failed: {sync_err}")
            return s

        # Ensure branch switch + append occur under the session lock
        session = await self.session_manager.execute_session_operation(session.session_id, _apply_and_sync)
        
        # Broadcast user message to all clients
        await session.broadcast_to_websockets(
            {"type": "user_message", "content": user_message, "timestamp": time.time()}
        )
        
        # Generate AI response using inference engine
        try:
            # Initialize inference engine if not already done
            # Important: Only reset if None - preserve engines set by /swap commands
            if session.command_context.inference_engine is None:
                logger.debug("Initializing inference engine from server config")
                session.command_context.inference_engine = session.inference_engine
            else:
                # Log current engine info for debugging swap issues
                engine_info = getattr(
                    session.command_context.inference_engine, "model_name", "Unknown"
                )
                logger.debug(f"Using existing inference engine: {engine_info}")
                
                # Validate that the existing engine is still usable
                try:
                    # Check if engine has required methods (basic validation)
                    if not hasattr(
                        session.command_context.inference_engine, "generate_response"
                    ):
                        logger.warning(
                            "Swapped inference engine missing generate_response method, falling back to original"
                        )
                        session.command_context.inference_engine = session.inference_engine
                except Exception as e:
                    logger.warning(
                        f"Swapped inference engine validation failed: {e}, falling back to original"
                    )
                    session.command_context.inference_engine = session.inference_engine
            
            # Create conversation for inference
            # Convert conversation history to Oumi conversation format
            oumi_messages = []
            
            # Add system prompt if configured
            if self.system_prompt:
                oumi_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
            
            # Add conversation history
            for msg in session.conversation_history:
                role_map = {
                    "user": Role.USER,
                    "assistant": Role.ASSISTANT,
                    "system": Role.SYSTEM,
                }
                role = role_map.get(msg.get("role"), Role.USER)
                content = msg.get("content", "")
                
                if content:  # Skip empty messages
                    oumi_messages.append(Message(role=role, content=content))
            
            # Create conversation
            conversation = Conversation(messages=oumi_messages)
            
            # Send "thinking" indicator
            await session.broadcast_to_websockets(
                {"type": "assistant_thinking", "timestamp": time.time()}
            )
            
            # Generate response
            result = session.command_context.inference_engine.generate_response(conversation)
            
            # Extract response content
            response_content = ""
            if result and len(result.messages) > 0:
                last_message = result.messages[-1]
                if last_message.role != Role.USER:
                    if isinstance(last_message.content, str):
                        response_content = last_message.content
                    elif isinstance(last_message.content, list):
                        # Handle list content by joining text parts
                        text_parts = []
                        for item in last_message.content:
                            if hasattr(item, "content") and item.content:
                                text_parts.append(str(item.content))
                        response_content = " ".join(text_parts)
                    else:
                        response_content = str(last_message.content)
            
            # Fallback if no response generated
            if not response_content:
                response_content = "I'm sorry, I couldn't generate a response."
            
            # Add response to conversation history
            session.conversation_history.append(
                {
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": time.time(),
                }
            )

            # Sync the active branch's conversation after assistant reply as well
            try:
                current_branch = session.branch_manager.get_current_branch()
                current_branch.conversation_history = session.conversation_history.copy()
                current_branch.last_active = datetime.now()
                logger.debug(
                    f"WS chat post-assistant: current_branch={session.branch_manager.current_branch_id}, conv_len={len(session.conversation_history)}, branch_len={len(current_branch.conversation_history)}"
                )
            except Exception as sync_err:
                logger.debug(f"Branch post-sync after assistant message failed: {sync_err}")
            
            # Broadcast assistant response
            await session.broadcast_to_websockets(
                {
                    "type": "assistant_message",
                    "content": response_content,
                    "timestamp": time.time(),
                }
            )
            
            # Update context usage after successful message exchange
            self.session_manager.update_context_usage(session.session_id)
            
            # Dual-write persistence (best-effort)
            if self.db:
                try:
                    self.db.ensure_session(session.session_id)
                    conv_id = self.db.ensure_conversation(session.session_id)
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
                        session.session_id, 
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
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_response = f"Error: {str(e)}"
            
            # Add error to conversation history
            session.conversation_history.append(
                {
                    "role": "assistant",
                    "content": error_response,
                    "timestamp": time.time(),
                }
            )
            
            # Broadcast error response
            await session.broadcast_to_websockets(
                {
                    "type": "assistant_message",
                    "content": error_response,
                    "timestamp": time.time(),
                    "is_error": True,
                }
            )
            
            # Update context usage even after errors
            self.session_manager.update_context_usage(session.session_id)
    
    async def handle_command_message(
        self, session, data: Dict[str, Any], ws: web.WebSocketResponse
    ):
        """Handle command execution requests.
        
        Args:
            session: WebChatSession instance for this connection
            data: Message data from the WebSocket
            ws: WebSocketResponse to send responses to
        """
        command_str = data.get("command", "")
        
        try:
            # Parse command
            if session.command_parser.is_command(command_str):
                parsed_command = session.command_parser.parse_command(command_str)
                
                # Execute command
                result = session.command_router.handle_command(parsed_command)
                
                # Send result back
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "command_result",
                            "command": command_str,
                            "success": result.success,
                            "message": result.message,
                            "should_continue": result.should_continue,
                        }
                    )
                )
                
                # If command affected conversation state, broadcast update
                if parsed_command.command in ["clear", "delete", "switch", "branch", "regen", "edit"]:
                    await session.broadcast_to_websockets(
                        {
                            "type": "conversation_update",
                            "conversation": session.serialize_conversation(),
                            "branches": session.get_enhanced_branch_info(self.db),
                            "current_branch": session.branch_manager.current_branch_id,
                        }
                    )
                
                # Handle commands that require follow-up inference (e.g., regen)
                if result.success and result.should_continue and hasattr(result, 'user_input_override') and result.user_input_override:
                    logger.info(f"Command '{parsed_command.command}' requires follow-up inference with user_input_override")
                    logger.info(f"User input override: {result.user_input_override[:100]}...")
                    
                    # For regeneration commands, perform inference directly
                    is_regeneration = getattr(result, 'is_regeneration', False)
                    if is_regeneration:
                        try:
                            # Perform inference using the user_input_override (edited content)
                            await session._perform_regeneration_inference(result.user_input_override)
                        except Exception as e:
                            logger.error(f"❌ Error during regeneration inference: {e}")
                            await session.broadcast_to_websockets(
                                {"type": "error", "message": f"Regeneration failed: {str(e)}", "timestamp": time.time()}
                            )
            
            else:
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "command_result",
                            "success": False,
                            "message": f"Invalid command: {command_str}",
                        }
                    )
                )
        
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            await ws.send_str(
                json.dumps(
                    {
                        "type": "command_result",
                        "success": False,
                        "message": f"Command failed: {str(e)}",
                    }
                )
            )
