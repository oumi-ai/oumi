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

"""Extended Oumi server with WebSocket and interactive command support."""

import asyncio
import copy
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Optional, Set

from aiohttp import WSMsgType, web
from rich.console import Console

# Import new enhanced components
try:
    from .sse_handler import SSEHandler
    from .api_responses import (
        ResponseFormatter,
        RequestValidator, 
        create_json_response,
        handle_api_errors,
        ErrorType
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced API features not available - missing dependencies")
    ENHANCED_FEATURES_AVAILABLE = False

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.commands.command_context import CommandContext
from oumi.core.commands.command_parser import CommandParser
from oumi.core.commands.command_router import CommandRouter
from oumi.core.commands.conversation_branches import ConversationBranchManager
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.monitoring import SystemMonitor
from oumi.core.thinking import ThinkingProcessor
from oumi.server import OpenAICompatibleServer
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
        self.websockets: set[web.WebSocketResponse] = set()

        # Last activity timestamp for cleanup
        self.last_activity = time.time()

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

        message_str = json.dumps(message)
        closed_sockets = set()

        for ws in self.websockets:
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
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def serialize_conversation(self) -> list:
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


class OumiWebServer(OpenAICompatibleServer):
    """Extended Oumi server with WebSocket and interactive command support."""

    def __init__(
        self,
        config: InferenceConfig,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the web server.

        Args:
            config: Inference configuration.
            system_prompt: Optional system prompt.
        """
        # Initialize base properties without calling super().__init__()
        # to avoid blocking inference engine initialization
        self.config = config
        self.system_prompt = system_prompt
        self.inference_engine = None  # Defer initialization

        # Model info for /v1/models endpoint
        self.model_info = {
            "id": getattr(config.model, "model_name", "oumi-model"),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "oumi",
        }

        # Session management
        self.sessions: dict[str, WebChatSession] = {}
        self.session_cleanup_interval = 3600  # 1 hour
        self.max_idle_time = 1800  # 30 minutes

        # Enhanced components (optional)
        if ENHANCED_FEATURES_AVAILABLE:
            self.sse_handler = SSEHandler()
            self.request_validator = RequestValidator()
            self.response_formatter = ResponseFormatter()
        else:
            self.sse_handler = None
            self.request_validator = None
            self.response_formatter = None

        # Start cleanup task
        self._cleanup_task = None

    def get_inference_engine(self):
        """Lazy initialization of inference engine."""
        if self.inference_engine is None:
            from oumi.infer import get_engine

            logger.info("🔄 Initializing inference engine...")
            self.inference_engine = get_engine(self.config)
            logger.info("✅ Inference engine initialized")
        return self.inference_engine

    async def handle_cors_preflight(self, request: web.Request) -> web.Response:
        """Handle CORS preflight OPTIONS requests."""
        logger.debug(f"🔗 CORS preflight request for {request.path}")
        
        response = web.Response(
            status=204,  # No Content for OPTIONS
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, DELETE",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Session-ID, X-Requested-With",
                "Access-Control-Max-Age": "86400",  # Cache preflight for 24 hours
            }
        )
        return response

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        logger.debug("🔍 Health endpoint called!")
        
        if ENHANCED_FEATURES_AVAILABLE and self.response_formatter:
            return create_json_response(
                self.response_formatter.success({
                    "status": "healthy",
                    "server": "oumi-webchat",
                    "version": "1.0.0",
                    "features": ["websocket", "sse", "chat", "branches", "commands"],
                    "enhanced_features": True,
                    "timestamp": time.time()
                })
            )
        else:
            return web.json_response({
                "status": "healthy",
                "server": "oumi-webchat",
                "version": "1.0.0",
                "features": ["websocket", "chat", "branches", "commands"],
                "enhanced_features": False,
                "timestamp": time.time()
            })

    async def handle_models(self, request: web.Request) -> web.Response:
        """List available models endpoint with enhanced config metadata."""
        # Get session-specific model info if available
        session_id = request.query.get("session_id", "default")
        
        try:
            session = await self.get_or_create_session(session_id)
            
            # Check if session has a swapped model
            if (hasattr(session.command_context, 'config') and 
                session.command_context.config is not None):
                active_config = session.command_context.config
                logger.debug(f"🔄 Using session's swapped config for /v1/models")
            else:
                active_config = self.config
                logger.debug(f"🔄 Using server's initial config for /v1/models")
            
            # Extract complete model metadata from active config
            model_name = getattr(active_config.model, "model_name", "oumi-model")
            engine = str(active_config.engine) if active_config.engine else "NATIVE"
            context_length = getattr(active_config.model, "model_max_length", 4096)
            
            # Create enhanced model info with config metadata
            enhanced_model_info = {
                "id": model_name,
                "object": "model", 
                "created": int(time.time()),
                "owned_by": "oumi",
                # Add config metadata for frontend display
                "config_metadata": {
                    "model_name": model_name,
                    "engine": engine,
                    "context_length": context_length,
                    "display_name": model_name.split('/')[-1] if '/' in model_name else model_name,
                    "description": f"{model_name} ({engine} engine)",
                    "model_family": self._extract_model_family(model_name),
                    "is_active_config": True  # Flag to indicate this is the active config
                }
            }
            
            logger.debug(f"📋 Returning enhanced model info: {enhanced_model_info['config_metadata']['display_name']}")
            return web.json_response({"object": "list", "data": [enhanced_model_info]})
            
        except Exception as e:
            logger.error(f"Error getting enhanced model info: {e}")
            # Fallback to basic model info
            return web.json_response({"object": "list", "data": [self.model_info]})
    
    def _extract_model_family(self, model_name: str) -> str:
        """Extract model family from model name for UI categorization."""
        model_lower = model_name.lower()
        
        # Extract family based on common patterns
        if 'llama' in model_lower:
            if '3.1' in model_lower:
                return 'llama3_1'
            elif '3.2' in model_lower:
                return 'llama3_2' 
            elif '3.3' in model_lower:
                return 'llama3_3'
            elif '4' in model_lower:
                return 'llama4'
            else:
                return 'llama'
        elif 'qwen' in model_lower:
            if '2.5' in model_lower:
                return 'qwen2_5'
            elif '3' in model_lower:
                return 'qwen3'
            else:
                return 'qwen'
        elif 'gemma' in model_lower:
            return 'gemma3'
        elif 'phi' in model_lower:
            if '4' in model_lower:
                return 'phi4'
            else:
                return 'phi3'
        elif 'deepseek' in model_lower:
            return 'deepseek_r1'
        elif 'gpt' in model_lower:
            return 'gpt_oss'
        else:
            return 'unknown'

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """Handle chat completions requests in OpenAI format."""
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
        model = data.get("model", self.model_info["id"])
        temperature = data.get("temperature", 1.0)
        max_tokens = data.get("max_tokens", 100)
        stream = data.get("stream", False)
        session_id = data.get("session_id")  # WebChat session ID

        try:
            # Convert OpenAI format messages to Oumi conversation format
            oumi_messages = []

            # Add system prompt if provided
            if self.system_prompt:
                from oumi.core.types.conversation import Message, Role

                oumi_messages.append(
                    Message(role=Role.SYSTEM, content=self.system_prompt)
                )

            # Convert messages
            for msg in messages:
                from oumi.core.types.conversation import Message, Role

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
                return web.json_response(
                    {
                        "error": {
                            "message": "No user message found",
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )

            latest_user_content = user_messages[-1].get("content", "")

            # CRITICAL FIX: Use session-specific inference engine and config after model swaps
            # Run inference (lazy-loaded)
            from oumi.infer import infer
            
            # Determine which config and engine to use
            session_config = self.config  # Default to server config
            session_engine = self.get_inference_engine()  # Default to server engine
            
            # If we have a session_id, check if the session has a swapped model
            if session_id:
                session = await self.get_or_create_session(session_id)
                if (hasattr(session.command_context, 'config') and 
                    session.command_context.config is not None):
                    session_config = session.command_context.config
                    logger.debug(f"🔄 Using session's swapped config: {getattr(session_config.model, 'model_name', 'Unknown')}")
                
                if (hasattr(session.command_context, 'inference_engine') and 
                    session.command_context.inference_engine is not None):
                    session_engine = session.command_context.inference_engine
                    logger.debug(f"🔄 Using session's swapped inference engine")
                else:
                    logger.debug(f"🔄 Using server's default inference engine")
                    
                # CRITICAL: Use the proper oumi.infer function with Conversation format
                # Instead of using just the latest user content, pass the full conversation
                from oumi.core.types.conversation import Conversation, Message, Role
                
                # Build the full conversation with history
                conversation_messages = []
                
                # Add system prompt if provided
                if self.system_prompt:
                    conversation_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
                
                # Add conversation history
                if session.conversation_history:
                    for msg in session.conversation_history:
                        role_mapping = {
                            "system": Role.SYSTEM,
                            "user": Role.USER, 
                            "assistant": Role.ASSISTANT,
                        }
                        role = role_mapping.get(msg.get("role"), Role.USER)
                        conversation_messages.append(Message(role=role, content=msg.get("content", "")))
                    logger.debug(f"🧠 Added {len(session.conversation_history)} messages from conversation history")
                
                # Add the new user message
                conversation_messages.append(Message(role=Role.USER, content=latest_user_content))
                
                # Create the conversation object
                full_conversation = Conversation(messages=conversation_messages)
                
                logger.debug(f"🧠 Using full conversation with {len(conversation_messages)} total messages for context")
            else:
                # Fallback: create a simple conversation with just the latest message
                from oumi.core.types.conversation import Conversation, Message, Role
                conversation_messages = []
                
                # Add system prompt if provided
                if self.system_prompt:
                    conversation_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
                
                conversation_messages.append(Message(role=Role.USER, content=latest_user_content))
                full_conversation = Conversation(messages=conversation_messages)
                logger.debug(f"🧠 No session context, using single message conversation")

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
                import time
                start_time = time.time()
                
                # Use the same inference engine interface as oumi chat command
                # ALL engines implement: inference_engine.infer(input=[conversations], inference_config=config)
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
                logger.error(f"❌ Inference engine call failed: {e}")
                # Fallback to the unified infer() wrapper function
                try:
                    logger.info(f"🔄 Falling back to unified infer() function")
                    results = infer(
                        config=session_config,
                        inputs=[latest_user_content],  # Simple string input fallback
                        system_prompt=self.system_prompt,
                        inference_engine=session_engine,
                    )
                    
                    if results and len(results) > 0:
                        response_conversation = results[0]
                        for msg in reversed(response_conversation.messages):
                            if msg.role not in [Role.USER, Role.SYSTEM]:
                                response_content = str(msg.content)
                                break
                        else:
                            response_content = "No response generated"
                    else:
                        response_content = "No response generated"
                        
                    logger.debug(f"✅ Fallback infer() succeeded: {len(response_content)} chars")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback infer() also failed: {fallback_error}")
                    response_content = f"Inference failed: {str(e)}"

            # Check if we have a valid response
            if not response_content or response_content.startswith("Inference failed:"):
                return web.json_response(
                    {
                        "error": {
                            "message": response_content or "No response generated",
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
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(latest_user_content.split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(latest_user_content.split())
                    + len(response_content.split()),
                },
            }

            # Update WebChat session if session_id provided
            if session_id:
                logger.info(
                    f"🔍 DEBUG: Updating WebChat session {session_id} from OpenAI API"
                )
                session = await self.get_or_create_session(session_id)

                # Add user message to session conversation history
                session.conversation_history.append(
                    {
                        "role": "user",
                        "content": latest_user_content,
                        "timestamp": time.time(),
                    }
                )

                # Add assistant response to session conversation history
                session.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": time.time(),
                    }
                )

                # CRITICAL: Sync conversation to current branch after each exchange
                session.branch_manager.sync_conversation_history(session.conversation_history)
                logger.info(
                    f"🔄 DEBUG: Synced conversation to branch '{session.branch_manager.current_branch_id}' - {len(session.conversation_history)} messages"
                )

                # Update context usage
                self._update_session_context_usage(session)
                logger.info(
                    f"🔍 DEBUG: WebChat session updated, conversation length: {len(session.conversation_history)}"
                )

            # Handle streaming vs non-streaming
            if stream:
                # For now, just return non-streaming response
                # TODO: Implement proper streaming
                return web.json_response(response_data)
            else:
                return web.json_response(response_data)

        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return web.json_response(
                {
                    "error": {
                        "message": f"Inference failed: {str(e)}",
                        "type": "server_error",
                    }
                },
                status=500,
            )

    async def get_or_create_session(self, session_id: str) -> WebChatSession:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = WebChatSession(session_id, self.config)
            logger.debug(f"🆕 Created new webchat session: {session_id}")
        else:
            logger.debug(f"🔄 Using existing session: {session_id}")

        session = self.sessions[session_id]
        
        # CRITICAL FIX: Add session integrity validation
        if not hasattr(session, 'branch_manager') or session.branch_manager is None:
            logger.error(f"🚨 CRITICAL: Session {session_id} has corrupted branch_manager! Recreating...")
            session.branch_manager = ConversationBranchManager(session.conversation_history)
            # Re-sync main branch
            if "main" in session.branch_manager.branches:
                session.branch_manager.branches["main"].conversation_history = session.conversation_history
        
        session.update_activity()
        return session

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time communication."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        session_id = request.query.get("session_id", str(uuid.uuid4()))
        session = await self.get_or_create_session(session_id)

        await session.add_websocket(ws)

        # Send initial session state
        await ws.send_str(
            json.dumps(
                {
                    "type": "session_init",
                    "session_id": session_id,
                    "conversation": session.serialize_conversation(),
                    "branches": session.branch_manager.list_branches(),
                    "current_branch": session.branch_manager.current_branch_id,
                    "model_info": {
                        "name": getattr(self.config.model, "model_name", "Unknown"),
                        "engine": str(self.config.engine),
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
        self, session: WebChatSession, data: dict, ws: web.WebSocketResponse
    ):
        """Handle individual WebSocket messages."""
        msg_type = data.get("type")

        if msg_type == "ping":
            await ws.send_str(json.dumps({"type": "pong"}))

        elif msg_type == "chat_message":
            await self.handle_chat_message(session, data, ws)

        elif msg_type == "command":
            await self.handle_command_message(session, data, ws)

        elif msg_type == "get_branches":
            branches = session.branch_manager.list_branches()
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
            await ws.send_str(
                json.dumps(
                    {"type": "error", "message": f"Unknown message type: {msg_type}"}
                )
            )

    async def handle_chat_message(
        self, session: WebChatSession, data: dict, ws: web.WebSocketResponse
    ):
        """Handle regular chat messages."""
        user_message = data.get("message", "")

        # Add user message to conversation
        session.conversation_history.append(
            {"role": "user", "content": user_message, "timestamp": time.time()}
        )

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
                session.command_context.inference_engine = self.get_inference_engine()
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
                        session.command_context.inference_engine = (
                            self.get_inference_engine()
                        )
                except Exception as e:
                    logger.warning(
                        f"Swapped inference engine validation failed: {e}, falling back to original"
                    )
                    session.command_context.inference_engine = (
                        self.get_inference_engine()
                    )

            # Create conversation for inference
            from oumi.core.types.conversation import Conversation, Message, Role

            # Convert conversation history to Oumi conversation format
            oumi_messages = []

            # Add system prompt if configured
            if self.system_prompt:
                oumi_messages.append(
                    Message(role=Role.SYSTEM, content=self.system_prompt)
                )

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
            result = session.command_context.inference_engine.generate_response(
                conversation
            )

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

            # Broadcast assistant response
            await session.broadcast_to_websockets(
                {
                    "type": "assistant_message",
                    "content": response_content,
                    "timestamp": time.time(),
                }
            )

            # Update context usage after successful message exchange
            self._update_session_context_usage(session)

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
            self._update_session_context_usage(session)

    async def handle_command_message(
        self, session: WebChatSession, data: dict, ws: web.WebSocketResponse
    ):
        """Handle command execution requests."""
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
                if parsed_command.command in ["clear", "delete", "switch", "branch"]:
                    await session.broadcast_to_websockets(
                        {
                            "type": "conversation_update",
                            "conversation": session.serialize_conversation(),
                            "branches": session.branch_manager.list_branches(),
                            "current_branch": session.branch_manager.current_branch_id,
                        }
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

    async def handle_command_api(self, request: web.Request) -> web.Response:
        """Handle command execution via REST API."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        session_id = data.get("session_id", "default")
        command = data.get("command", "")
        args = data.get("args", [])

        session = await self.get_or_create_session(session_id)

        try:
            # Create a string buffer to capture console output
            import io

            from rich.console import Console

            # Create a temporary console that writes to string buffer
            string_buffer = io.StringIO()
            temp_console = Console(file=string_buffer, width=80)

            # Temporarily replace the session's console
            original_console = session.command_context.console
            session.command_context.console = temp_console

            try:
                # Execute command via command router
                from oumi.core.commands.command_parser import ParsedCommand

                parsed_command = ParsedCommand(
                    command=command,
                    args=args,
                    kwargs={},
                    raw_input=f"/{command}({','.join(args)})",
                )
                result = session.command_router.handle_command(parsed_command)

                # Capture the console output
                console_output = string_buffer.getvalue()

                # Combine command result message with console output
                full_message = result.message or ""
                if console_output.strip():
                    if full_message:
                        full_message += "\n\n" + console_output.strip()
                    else:
                        full_message = console_output.strip()

            finally:
                # Restore the original console
                session.command_context.console = original_console

            response_data = {
                "success": result.success,
                "message": full_message,
                "should_continue": result.should_continue,
            }

            # Add specific data for certain commands
            if command in ["branches", "list_branches"]:
                response_data["branches"] = session.branch_manager.list_branches()
                response_data["current_branch"] = (
                    session.branch_manager.current_branch_id
                )

            elif command == "show":
                response_data["conversation"] = session.serialize_conversation()
            
            elif command == "swap" and result.success:
                # CRITICAL FIX: Update server's model_info when model swap is successful
                # This ensures the /v1/models endpoint returns the new model information
                try:
                    if hasattr(session.command_context, 'config') and session.command_context.config:
                        new_config = session.command_context.config
                        # Update the server's model_info to reflect the swapped model
                        self.model_info = {
                            "id": getattr(new_config.model, "model_name", "oumi-model"),
                            "object": "model", 
                            "created": int(time.time()),
                            "owned_by": "oumi"
                        }
                        logger.info(f"✅ Updated server model_info after swap: {self.model_info['id']}")
                    else:
                        logger.warning("⚠️ Cannot update model_info: session config not available")
                except Exception as e:
                    logger.error(f"❌ Error updating model_info after swap: {e}")

            return web.json_response(response_data)

        except Exception as e:
            import traceback

            logger.error(f"API command execution error: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return web.json_response({"error": f"Command failed: {str(e)}"}, status=500)

    async def handle_branches_api(self, request: web.Request) -> web.Response:
        """Handle branch operations via REST API."""
        session_id = request.query.get("session_id", "default")
        logger.debug(f"🌐 DEBUG: Branch API called with session_id: '{session_id}'")
        session = await self.get_or_create_session(session_id)

        if request.method == "GET":
            # DEBUG: Check raw branch storage
            logger.debug(f"📋 DEBUG: GET branches request - session_id: '{session_id}'")
            logger.debug(f"📋 DEBUG: Session object ID: {id(session)}")
            logger.debug(f"📋 DEBUG: Branch manager object ID: {id(session.branch_manager)}")
            logger.debug(f"📋 DEBUG: Raw branches dict: {list(session.branch_manager.branches.keys())}")
            logger.debug(f"📋 DEBUG: Branch counter: {session.branch_manager._branch_counter}")
            
            branches = session.branch_manager.list_branches()
            current_branch = session.branch_manager.current_branch_id
            logger.debug(f"📋 DEBUG: Get branches HTTP request - current: '{current_branch}', available: {[b['id'] for b in branches]}")
            logger.debug(f"📋 DEBUG: Branch details: {[(b['id'], b['message_count'], b['created_at']) for b in branches]}")
            return web.json_response(
                {
                    "branches": branches,
                    "current_branch": current_branch,
                }
            )

        elif request.method == "POST":
            try:
                data = await request.json()
                action = data.get("action")
                
                # Check if session_id is also in POST data (for consistency)
                post_session_id = data.get("session_id")
                if post_session_id and post_session_id != session_id:
                    logger.warning(f"⚠️  Session ID mismatch: query='{session_id}', post='{post_session_id}' - using POST value")
                    session_id = post_session_id
                    session = await self.get_or_create_session(session_id)

                if action == "switch":
                    branch_id = data.get("branch_id")
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

                    return web.json_response(
                        {
                            "success": success,
                            "message": message,
                            "conversation": session.serialize_conversation(),
                            "current_branch": session.branch_manager.current_branch_id,
                        }
                    )

                elif action == "create":
                    from_branch = data.get(
                        "from_branch", session.branch_manager.current_branch_id
                    )
                    name = data.get("name")
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

                    # CRITICAL FIX: Sync main branch conversation before creating new branch
                    if from_branch == "main":
                        logger.debug(f"🔄 DEBUG: Syncing main branch conversation history before branch creation")
                        session.branch_manager.sync_conversation_history(session.conversation_history)
                        # Re-check source branch after sync
                        main_branch = session.branch_manager.branches.get("main")
                        if main_branch:
                            logger.debug(f"🔄 DEBUG: Main branch conversation after sync: {len(main_branch.conversation_history)} messages")

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

                elif action == "delete":
                    branch_id = data.get("branch_id")
                    logger.debug(f"🗑️  DEBUG: Branch delete requested - branch_id: '{branch_id}'")
                    logger.debug(f"🗑️  DEBUG: Available branches before delete: {[b['id'] for b in session.branch_manager.list_branches()]}")
                    success, message = session.branch_manager.delete_branch(branch_id)
                    logger.debug(f"🗑️  DEBUG: Branch delete result - success: {success}, message: '{message}'")
                    logger.debug(f"🗑️  DEBUG: Available branches after delete: {[b['id'] for b in session.branch_manager.list_branches()]}")

                    return web.json_response(
                        {
                            "success": success,
                            "message": message,
                            "branches": session.branch_manager.list_branches(),
                            "current_branch": session.branch_manager.current_branch_id,
                        }
                    )

                else:
                    return web.json_response(
                        {"error": f"Unknown action: {action}"}, status=400
                    )

            except Exception as e:
                logger.error(f"Branch API error: {e}")
                return web.json_response({"error": str(e)}, status=500)

    async def handle_sync_conversation_api(self, request: web.Request) -> web.Response:
        """Handle syncing conversation from frontend to backend session."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        session_id = data.get("session_id", "default")
        conversation = data.get("conversation", [])

        session = await self.get_or_create_session(session_id)

        # Update the session's conversation history
        session.conversation_history.clear()
        session.conversation_history.extend(conversation)
        session.update_activity()

        return web.json_response({"success": True})

    async def handle_get_conversation_api(self, request: web.Request) -> web.Response:
        """Handle getting current conversation from backend session."""
        session_id = request.query.get("session_id", "default")
        session = await self.get_or_create_session(session_id)

        return web.json_response({"conversation": session.serialize_conversation()})

    async def handle_system_stats_api(self, request: web.Request) -> web.Response:
        """Handle getting system stats from backend session."""
        session_id = request.query.get("session_id", "default")
        session = await self.get_or_create_session(session_id)

        # Update context usage based on current conversation
        self._update_session_context_usage(session)

        # Get stats from SystemMonitor
        stats = session.system_monitor.get_stats()

        response_data = {
            "cpu_percent": stats.cpu_percent,
            "ram_used_gb": stats.ram_used_gb,
            "ram_total_gb": stats.ram_total_gb,
            "ram_percent": stats.ram_percent,
            "gpu_vram_used_gb": stats.gpu_vram_used_gb,
            "gpu_vram_total_gb": stats.gpu_vram_total_gb,
            "gpu_vram_percent": stats.gpu_vram_percent,
            "context_used_tokens": stats.context_used_tokens,
            "context_max_tokens": stats.context_max_tokens,
            "context_percent": stats.context_percent,
            "conversation_turns": stats.conversation_turns,
        }

        logger.debug(f"System stats for session {session_id}: {stats.context_used_tokens}/{stats.context_max_tokens} tokens, {stats.conversation_turns} turns")

        return web.json_response(response_data)

    async def handle_get_configs_api(self, request: web.Request) -> web.Response:
        """Handle getting available inference configuration files."""
        try:
            configs = self._scan_inference_config_files()
            return web.json_response({"configs": configs})
        except Exception as e:
            logger.error(f"Error getting configs: {e}")
            return web.json_response(
                {"error": "Failed to scan configuration files"}, status=500
            )

    def _scan_inference_config_files(self):
        """Scan the configs directory for inference YAML files (*_infer.yaml)."""
        import os
        from pathlib import Path
        import yaml

        configs = []
        
        # Get the configs directory relative to this file
        current_dir = Path(__file__).parent.parent.parent.parent  # Go up to oumi root
        configs_dir = current_dir / "configs" / "recipes"
        
        if not configs_dir.exists():
            logger.warning(f"Configs directory not found: {configs_dir}")
            return configs
            
        logger.debug(f"📁 Scanning configs directory: {configs_dir}")
        
        # Walk through all subdirectories looking for *_infer.yaml files
        for root, dirs, files in os.walk(configs_dir):
            for file in files:
                if file.endswith("_infer.yaml"):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(configs_dir)
                    
                    try:
                        # Read the YAML to extract model information
                        with open(file_path, 'r') as f:
                            config_data = yaml.safe_load(f)
                        
                        # Extract key information
                        model_name = config_data.get('model', {}).get('model_name', 'Unknown')
                        engine = config_data.get('engine', 'UNKNOWN')
                        context_length = config_data.get('model', {}).get('model_max_length', 4096)
                        
                        # Determine category based on path or model name
                        path_parts = str(relative_path).split('/')
                        model_family = path_parts[0] if path_parts else 'unknown'
                        
                        # Create a display name from the file path
                        display_name = str(relative_path).replace('/', ' > ').replace('_infer.yaml', '')
                        
                        config_info = {
                            'id': str(relative_path),
                            'config_path': str(file_path),
                            'relative_path': str(relative_path),
                            'display_name': display_name,
                            'model_name': model_name,
                            'engine': engine,
                            'context_length': context_length,
                            'model_family': model_family,
                            'filename': file,
                        }
                        
                        configs.append(config_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse config {file_path}: {e}")
                        continue
        
        logger.debug(f"📋 Found {len(configs)} inference configurations")
        return sorted(configs, key=lambda x: (x['model_family'], x['display_name']))

    def _update_session_context_usage(self, session):
        """Update SystemMonitor with current conversation context usage."""
        try:
            # Use the existing ContextWindowManager for proper token estimation
            context_manager = session.command_context.context_window_manager
            total_tokens = 0

            logger.info(
                f"🔍 DEBUG: Updating context usage for session {session.session_id}"
            )
            logger.info(
                f"🔍 DEBUG: Conversation history length: {len(session.conversation_history)}"
            )

            for i, msg in enumerate(session.conversation_history):
                content = msg.get("content", "")
                if content:
                    msg_tokens = context_manager.estimate_tokens(content)
                    total_tokens += msg_tokens
                    logger.info(
                        f"🔍 DEBUG: Message {i}: {msg_tokens} tokens, content preview: {content[:50]}..."
                    )

            logger.debug(f"🔍 DEBUG: Total tokens calculated: {total_tokens}")
            logger.debug(
                f"🔍 DEBUG: Max context tokens: {session.system_monitor.max_context_tokens}"
            )

            session.system_monitor.update_context_usage(total_tokens)
            session.system_monitor.update_conversation_turns(
                len(session.conversation_history) // 2
            )

            # Verify the update worked
            stats = session.system_monitor.get_stats()
            logger.info(
                f"🔍 DEBUG: SystemMonitor stats after update - context_used: {stats.context_used_tokens}, context_max: {stats.context_max_tokens}, percent: {stats.context_percent}"
            )

        except Exception as e:
            logger.warning(f"Failed to update context usage: {e}")
            import traceback

            logger.warning(f"Full traceback: {traceback.format_exc()}")

    async def handle_sse_chat(self, request: web.Request) -> web.StreamResponse:
        """Handle Server-Sent Events for streaming chat responses."""
        if not ENHANCED_FEATURES_AVAILABLE or not self.sse_handler:
            return web.json_response(
                {"error": "SSE features not available"}, 
                status=501
            )
        return await self.sse_handler.handle_sse_chat(request)

    async def handle_sse_events(self, request: web.Request) -> web.StreamResponse:
        """Handle general Server-Sent Events endpoint."""
        if not ENHANCED_FEATURES_AVAILABLE or not self.sse_handler:
            return web.json_response(
                {"error": "SSE features not available"}, 
                status=501
            )
        return await self.sse_handler.handle_sse_events(request)

    async def handle_enhanced_chat_completions(self, request: web.Request) -> web.Response:
        """Enhanced chat completions with better validation and error handling."""
        if not ENHANCED_FEATURES_AVAILABLE or not self.request_validator:
            return web.json_response(
                {"error": "Enhanced features not available"}, 
                status=501
            )
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
            session = await self.get_or_create_session(session_id)

            # Convert OpenAI format messages to Oumi conversation format
            oumi_messages = []

            # Add system prompt if provided
            if self.system_prompt:
                from oumi.core.types.conversation import Message, Role
                oumi_messages.append(
                    Message(role=Role.SYSTEM, content=self.system_prompt)
                )

            # Convert messages
            for msg in messages:
                from oumi.core.types.conversation import Message, Role
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

            latest_user_content = user_messages[-1].get("content", "")

            # Run inference (lazy-loaded)
            from oumi.infer import infer

            results = infer(
                config=self.config,
                inputs=[latest_user_content],
                system_prompt=self.system_prompt,
                inference_engine=self.get_inference_engine(),
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
                from oumi.core.types.conversation import Role
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
                "model": getattr(self.config.model, "model_name", "oumi-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(latest_user_content.split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(latest_user_content.split()) + len(response_content.split()),
                },
            }

            # Update WebChat session
            session.conversation_history.extend([
                {
                    "role": "user",
                    "content": latest_user_content,
                    "timestamp": time.time(),
                },
                {
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": time.time(),
                }
            ])

            # Update context usage
            self._update_session_context_usage(session)

            return create_json_response(response_data)

        except Exception as e:
            logger.error(f"Enhanced chat completion error: {e}")
            response_data, status_code = self.response_formatter.internal_error(
                message=f"Chat completion failed: {str(e)}"
            )
            return create_json_response(response_data, status_code)

    async def cleanup_sessions(self):
        """Clean up inactive sessions."""
        while True:
            try:
                await asyncio.sleep(self.session_cleanup_interval)

                current_time = time.time()
                expired_sessions = []

                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > self.max_idle_time:
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    logger.debug(f"Cleaning up expired session: {session_id}")
                    session = self.sessions.pop(session_id)

                    # Close any remaining WebSocket connections
                    for ws in session.websockets.copy():
                        await ws.close()

            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    def create_app(self) -> web.Application:
        """Create and configure the aiohttp application with WebSocket support."""
        # Create base app without calling super() to avoid CORS middleware conflicts
        app = web.Application()

        # Test with simple function-based handler to debug binding issue
        async def simple_health(request):
            logger.debug("🔍 Simple health handler called!")
            return web.json_response({"status": "simple_ok"})

        # Add base routes from OpenAICompatibleServer manually
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        
        # Enhanced endpoints (conditional)
        if ENHANCED_FEATURES_AVAILABLE:
            app.router.add_post("/v1/chat/completions/enhanced", self.handle_enhanced_chat_completions)
            
            # Server-Sent Events endpoints
            app.router.add_get("/v1/oumi/sse/chat", self.handle_sse_chat)
            app.router.add_get("/v1/oumi/sse/events", self.handle_sse_events)

        # Add new WebChat endpoints
        app.router.add_get("/v1/oumi/ws", self.handle_websocket)
        app.router.add_post("/v1/oumi/command", self.handle_command_api)
        app.router.add_get("/v1/oumi/branches", self.handle_branches_api)
        app.router.add_post("/v1/oumi/branches", self.handle_branches_api)
        app.router.add_post(
            "/v1/oumi/sync_conversation", self.handle_sync_conversation_api
        )
        app.router.add_get("/v1/oumi/conversation", self.handle_get_conversation_api)
        app.router.add_get("/v1/oumi/system_stats", self.handle_system_stats_api)
        app.router.add_get("/v1/oumi/configs", self.handle_get_configs_api)

        # Add CORS OPTIONS handlers for all endpoints that need preflight
        cors_endpoints = [
            "/health",  # Add health endpoint for connection testing
            "/v1/chat/completions",
            "/v1/models",  # Add models endpoint for CORS
            "/v1/oumi/branches", 
            "/v1/oumi/conversation",
            "/v1/oumi/command",
            "/v1/oumi/sync_conversation",
            "/v1/oumi/system_stats",  # Add system stats endpoint
            "/v1/oumi/configs",  # Add configs endpoint
        ]
        
        if ENHANCED_FEATURES_AVAILABLE:
            cors_endpoints.extend([
                "/v1/chat/completions/enhanced",
                "/v1/oumi/sse/chat",
                "/v1/oumi/sse/events"
            ])
            
        for endpoint in cors_endpoints:
            app.router.add_options(endpoint, self.handle_cors_preflight)

        # Add enhanced request logging middleware
        @web.middleware
        async def logging_middleware(request, handler):
            start_time = time.time()
            client_ip = request.headers.get('X-Forwarded-For', request.remote)
            
            logger.debug(f"🌐 {request.method} {request.path} from {client_ip}")
            
            # Log request headers for debugging CORS/connection issues
            if request.headers.get('Origin'):
                logger.debug(f"🔍 Origin: {request.headers.get('Origin')}")
            if request.headers.get('User-Agent'):
                logger.debug(f"🔍 User-Agent: {request.headers.get('User-Agent')}")
            
            # Note: Don't log request body in middleware as it consumes the stream
            # and prevents handlers from reading it. Body logging can be added
            # in individual handlers if needed.
            
            try:
                response = await handler(request)
                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"✅ {request.method} {request.path} -> {response.status} ({elapsed:.1f}ms)")
                return response
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                logger.error(f"❌ {request.method} {request.path} -> ERROR: {e} ({elapsed:.1f}ms)")
                raise

        app.middlewares.append(logging_middleware)
        
        # Add CORS middleware to ensure all responses include CORS headers
        @web.middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            
            # Add CORS headers to all responses (not just preflight OPTIONS)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Session-ID, X-Requested-With"
            
            return response
            
        app.middlewares.append(cors_middleware)

        return app

    async def start_background_tasks(self, app: web.Application):
        """Start background tasks."""
        self._cleanup_task = asyncio.create_task(self.cleanup_sessions())

    async def cleanup_background_tasks(self, app: web.Application):
        """Clean up background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


def run_webchat_server(
    config: InferenceConfig,
    host: str = "0.0.0.0",
    port: int = 9000,
    system_prompt: Optional[str] = None,
) -> None:
    """Run the Oumi WebChat server."""
    server = OumiWebServer(config, system_prompt)
    app = server.create_app()

    # Temporarily disable startup/cleanup handlers to debug hanging issue
    # app.on_startup.append(server.start_background_tasks)
    # app.on_cleanup.append(server.cleanup_background_tasks)

    logger.info("🚀 Starting Oumi WebChat server")
    logger.info(f"📍 Server URL: http://{host}:{port}")
    logger.info("🌐 Real-time endpoints:")
    logger.info(f"   • WebSocket: ws://{host}:{port}/v1/oumi/ws")
    if ENHANCED_FEATURES_AVAILABLE:
        logger.info(f"   • SSE Chat: http://{host}:{port}/v1/oumi/sse/chat")
        logger.info(f"   • SSE Events: http://{host}:{port}/v1/oumi/sse/events")
    logger.info("🔗 API endpoints:")
    logger.info(f"   • Chat: http://{host}:{port}/v1/chat/completions")
    if ENHANCED_FEATURES_AVAILABLE:
        logger.info(f"   • Enhanced Chat: http://{host}:{port}/v1/chat/completions/enhanced")
    logger.info(f"   • Commands: http://{host}:{port}/v1/oumi/command")
    logger.info(f"   • Branches: http://{host}:{port}/v1/oumi/branches")
    logger.info(f"   • System Stats: http://{host}:{port}/v1/oumi/system_stats")
    if ENHANCED_FEATURES_AVAILABLE:
        logger.info("✨ Enhanced features enabled (SSE, validation, structured responses)")
    else:
        logger.info("ℹ️  Enhanced features disabled (missing dependencies)")
    logger.info("🛑 Press Ctrl+C to stop")

    # Debug: Check routes before starting
    logger.info(f"🔧 Debug: App has {len(list(app.router.routes()))} routes:")
    for route in app.router.routes():
        logger.info(f"   Route: {route.method} {route.resource.canonical}")

    try:
        logger.info("🔧 Starting web.run_app with proper threading setup...")
        # Use standard web.run_app - the issue was parameters, not the method
        web.run_app(
            app,
            host=host,
            port=port,
            handle_signals=False,  # Safe for threads
            print=lambda *args: logger.info(
                f"aiohttp: {' '.join(map(str, args))}"
            ),  # Custom print function
        )
        logger.info("🔧 web.run_app returned")
    except Exception as e:
        logger.error(f"❌ Error in web.run_app: {e}")
        import traceback

        traceback.print_exc()
        raise
