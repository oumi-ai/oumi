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
import json
import time
import uuid
from typing import Dict, Optional, Set

from aiohttp import web, WSMsgType
from rich.console import Console

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.commands.command_context import CommandContext
from oumi.core.commands.command_parser import CommandParser
from oumi.core.commands.command_router import CommandRouter
from oumi.core.commands.conversation_branches import ConversationBranchManager
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.input.enhanced_input import EnhancedInput
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
                result.append({
                    'role': msg.get('role', 'unknown'),
                    'content': msg.get('content', ''),
                    'timestamp': msg.get('timestamp', time.time())
                })
            else:
                # Handle other message formats if needed
                result.append({
                    'role': 'unknown',
                    'content': str(msg),
                    'timestamp': time.time()
                })
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
        self.sessions: Dict[str, WebChatSession] = {}
        self.session_cleanup_interval = 3600  # 1 hour
        self.max_idle_time = 1800  # 30 minutes

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

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        logger.info("🔍 Health endpoint called!")
        return web.json_response({"status": "ok"})

    async def handle_models(self, request: web.Request) -> web.Response:
        """List available models endpoint."""
        return web.json_response({"object": "list", "data": [self.model_info]})

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

            # Run inference (lazy-loaded)
            from oumi.infer import infer
            results = infer(
                config=self.config,
                inputs=[latest_user_content],
                system_prompt=self.system_prompt,
                inference_engine=self.get_inference_engine(),  # Lazy initialization
            )

            if not results:
                return web.json_response(
                    {
                        "error": {
                            "message": "No response generated",
                            "type": "server_error",
                        }
                    },
                    status=500,
                )

            # Extract response content
            response_content = ""
            conversation = results[0]  # Take first result
            for message in conversation.messages:
                from oumi.core.types.conversation import Role
                # Skip user messages and system messages, only get assistant responses
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
                logger.info(f"🔍 DEBUG: Updating WebChat session {session_id} from OpenAI API")
                session = await self.get_or_create_session(session_id)

                # Add user message to session conversation history
                session.conversation_history.append({
                    'role': 'user',
                    'content': latest_user_content,
                    'timestamp': time.time()
                })

                # Add assistant response to session conversation history
                session.conversation_history.append({
                    'role': 'assistant',
                    'content': response_content,
                    'timestamp': time.time()
                })

                # Update context usage
                self._update_session_context_usage(session)
                logger.info(f"🔍 DEBUG: WebChat session updated, conversation length: {len(session.conversation_history)}")

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
            logger.info(f"Created new webchat session: {session_id}")

        session = self.sessions[session_id]
        session.update_activity()
        return session

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time communication."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        session_id = request.query.get('session_id', str(uuid.uuid4()))
        session = await self.get_or_create_session(session_id)

        await session.add_websocket(ws)

        # Send initial session state
        await ws.send_str(json.dumps({
            'type': 'session_init',
            'session_id': session_id,
            'conversation': session.serialize_conversation(),
            'branches': session.branch_manager.list_branches(),
            'current_branch': session.branch_manager.current_branch_id,
            'model_info': {
                'name': getattr(self.config.model, 'model_name', 'Unknown'),
                'engine': str(self.config.engine)
            }
        }))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_websocket_message(session, data, ws)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON format'
                        }))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')

        except asyncio.CancelledError:
            pass
        finally:
            await session.remove_websocket(ws)

        return ws

    async def handle_websocket_message(
        self,
        session: WebChatSession,
        data: dict,
        ws: web.WebSocketResponse
    ):
        """Handle individual WebSocket messages."""
        msg_type = data.get('type')

        if msg_type == 'ping':
            await ws.send_str(json.dumps({'type': 'pong'}))

        elif msg_type == 'chat_message':
            await self.handle_chat_message(session, data, ws)

        elif msg_type == 'command':
            await self.handle_command_message(session, data, ws)

        elif msg_type == 'get_branches':
            await ws.send_str(json.dumps({
                'type': 'branches_update',
                'branches': session.branch_manager.list_branches(),
                'current_branch': session.branch_manager.current_branch_id
            }))

        elif msg_type == 'system_monitor':
            monitor_stats = session.system_monitor.get_stats()
            await ws.send_str(json.dumps({
                'type': 'system_update',
                'data': {
                    'cpu_percent': monitor_stats.cpu_percent,
                    'ram_used_gb': monitor_stats.ram_used_gb,
                    'ram_total_gb': monitor_stats.ram_total_gb,
                    'ram_percent': monitor_stats.ram_percent,
                    'gpu_vram_used_gb': monitor_stats.gpu_vram_used_gb,
                    'gpu_vram_total_gb': monitor_stats.gpu_vram_total_gb,
                    'gpu_vram_percent': monitor_stats.gpu_vram_percent,
                    'context_used_tokens': monitor_stats.context_used_tokens,
                    'context_max_tokens': monitor_stats.context_max_tokens,
                    'context_percent': monitor_stats.context_percent,
                    'conversation_turns': monitor_stats.conversation_turns
                }
            }))

        else:
            await ws.send_str(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {msg_type}'
            }))

    async def handle_chat_message(
        self,
        session: WebChatSession,
        data: dict,
        ws: web.WebSocketResponse
    ):
        """Handle regular chat messages."""
        user_message = data.get('message', '')

        # Add user message to conversation
        session.conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': time.time()
        })

        # Broadcast user message to all clients
        await session.broadcast_to_websockets({
            'type': 'user_message',
            'content': user_message,
            'timestamp': time.time()
        })

        # Generate AI response using inference engine
        try:
            # Initialize inference engine if not already done
            # Important: Only reset if None - preserve engines set by /swap commands
            if session.command_context.inference_engine is None:
                logger.debug("Initializing inference engine from server config")
                session.command_context.inference_engine = self.get_inference_engine()
            else:
                # Log current engine info for debugging swap issues
                engine_info = getattr(session.command_context.inference_engine, 'model_name', 'Unknown')
                logger.debug(f"Using existing inference engine: {engine_info}")

                # Validate that the existing engine is still usable
                try:
                    # Check if engine has required methods (basic validation)
                    if not hasattr(session.command_context.inference_engine, 'generate_response'):
                        logger.warning("Swapped inference engine missing generate_response method, falling back to original")
                        session.command_context.inference_engine = self.get_inference_engine()
                except Exception as e:
                    logger.warning(f"Swapped inference engine validation failed: {e}, falling back to original")
                    session.command_context.inference_engine = self.get_inference_engine()

            # Create conversation for inference
            from oumi.core.types.conversation import Conversation, Message, Role

            # Convert conversation history to Oumi conversation format
            oumi_messages = []

            # Add system prompt if configured
            if self.system_prompt:
                oumi_messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))

            # Add conversation history
            for msg in session.conversation_history:
                role_map = {
                    'user': Role.USER,
                    'assistant': Role.ASSISTANT,
                    'system': Role.SYSTEM
                }
                role = role_map.get(msg.get('role'), Role.USER)
                content = msg.get('content', '')

                if content:  # Skip empty messages
                    oumi_messages.append(Message(role=role, content=content))

            # Create conversation
            conversation = Conversation(messages=oumi_messages)

            # Send "thinking" indicator
            await session.broadcast_to_websockets({
                'type': 'assistant_thinking',
                'timestamp': time.time()
            })

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
                            if hasattr(item, 'content') and item.content:
                                text_parts.append(str(item.content))
                        response_content = " ".join(text_parts)
                    else:
                        response_content = str(last_message.content)

            # Fallback if no response generated
            if not response_content:
                response_content = "I'm sorry, I couldn't generate a response."

            # Add response to conversation history
            session.conversation_history.append({
                'role': 'assistant',
                'content': response_content,
                'timestamp': time.time()
            })

            # Broadcast assistant response
            await session.broadcast_to_websockets({
                'type': 'assistant_message',
                'content': response_content,
                'timestamp': time.time()
            })

            # Update context usage after successful message exchange
            self._update_session_context_usage(session)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_response = f"Error: {str(e)}"

            # Add error to conversation history
            session.conversation_history.append({
                'role': 'assistant',
                'content': error_response,
                'timestamp': time.time()
            })

            # Broadcast error response
            await session.broadcast_to_websockets({
                'type': 'assistant_message',
                'content': error_response,
                'timestamp': time.time(),
                'is_error': True
            })

            # Update context usage even after errors
            self._update_session_context_usage(session)

    async def handle_command_message(
        self,
        session: WebChatSession,
        data: dict,
        ws: web.WebSocketResponse
    ):
        """Handle command execution requests."""
        command_str = data.get('command', '')

        try:
            # Parse command
            if session.command_parser.is_command(command_str):
                parsed_command = session.command_parser.parse_command(command_str)

                # Execute command
                result = session.command_router.handle_command(parsed_command)

                # Send result back
                await ws.send_str(json.dumps({
                    'type': 'command_result',
                    'command': command_str,
                    'success': result.success,
                    'message': result.message,
                    'should_continue': result.should_continue
                }))

                # If command affected conversation state, broadcast update
                if parsed_command.command in ['clear', 'delete', 'switch', 'branch']:
                    await session.broadcast_to_websockets({
                        'type': 'conversation_update',
                        'conversation': session.serialize_conversation(),
                        'branches': session.branch_manager.list_branches(),
                        'current_branch': session.branch_manager.current_branch_id
                    })

            else:
                await ws.send_str(json.dumps({
                    'type': 'command_result',
                    'success': False,
                    'message': f'Invalid command: {command_str}'
                }))

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            await ws.send_str(json.dumps({
                'type': 'command_result',
                'success': False,
                'message': f'Command failed: {str(e)}'
            }))

    async def handle_command_api(self, request: web.Request) -> web.Response:
        """Handle command execution via REST API."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {'error': 'Invalid JSON'}, status=400
            )

        session_id = data.get('session_id', 'default')
        command = data.get('command', '')
        args = data.get('args', [])

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
                    raw_input=f"/{command}({','.join(args)})"
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
                'success': result.success,
                'message': full_message,
                'should_continue': result.should_continue
            }

            # Add specific data for certain commands
            if command in ['branches', 'list_branches']:
                response_data['branches'] = session.branch_manager.list_branches()
                response_data['current_branch'] = session.branch_manager.current_branch_id

            elif command == 'show':
                response_data['conversation'] = session.serialize_conversation()

            return web.json_response(response_data)

        except Exception as e:
            import traceback
            logger.error(f"API command execution error: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return web.json_response(
                {'error': f'Command failed: {str(e)}'}, status=500
            )

    async def handle_branches_api(self, request: web.Request) -> web.Response:
        """Handle branch operations via REST API."""
        session_id = request.query.get('session_id', 'default')
        session = await self.get_or_create_session(session_id)

        if request.method == 'GET':
            return web.json_response({
                'branches': session.branch_manager.list_branches(),
                'current_branch': session.branch_manager.current_branch_id
            })

        elif request.method == 'POST':
            try:
                data = await request.json()
                action = data.get('action')

                if action == 'switch':
                    branch_id = data.get('branch_id')
                    success, message, branch = session.branch_manager.switch_branch(branch_id)

                    if success and branch:
                        # Update conversation history
                        session.conversation_history.clear()
                        session.conversation_history.extend(branch.conversation_history)

                    return web.json_response({
                        'success': success,
                        'message': message,
                        'conversation': session.serialize_conversation(),
                        'current_branch': session.branch_manager.current_branch_id
                    })

                elif action == 'create':
                    from_branch = data.get('from_branch', session.branch_manager.current_branch_id)
                    name = data.get('name')

                    success, message, new_branch = session.branch_manager.create_branch(
                        from_branch_id=from_branch, name=name
                    )

                    return web.json_response({
                        'success': success,
                        'message': message,
                        'branches': session.branch_manager.list_branches()
                    })

                elif action == 'delete':
                    branch_id = data.get('branch_id')
                    success, message = session.branch_manager.delete_branch(branch_id)

                    return web.json_response({
                        'success': success,
                        'message': message,
                        'branches': session.branch_manager.list_branches(),
                        'current_branch': session.branch_manager.current_branch_id
                    })

                else:
                    return web.json_response({'error': f'Unknown action: {action}'}, status=400)

            except Exception as e:
                logger.error(f"Branch API error: {e}")
                return web.json_response({'error': str(e)}, status=500)

    async def handle_sync_conversation_api(self, request: web.Request) -> web.Response:
        """Handle syncing conversation from frontend to backend session."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON'}, status=400)

        session_id = data.get('session_id', 'default')
        conversation = data.get('conversation', [])

        session = await self.get_or_create_session(session_id)

        # Update the session's conversation history
        session.conversation_history.clear()
        session.conversation_history.extend(conversation)
        session.update_activity()

        return web.json_response({'success': True})

    async def handle_get_conversation_api(self, request: web.Request) -> web.Response:
        """Handle getting current conversation from backend session."""
        session_id = request.query.get('session_id', 'default')
        session = await self.get_or_create_session(session_id)

        return web.json_response({
            'conversation': session.serialize_conversation()
        })

    async def handle_system_stats_api(self, request: web.Request) -> web.Response:
        """Handle getting system stats from backend session."""
        session_id = request.query.get('session_id', 'default')
        session = await self.get_or_create_session(session_id)

        logger.info(f"🔍 DEBUG: System stats request for session {session_id}")

        # Update context usage based on current conversation
        self._update_session_context_usage(session)

        # Get stats from SystemMonitor
        stats = session.system_monitor.get_stats()

        response_data = {
            'cpu_percent': stats.cpu_percent,
            'ram_used_gb': stats.ram_used_gb,
            'ram_total_gb': stats.ram_total_gb,
            'ram_percent': stats.ram_percent,
            'gpu_vram_used_gb': stats.gpu_vram_used_gb,
            'gpu_vram_total_gb': stats.gpu_vram_total_gb,
            'gpu_vram_percent': stats.gpu_vram_percent,
            'context_used_tokens': stats.context_used_tokens,
            'context_max_tokens': stats.context_max_tokens,
            'context_percent': stats.context_percent,
            'conversation_turns': stats.conversation_turns
        }

        logger.info(f"🔍 DEBUG: Returning system stats: {response_data}")

        return web.json_response(response_data)

    def _update_session_context_usage(self, session):
        """Update SystemMonitor with current conversation context usage."""
        try:
            # Use the existing ContextWindowManager for proper token estimation
            context_manager = session.command_context.context_window_manager
            total_tokens = 0

            logger.info(f"🔍 DEBUG: Updating context usage for session {session.session_id}")
            logger.info(f"🔍 DEBUG: Conversation history length: {len(session.conversation_history)}")

            for i, msg in enumerate(session.conversation_history):
                content = msg.get('content', '')
                if content:
                    msg_tokens = context_manager.estimate_tokens(content)
                    total_tokens += msg_tokens
                    logger.info(f"🔍 DEBUG: Message {i}: {msg_tokens} tokens, content preview: {content[:50]}...")

            logger.info(f"🔍 DEBUG: Total tokens calculated: {total_tokens}")
            logger.info(f"🔍 DEBUG: Max context tokens: {session.system_monitor.max_context_tokens}")

            session.system_monitor.update_context_usage(total_tokens)
            session.system_monitor.update_conversation_turns(len(session.conversation_history) // 2)

            # Verify the update worked
            stats = session.system_monitor.get_stats()
            logger.info(f"🔍 DEBUG: SystemMonitor stats after update - context_used: {stats.context_used_tokens}, context_max: {stats.context_max_tokens}, percent: {stats.context_percent}")

        except Exception as e:
            logger.warning(f"Failed to update context usage: {e}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")

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
                    logger.info(f"Cleaning up expired session: {session_id}")
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
            logger.info("🔍 Simple health handler called!")
            return web.json_response({"status": "simple_ok"})

        # Add base routes from OpenAICompatibleServer manually
        app.router.add_get("/health", simple_health)  # Test simple function
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)

        # Add new WebChat endpoints
        app.router.add_get("/v1/oumi/ws", self.handle_websocket)
        app.router.add_post("/v1/oumi/command", self.handle_command_api)
        app.router.add_get("/v1/oumi/branches", self.handle_branches_api)
        app.router.add_post("/v1/oumi/branches", self.handle_branches_api)
        app.router.add_post("/v1/oumi/sync_conversation", self.handle_sync_conversation_api)
        app.router.add_get("/v1/oumi/conversation", self.handle_get_conversation_api)
        app.router.add_get("/v1/oumi/system_stats", self.handle_system_stats_api)

        # Add debug middleware (disabled for now due to parameter issues)
        # async def debug_middleware(request, handler):
        #     logger.info(f"🔍 Request received: {request.method} {request.path}")
        #     try:
        #         response = await handler(request)
        #         logger.info(f"🔍 Response sent: {response.status}")
        #         return response
        #     except Exception as e:
        #         logger.error(f"🔍 Handler error: {e}")
        #         raise
        #
        # app.middlewares.append(debug_middleware)

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
    logger.info("🌐 WebSocket endpoint: ws://{host}:{port}/v1/oumi/ws")
    logger.info("🔗 API endpoints:")
    logger.info(f"   • Chat: http://{host}:{port}/v1/chat/completions")
    logger.info(f"   • Commands: http://{host}:{port}/v1/oumi/command")
    logger.info(f"   • Branches: http://{host}:{port}/v1/oumi/branches")
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
            print=lambda *args: logger.info(f"aiohttp: {' '.join(map(str, args))}")  # Custom print function
        )
        logger.info("🔧 web.run_app returned")
    except Exception as e:
        logger.error(f"❌ Error in web.run_app: {e}")
        import traceback
        traceback.print_exc()
        raise
