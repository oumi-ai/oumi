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

import aiohttp_cors
from aiohttp import web, WSMsgType
from rich.console import Console

from oumi.core.commands.command_context import CommandContext
from oumi.core.commands.command_parser import CommandParser
from oumi.core.commands.command_router import CommandRouter
from oumi.core.commands.conversation_branches import ConversationBranchManager
from oumi.core.configs import InferenceConfig
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
        self.system_monitor = SystemMonitor(self.console)
        self.thinking_processor = ThinkingProcessor()
        self.branch_manager = ConversationBranchManager(self.conversation_history)
        
        # Initialize command system
        self.command_context = CommandContext(
            console=self.console,
            config=config,
            conversation_history=self.conversation_history,
            inference_engine=None,  # Will be set when needed
            system_monitor=self.system_monitor,
            thinking_processor=self.thinking_processor,
            branch_manager=self.branch_manager,
            _style=type('Style', (), {'use_emoji': True, 'expand_panels': True})()
        )
        
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
        super().__init__(config, system_prompt)
        
        # Session management
        self.sessions: Dict[str, WebChatSession] = {}
        self.session_cleanup_interval = 3600  # 1 hour
        self.max_idle_time = 1800  # 30 minutes
        
        # Start cleanup task
        self._cleanup_task = None

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
            monitor_data = session.system_monitor.get_current_stats()
            await ws.send_str(json.dumps({
                'type': 'system_update',
                'data': monitor_data
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
            if session.command_context.inference_engine is None:
                from oumi.infer import get_engine
                session.command_context.inference_engine = get_engine(session.config)
            
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
                result = session.command_router.execute_command(parsed_command)
                
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
            # Execute command via command router
            from oumi.core.commands.command_parser import ParsedCommand
            parsed_command = ParsedCommand(command=command, args=args)
            result = session.command_router.execute_command(parsed_command)
            
            response_data = {
                'success': result.success,
                'message': result.message,
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
            logger.error(f"API command execution error: {e}")
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
        app = super().create_app()
        
        # Add new endpoints
        app.router.add_get("/v1/oumi/ws", self.handle_websocket)
        app.router.add_post("/v1/oumi/command", self.handle_command_api)
        app.router.add_get("/v1/oumi/branches", self.handle_branches_api)
        app.router.add_post("/v1/oumi/branches", self.handle_branches_api)
        
        # Add CORS support for WebSocket
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(app.router.routes()):
            cors.add(route)
        
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
    port: int = 8000,
    system_prompt: Optional[str] = None,
) -> None:
    """Run the Oumi WebChat server."""
    server = OumiWebServer(config, system_prompt)
    app = server.create_app()
    
    # Add startup/cleanup handlers
    app.on_startup.append(server.start_background_tasks)
    app.on_cleanup.append(server.cleanup_background_tasks)
    
    logger.info("🚀 Starting Oumi WebChat server")
    logger.info(f"📍 Server URL: http://{host}:{port}")
    logger.info("🌐 WebSocket endpoint: ws://{host}:{port}/v1/oumi/ws")
    logger.info("🔗 API endpoints:")
    logger.info(f"   • Chat: http://{host}:{port}/v1/chat/completions")
    logger.info(f"   • Commands: http://{host}:{port}/v1/oumi/command")
    logger.info(f"   • Branches: http://{host}:{port}/v1/oumi/branches")
    logger.info("🛑 Press Ctrl+C to stop")
    
    # Run the server
    web.run_app(app, host=host, port=port, access_log=logger)