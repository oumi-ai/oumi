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

"""Modular Oumi WebChat server with WebSocket and interactive command support."""

import asyncio
import time
from typing import Dict, Optional, Any

from aiohttp import web

# Import logger first for use in error handling
from oumi.utils.logging import logger

# Import core components
from oumi.core.configs import InferenceConfig
from oumi.core.inference import BaseInferenceEngine
import oumi
from oumi.infer import get_engine

# Import structured components
from oumi.webchat.core.session_manager import SessionManager
# Re-export for backward-compat imports in tests
from oumi.webchat.core.session import WebChatSession  # noqa: F401
from oumi.webchat.handlers.branch_handler import BranchHandler
from oumi.webchat.handlers.chat_handler import ChatHandler
from oumi.webchat.handlers.command_handler import CommandHandler
from oumi.webchat.handlers.config_handler import ConfigHandler
from oumi.webchat.handlers.system_handler import SystemHandler
from oumi.webchat.handlers.ws_handler import WebSocketHandler
from oumi.webchat.services.branch_service import BranchService
from oumi.webchat.services.command_service import CommandService
from oumi.webchat.services.inference_service import InferenceService
from oumi.webchat.services.persistence_service import PersistenceService
from oumi.webchat.routes import setup_routes, add_middlewares


# Check for enhanced features
try:
    from oumi.webchat.api_responses import ResponseFormatter, RequestValidator
    from oumi.webchat.sse_handler import SSEHandler
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced API features not available - missing dependencies")
    ENHANCED_FEATURES_AVAILABLE = False


class OumiWebServer:
    """Modular Oumi WebChat server with WebSocket and interactive command support."""
    
    def __init__(
        self,
        config: InferenceConfig,
        system_prompt: Optional[str] = None,
        db_path: Optional[str] = None,
        base_dir: Optional[str] = None,
        session_locking_enabled: bool = True
    ):
        """Initialize the web server with modular components.
        
        Args:
            config: Inference configuration.
            system_prompt: Optional system prompt.
            db_path: Optional path to the SQLite database file.
            base_dir: Optional base directory path.
            session_locking_enabled: Whether to enable session locking for concurrency control.
        """
        # Initialize base properties
        self.config = config
        self.system_prompt = system_prompt
        self.base_dir = base_dir
        
        # Set up OpenAI-compatible model info
        self.model_info = {
            "id": getattr(config.model, "model_name", "oumi-model"),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "oumi",
        }
        
        # Apply session locking setting to config for sharing with components
        setattr(config, 'session_locking_enabled', session_locking_enabled)
        
        # Initialize services
        logger.info("Initializing services...")
        self.persistence_service = PersistenceService(db_path)
        self.inference_service = InferenceService(config)
        self.branch_service = BranchService(self.persistence_service)
        self.command_service = CommandService()
        
        # Initialize session manager
        logger.info("Initializing session manager...")
        self.session_manager = SessionManager(config, system_prompt)
        
        # Initialize handlers
        logger.info("Initializing handlers...")
        self.system_handler = SystemHandler(
            self.session_manager, 
            enhanced_features_available=ENHANCED_FEATURES_AVAILABLE
        )
        self.chat_handler = ChatHandler(
            self.session_manager, 
            system_prompt,
            self.persistence_service.db if self.persistence_service.is_enabled else None,
            enhanced_features_available=ENHANCED_FEATURES_AVAILABLE
        )
        self.branch_handler = BranchHandler(
            self.session_manager,
            self.persistence_service.db if self.persistence_service.is_enabled else None
        )
        self.command_handler = CommandHandler(
            self.session_manager,
            self.persistence_service.db if self.persistence_service.is_enabled else None
        )
        self.config_handler = ConfigHandler(self.base_dir)
        self.ws_handler = WebSocketHandler(
            self.session_manager,
            system_prompt,
            self.persistence_service.db if self.persistence_service.is_enabled else None
        )
        
        # Enhanced features
        self.sse_handler = None
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                self.sse_handler = SSEHandler()
                logger.info("✨ Enhanced features enabled")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize enhanced features: {e}")
        
        # Session cleanup task
        self._cleanup_task = None
        
        logger.info("✅ OumiWebServer initialized successfully")
    
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
    
    def create_app(self) -> web.Application:
        """Create and configure the aiohttp application.
        
        Returns:
            Configured aiohttp web Application.
        """
        app = web.Application()
        
        # Make SSE handler available to the app if enabled
        if self.sse_handler:
            app.sse_handler = self.sse_handler
        
        # Set up routes
        setup_routes(
            app,
            self.system_handler,
            self.chat_handler,
            self.branch_handler,
            self.command_handler,
            self.config_handler,
            self.ws_handler,
            enhanced_features_available=ENHANCED_FEATURES_AVAILABLE,
            cors_handler=self.handle_cors_preflight
        )
        
        # Add middlewares
        add_middlewares(app)
        
        return app

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests.
        
        Args:
            request: The HTTP request object
            
        Returns:
            JSON response with health status
        """
        return web.json_response({"status": "ok"})
        
    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """Handle chat completions requests through the ChatHandler.
        
        This preserves the WebChat semantics (client-provided messages are authoritative; branch-aware sync).
        
        Args:
            request: The HTTP request object
            
        Returns:
            HTTP response from the chat handler
        """
        return await self.chat_handler.handle_chat_completions(request)
    
    async def start_background_tasks(self, app: web.Application):
        """Start background tasks."""
        self._cleanup_task = asyncio.create_task(self.session_manager.start_cleanup_task())
    
    async def cleanup_background_tasks(self, app: web.Application):
        """Clean up background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all engines
        self.inference_service.clear_all_engines()
        
        # Close SSE handler if available
        if self.sse_handler:
            self.sse_handler.close_all_streams()


def run_webchat_server(
    config: InferenceConfig,
    host: str = "0.0.0.0",
    port: int = 9000,
    system_prompt: Optional[str] = None,
    db_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    session_locking_enabled: bool = True,
) -> None:
    """Run the Oumi WebChat server.
    
    Args:
        config: Inference configuration.
        host: Host to listen on.
        port: Port to listen on.
        system_prompt: Optional system prompt.
        db_path: Optional path to the SQLite database file.
        base_dir: Optional base directory path.
        session_locking_enabled: Whether to enable session locking for concurrency control.
    """
    server = OumiWebServer(
        config, 
        system_prompt, 
        db_path, 
        base_dir,
        session_locking_enabled
    )
    app = server.create_app()
    
    # Enable startup/cleanup handlers for proper session management
    app.on_startup.append(server.start_background_tasks)
    app.on_cleanup.append(server.cleanup_background_tasks)
    
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
            print=lambda *args: logger.info(f"aiohttp: {' '.join(map(str, args))}"),
        )
        logger.info("🔧 web.run_app returned")
    except Exception as e:
        logger.error(f"❌ Error in web.run_app: {e}")
        import traceback
        
        traceback.print_exc()
        raise
