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

"""Centralized route definitions for Oumi WebChat server."""

from typing import List, Callable, Any, Dict
import time

from aiohttp import web

from oumi.utils.logging import logger
from oumi.webchat.handlers.branch_handler import BranchHandler
from oumi.webchat.handlers.chat_handler import ChatHandler
from oumi.webchat.handlers.command_handler import CommandHandler
from oumi.webchat.handlers.config_handler import ConfigHandler
from oumi.webchat.handlers.system_handler import SystemHandler
from oumi.webchat.handlers.ws_handler import WebSocketHandler


class RouteGroup:
    """Group of related routes with common prefix."""
    
    def __init__(self, prefix: str, description: str = ""):
        """Initialize route group.
        
        Args:
            prefix: URL prefix for all routes in this group.
            description: Optional description of the group.
        """
        self.prefix = prefix
        self.description = description
        self.routes = []
    
    def add_route(self, method: str, path: str, handler: Callable, name: str = None) -> None:
        """Add a route to the group.
        
        Args:
            method: HTTP method (GET, POST, etc.).
            path: URL path relative to the group prefix.
            handler: Route handler function.
            name: Optional route name.
        """
        full_path = self.prefix + path
        self.routes.append((method, full_path, handler, name))
    
    def get(self, path: str, handler: Callable, name: str = None) -> None:
        """Add a GET route.
        
        Args:
            path: URL path relative to the group prefix.
            handler: Route handler function.
            name: Optional route name.
        """
        self.add_route("GET", path, handler, name)
    
    def post(self, path: str, handler: Callable, name: str = None) -> None:
        """Add a POST route.
        
        Args:
            path: URL path relative to the group prefix.
            handler: Route handler function.
            name: Optional route name.
        """
        self.add_route("POST", path, handler, name)
    
    def options(self, path: str, handler: Callable, name: str = None) -> None:
        """Add an OPTIONS route.
        
        Args:
            path: URL path relative to the group prefix.
            handler: Route handler function.
            name: Optional route name.
        """
        self.add_route("OPTIONS", path, handler, name)
    
    def register(self, app: web.Application) -> None:
        """Register all routes in this group with the application.
        
        Args:
            app: aiohttp application to register routes with.
        """
        for method, path, handler, name in self.routes:
            app.router.add_route(method, path, handler, name=name)
            logger.debug(f"Registered route: {method} {path}")


def setup_routes(
    app: web.Application,
    system_handler: SystemHandler,
    chat_handler: ChatHandler,
    branch_handler: BranchHandler,
    command_handler: CommandHandler,
    config_handler: ConfigHandler,
    ws_handler: WebSocketHandler,
    enhanced_features_available: bool = False,
    cors_handler: Callable = None
) -> None:
    """Set up all application routes.
    
    Args:
        app: aiohttp application to add routes to.
        system_handler: System handler for health and models endpoints.
        chat_handler: Chat handler for chat completion endpoints.
        branch_handler: Branch handler for branch operation endpoints.
        command_handler: Command handler for command execution endpoints.
        config_handler: Config handler for config management endpoints.
        ws_handler: WebSocket handler for WebSocket connections.
        enhanced_features_available: Whether enhanced API features are available.
        cors_handler: Optional CORS preflight handler.
    """
    # Base routes
    base = RouteGroup("", "Base routes")
    base.get("/health", system_handler.handle_health)
    
    # OpenAI compatibility routes
    openai = RouteGroup("/v1", "OpenAI-compatible API routes")
    openai.get("/models", system_handler.handle_models)
    openai.post("/chat/completions", chat_handler.handle_chat_completions)
    
    # Enhanced OpenAI routes (optional)
    if enhanced_features_available and hasattr(chat_handler, 'handle_enhanced_chat_completions'):
        openai.post("/chat/completions/enhanced", chat_handler.handle_enhanced_chat_completions)
    
    # Oumi-specific routes
    oumi = RouteGroup("/v1/oumi", "Oumi-specific API routes")
    oumi.get("/ws", ws_handler.handle_websocket)
    oumi.post("/command", command_handler.handle_command_api)
    
    # Branch routes
    oumi.get("/branches", branch_handler.handle_branches_api)
    oumi.post("/branches", branch_handler.handle_branches_api)
    oumi.post("/sync_conversation", branch_handler.handle_sync_conversation_api)
    oumi.get("/conversation", branch_handler.handle_get_conversation_api)
    oumi.post("/reset_history", branch_handler.handle_reset_history_api)
    
    # System routes
    oumi.get("/system_stats", system_handler.handle_system_stats_api)
    oumi.get("/configs", config_handler.handle_get_configs_api)
    oumi.post("/clear_model", system_handler.handle_clear_model_api)
    
    # SSE routes (optional)
    if enhanced_features_available and hasattr(app, 'sse_handler') and app.sse_handler:
        oumi.get("/sse/chat", app.sse_handler.handle_sse_chat)
        oumi.get("/sse/events", app.sse_handler.handle_sse_events)
    
    # CORS preflight routes
    if cors_handler:
        cors_endpoints = [
            "/health",
            "/v1/chat/completions",
            "/v1/models",
            "/v1/oumi/branches", 
            "/v1/oumi/conversation",
            "/v1/oumi/command",
            "/v1/oumi/sync_conversation",
            "/v1/oumi/system_stats",
            "/v1/oumi/configs",
            "/v1/oumi/clear_model",
            "/v1/oumi/reset_history",
        ]
        
        if enhanced_features_available:
            cors_endpoints.extend([
                "/v1/chat/completions/enhanced",
                "/v1/oumi/sse/chat",
                "/v1/oumi/sse/events"
            ])
        
        for endpoint in cors_endpoints:
            app.router.add_options(endpoint, cors_handler)
    
    # Register all route groups
    base.register(app)
    openai.register(app)
    oumi.register(app)
    
    # Log registered routes
    all_routes = [(r.method, r.resource.canonical) for r in app.router.routes()]
    route_count = len(all_routes)
    logger.info(f"Registered {route_count} routes")


def add_middlewares(app: web.Application) -> None:
    """Add standard middlewares to the application.
    
    Args:
        app: aiohttp application to add middlewares to.
    """
    # Add logging middleware
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
        
        try:
            response = await handler(request)
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"✅ {request.method} {request.path} -> {response.status} ({elapsed:.1f}ms)")
            return response
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"❌ {request.method} {request.path} -> ERROR: {e} ({elapsed:.1f}ms)")
            raise
    
    # Add CORS middleware
    @web.middleware
    async def cors_middleware(request, handler):
        response = await handler(request)
        
        # Add CORS headers to all responses (not just preflight OPTIONS)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Session-ID, X-Requested-With"
        
        return response
    
    # Add middlewares to app
    app.middlewares.append(logging_middleware)
    app.middlewares.append(cors_middleware)
