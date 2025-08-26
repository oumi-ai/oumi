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

"""Test utilities for WebChat functionality."""

import asyncio
import json
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch, AsyncMock

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from oumi.core.configs import InferenceConfig
from tests.utils.chat_test_utils import ChatTestSession, create_test_inference_config


class MockWebSocketClient:
    """Mock WebSocket client for testing WebSocket communication."""

    def __init__(self, uri: str = "ws://localhost:8080/ws"):
        self.uri = uri
        self.connected = False
        self.messages_sent: List[Dict[str, Any]] = []
        self.messages_received: List[Dict[str, Any]] = []
        self.connection_error: Optional[Exception] = None
        self.auto_responses: Dict[str, Dict[str, Any]] = {}

    async def connect(self):
        """Mock WebSocket connection."""
        if self.connection_error:
            raise self.connection_error
        self.connected = True

    async def disconnect(self):
        """Mock WebSocket disconnection."""
        self.connected = False

    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and optionally receive an auto-response."""
        if not self.connected:
            raise ConnectionClosed(None, None)
        
        self.messages_sent.append(message)
        
        # Check for auto-responses
        message_type = message.get("type", "")
        if message_type in self.auto_responses:
            response = self.auto_responses[message_type].copy()
            response["timestamp"] = time.time()
            self.messages_received.append(response)
            return response
        
        # Default success response
        response = {
            "type": "response",
            "success": True,
            "message": "Mock response",
            "timestamp": time.time()
        }
        self.messages_received.append(response)
        return response

    def add_auto_response(self, message_type: str, response: Dict[str, Any]):
        """Add an automatic response for a given message type."""
        self.auto_responses[message_type] = response

    def get_sent_messages(self, message_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get sent messages, optionally filtered by type."""
        if message_type:
            return [msg for msg in self.messages_sent if msg.get("type") == message_type]
        return self.messages_sent.copy()

    def get_received_messages(self, message_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get received messages, optionally filtered by type."""
        if message_type:
            return [msg for msg in self.messages_received if msg.get("type") == message_type]
        return self.messages_received.copy()

    def clear_history(self):
        """Clear message history."""
        self.messages_sent.clear()
        self.messages_received.clear()


class MockWebChatSession:
    """Mock WebChat session for testing session management."""

    def __init__(self, session_id: str = "test_session_123", config: Optional[InferenceConfig] = None):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.conversation_history: List[Dict[str, Any]] = []
        self.branch_manager = Mock()
        self.command_context = Mock()
        self.system_monitor = Mock()
        self.is_active = True
        
        # Mock branch manager methods
        self.branch_manager.get_branch_tree.return_value = {
            "nodes": [{"id": "main", "label": "Main", "active": True}],
            "edges": []
        }
        self.branch_manager.get_current_branch.return_value = "main"
        
        # Mock command context
        self.command_context.conversation_history = []
        self.command_context.config = config or create_test_inference_config()

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        self.conversation_history.append(message)
        self.update_activity()

    def cleanup(self):
        """Clean up session resources."""
        self.is_active = False

    def is_expired(self, timeout_seconds: int = 1800) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_activity) > timeout_seconds


class WebChatTestServer:
    """Test server for WebChat functionality."""

    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 8080,
        config: Optional[InferenceConfig] = None
    ):
        self.host = host
        self.port = port
        self.config = config or create_test_inference_config()
        self.sessions: Dict[str, MockWebChatSession] = {}
        self.running = False
        self.server_thread: Optional[threading.Thread] = None
        self.startup_error: Optional[Exception] = None
        
        # Mock endpoints
        self.endpoints_called: List[Tuple[str, str, Dict[str, Any]]] = []  # (method, path, data)
        
    def start(self) -> bool:
        """Start the mock server."""
        if self.startup_error:
            raise self.startup_error
        
        self.running = True
        return True

    def stop(self):
        """Stop the mock server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=1.0)

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new WebChat session."""
        if not session_id:
            session_id = f"session_{len(self.sessions)}_{int(time.time())}"
        
        session = MockWebChatSession(session_id, self.config)
        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[MockWebChatSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def cleanup_expired_sessions(self, timeout_seconds: int = 1800):
        """Clean up expired sessions."""
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(timeout_seconds)
        ]
        
        for session_id in expired_sessions:
            session = self.sessions[session_id]
            session.cleanup()
            del self.sessions[session_id]

    def handle_rest_request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle a REST API request."""
        self.endpoints_called.append((method, path, data or {}))
        
        # Mock responses for common endpoints
        if path.startswith("/branches"):
            return {"branches": ["main"], "current": "main"}
        elif path.startswith("/system"):
            return {"gpu": {"usage": 0.5}, "cpu": {"usage": 0.3}, "memory": {"usage": 0.4}}
        elif path.startswith("/sessions") and method == "POST":
            return {"status": "ok", "message": "Session created"}
        elif path.startswith("/sessions"):
            if method == "GET" and len(path.split("/")) > 2:  # /sessions/{session_id}
                return {"status": "ok", "session_id": path.split("/")[-1]}
            return {"sessions": list(self.sessions.keys())}
        else:
            return {"status": "ok", "message": f"Mock response for {method} {path}"}

    def is_healthy(self) -> bool:
        """Check if server is healthy."""
        return self.running

    def get_port(self) -> int:
        """Get server port."""
        return self.port

    def set_startup_error(self, error: Exception):
        """Set an error to be raised on startup."""
        self.startup_error = error


@contextmanager
def mock_webchat_server(
    host: str = "localhost",
    port: int = 8080,
    config: Optional[InferenceConfig] = None,
    startup_error: Optional[Exception] = None
):
    """Context manager for mock WebChat server."""
    server = WebChatTestServer(host, port, config)
    
    if startup_error:
        server.set_startup_error(startup_error)
    
    try:
        server.start()
        yield server
    finally:
        server.stop()


class WebSocketTestClient:
    """Real WebSocket test client for integration testing."""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        self.connected = False
        self.messages_received: List[Dict[str, Any]] = []
        self._receive_task: Optional[asyncio.Task] = None
        
    async def connect(self, timeout: float = 5.0):
        """Connect to WebSocket server."""
        try:
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.uri),
                timeout=timeout
            )
            self.connected = True
            self._receive_task = asyncio.create_task(self._receive_messages())
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.uri}: {e}")
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.connected = False
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the server."""
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to server")
        
        await self.websocket.send(json.dumps(message))
    
    async def wait_for_message(
        self, 
        message_type: Optional[str] = None,
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """Wait for a specific type of message."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            for message in self.messages_received:
                if message_type is None or message.get("type") == message_type:
                    self.messages_received.remove(message)
                    return message
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Timeout waiting for message type: {message_type}")
    
    async def _receive_messages(self):
        """Receive messages from server."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    self.messages_received.append(data)
                except json.JSONDecodeError:
                    # Ignore invalid JSON
                    pass
        except ConnectionClosed:
            pass


def create_test_gradio_interface(config: Optional[InferenceConfig] = None):
    """Create a test Gradio interface with mocked components."""
    config = config or create_test_inference_config()
    
    # Mock Gradio components
    mock_interface = Mock()
    mock_interface.chatbot = Mock()
    mock_interface.msg_input = Mock()
    mock_interface.upload_btn = Mock()
    mock_interface.clear_btn = Mock()
    mock_interface.delete_btn = Mock()
    mock_interface.regen_btn = Mock()
    mock_interface.export_btn = Mock()
    mock_interface.help_btn = Mock()
    mock_interface.branch_tree = Mock()
    mock_interface.system_monitor = Mock()
    mock_interface.settings_panel = Mock()
    
    # Mock interface methods
    mock_interface.update_chatbot = Mock()
    mock_interface.update_branch_tree = Mock()
    mock_interface.update_system_monitor = Mock()
    mock_interface.show_message = Mock()
    mock_interface.clear_chat = Mock()
    
    return mock_interface


def create_mock_file_upload(filename: str, content: bytes, content_type: str = "text/plain"):
    """Create a mock file upload object."""
    mock_file = Mock()
    mock_file.name = filename
    mock_file.orig_name = filename
    mock_file.size = len(content)
    mock_file.content_type = content_type
    
    # Mock file reading
    def read_content():
        return content
    
    mock_file.read = read_content
    
    # Mock file path for temporary files
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(content)
    temp_file.close()
    mock_file.temp_path = temp_file.name
    
    return mock_file


def assert_websocket_message(
    message: Dict[str, Any],
    expected_type: str,
    expected_fields: Optional[List[str]] = None
):
    """Assert that a WebSocket message has the expected structure."""
    assert "type" in message, "Message missing 'type' field"
    assert message["type"] == expected_type, f"Expected type '{expected_type}', got '{message['type']}'"
    
    if expected_fields:
        for field in expected_fields:
            assert field in message, f"Message missing required field '{field}'"


def assert_session_state(
    session: MockWebChatSession,
    expected_messages: Optional[int] = None,
    expected_branch: Optional[str] = None,
    should_be_active: bool = True
):
    """Assert that a session has the expected state."""
    assert session.is_active == should_be_active, f"Session active state mismatch"
    
    if expected_messages is not None:
        assert len(session.conversation_history) == expected_messages, \
            f"Expected {expected_messages} messages, got {len(session.conversation_history)}"
    
    if expected_branch is not None:
        current_branch = session.branch_manager.get_current_branch()
        assert current_branch == expected_branch, \
            f"Expected branch '{expected_branch}', got '{current_branch}'"


def wait_for_condition(
    condition_func,
    timeout: float = 5.0,
    check_interval: float = 0.1,
    error_message: str = "Condition not met within timeout"
):
    """Wait for a condition to become true."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(check_interval)
    
    raise TimeoutError(error_message)


class PortTestHelper:
    """Helper for testing port management."""
    
    @staticmethod
    def find_free_port() -> int:
        """Find a free port for testing."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    @staticmethod
    def is_port_open(host: str, port: int) -> bool:
        """Check if a port is open."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                return s.connect_ex((host, port)) == 0
        except Exception:
            return False
    
    @staticmethod
    @contextmanager
    def occupy_port(host: str, port: int):
        """Context manager to temporarily occupy a port."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            sock.listen(1)
            yield sock
        finally:
            sock.close()


# Test data and fixtures
def get_test_websocket_messages() -> Dict[str, Dict[str, Any]]:
    """Get sample WebSocket messages for testing."""
    return {
        "chat_message": {
            "type": "chat",
            "message": "Hello, how are you?",
            "session_id": "test_session"
        },
        "command_message": {
            "type": "command",
            "command": "/clear()",
            "session_id": "test_session"
        },
        "branch_create": {
            "type": "branch_create",
            "branch_name": "test_branch",
            "session_id": "test_session"
        },
        "system_stats": {
            "type": "system_stats",
            "session_id": "test_session"
        }
    }


def get_test_rest_endpoints() -> List[Tuple[str, str, Dict[str, Any]]]:
    """Get sample REST API endpoints for testing."""
    return [
        ("GET", "/health", {}),
        ("GET", "/sessions", {}),
        ("POST", "/sessions", {"config": "test"}),
        ("GET", "/branches/test_session", {}),
        ("POST", "/branches/test_session", {"name": "new_branch"}),
        ("DELETE", "/branches/test_session/branch_name", {}),
        ("GET", "/system/stats", {}),
        ("POST", "/chat/completions", {"messages": [{"role": "user", "content": "test"}]})
    ]


def cleanup_test_files(*file_paths: str):
    """Clean up test files after test completion."""
    for file_path in file_paths:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors