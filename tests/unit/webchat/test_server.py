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

"""Unit tests for WebChat server components."""

import time

import pytest

from oumi.core.configs import InferenceConfig
from tests.unit.webchat.utils.webchat_test_utils import (
    MockWebChatSession,
    WebChatTestServer,
    assert_session_state,
    mock_webchat_server,
)


class TestWebChatSession:
    """Test the WebChatSession component."""

    def test_session_initialization(self):
        """Test that WebChatSession initializes correctly."""
        session = MockWebChatSession("test_session_123")

        assert session.session_id == "test_session_123"
        assert session.is_active
        assert len(session.conversation_history) == 0
        assert session.branch_manager is not None
        assert session.command_context is not None
        assert session.system_monitor is not None

        # Check that created_at and last_activity are recent
        current_time = time.time()
        assert abs(session.created_at - current_time) < 1.0
        assert abs(session.last_activity - current_time) < 1.0

    def test_session_activity_tracking(self):
        """Test that session activity is tracked correctly."""
        session = MockWebChatSession("test_session")
        initial_activity = session.last_activity

        # Wait a bit to ensure time difference
        time.sleep(0.01)

        # Update activity
        session.update_activity()
        assert session.last_activity > initial_activity

        # Adding messages should also update activity
        time.sleep(0.01)
        session.add_message("user", "Hello")
        assert session.last_activity > initial_activity
        assert len(session.conversation_history) == 1
        assert session.conversation_history[0]["role"] == "user"
        assert session.conversation_history[0]["content"] == "Hello"

    def test_session_expiration(self):
        """Test session expiration logic."""
        session = MockWebChatSession("test_session")

        # Fresh session should not be expired
        assert not session.is_expired(timeout_seconds=10)

        # Manually set old activity time
        session.last_activity = time.time() - 20
        assert session.is_expired(timeout_seconds=10)

        # Update activity should reset expiration
        session.update_activity()
        assert not session.is_expired(timeout_seconds=10)

    def test_session_cleanup(self):
        """Test session cleanup functionality."""
        session = MockWebChatSession("test_session")
        assert session.is_active

        session.cleanup()
        assert not session.is_active

    def test_session_conversation_management(self):
        """Test conversation history management."""
        session = MockWebChatSession("test_session")

        # Add multiple messages
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        session.add_message("user", "How are you?")

        assert len(session.conversation_history) == 3
        assert session.conversation_history[0]["role"] == "user"
        assert session.conversation_history[1]["role"] == "assistant"
        assert session.conversation_history[2]["role"] == "user"

        # Check timestamps are in order
        timestamps = [msg["timestamp"] for msg in session.conversation_history]
        assert timestamps == sorted(timestamps)

    def test_session_branch_manager_integration(self):
        """Test integration with branch manager."""
        session = MockWebChatSession("test_session")

        # Test branch manager mock responses
        branch_tree = session.branch_manager.get_branch_tree()
        assert "nodes" in branch_tree
        assert "edges" in branch_tree
        assert len(branch_tree["nodes"]) > 0

        current_branch = session.branch_manager.get_current_branch()
        assert current_branch == "main"

    def test_session_command_context_integration(self):
        """Test integration with command context."""
        session = MockWebChatSession("test_session")

        # Command context should have basic structure
        assert hasattr(session.command_context, "conversation_history")
        assert hasattr(session.command_context, "config")
        assert isinstance(session.command_context.config, InferenceConfig)


class TestWebChatTestServer:
    """Test the WebChatTestServer mock component."""

    def test_server_initialization(self):
        """Test server initialization."""
        server = WebChatTestServer("localhost", 8080)

        assert server.host == "localhost"
        assert server.port == 8080
        assert server.config is not None
        assert not server.running
        assert len(server.sessions) == 0
        assert len(server.endpoints_called) == 0

    def test_server_startup_and_shutdown(self):
        """Test server lifecycle."""
        server = WebChatTestServer()

        # Server should start successfully
        assert server.start()
        assert server.running
        assert server.is_healthy()

        # Server should stop gracefully
        server.stop()
        assert not server.running

    def test_server_startup_error_handling(self):
        """Test server startup error handling."""
        server = WebChatTestServer()
        test_error = RuntimeError("Test startup error")
        server.set_startup_error(test_error)

        with pytest.raises(RuntimeError, match="Test startup error"):
            server.start()

    def test_session_management(self):
        """Test session creation and management."""
        server = WebChatTestServer()
        server.start()

        # Create sessions
        session_id1 = server.create_session()
        session_id2 = server.create_session("custom_session")

        assert len(server.sessions) == 2
        assert "custom_session" in server.sessions
        assert session_id1 in server.sessions

        # Retrieve sessions
        session1 = server.get_session(session_id1)
        session2 = server.get_session("custom_session")

        assert session1 is not None
        assert session2 is not None
        assert session1.session_id == session_id1
        assert session2.session_id == "custom_session"

        # Non-existent session
        assert server.get_session("nonexistent") is None

    def test_session_cleanup(self):
        """Test automated session cleanup."""
        server = WebChatTestServer()
        server.start()

        # Create sessions with different activity times
        session_id1 = server.create_session("active_session")
        session_id2 = server.create_session("expired_session")

        # Make one session expired
        expired_session = server.get_session("expired_session")
        expired_session.last_activity = time.time() - 3600  # 1 hour ago

        assert len(server.sessions) == 2

        # Clean up expired sessions (30 minute timeout)
        server.cleanup_expired_sessions(timeout_seconds=1800)

        assert len(server.sessions) == 1
        assert "active_session" in server.sessions
        assert "expired_session" not in server.sessions

        # Verify the expired session was properly cleaned up
        # (In real implementation, this would clean up resources)

    def test_rest_api_handling(self):
        """Test REST API request handling."""
        server = WebChatTestServer()
        server.start()

        # Test various endpoints
        response1 = server.handle_rest_request("GET", "/branches/session1")
        assert "branches" in response1
        assert "current" in response1

        response2 = server.handle_rest_request("GET", "/system/stats")
        assert "gpu" in response2
        assert "cpu" in response2
        assert "memory" in response2

        response3 = server.handle_rest_request("POST", "/sessions", {"config": "test"})
        assert "status" in response3

        # Check that requests were recorded
        assert len(server.endpoints_called) == 3
        assert server.endpoints_called[0] == ("GET", "/branches/session1", {})
        assert server.endpoints_called[1] == ("GET", "/system/stats", {})
        assert server.endpoints_called[2] == ("POST", "/sessions", {"config": "test"})

    def test_server_context_manager(self):
        """Test server context manager functionality."""
        with mock_webchat_server(port=8081) as server:
            assert server.running
            assert server.port == 8081
            assert server.is_healthy()

        # Server should be stopped after context exit
        assert not server.running

    def test_server_context_manager_with_error(self):
        """Test server context manager with startup error."""
        test_error = ConnectionError("Port already in use")

        with pytest.raises(ConnectionError, match="Port already in use"):
            with mock_webchat_server(startup_error=test_error):
                pass

    def test_server_port_management(self):
        """Test server port management."""
        server = WebChatTestServer(port=9999)
        assert server.get_port() == 9999

        server.port = 8888
        assert server.get_port() == 8888

    def test_server_health_check(self):
        """Test server health check functionality."""
        server = WebChatTestServer()

        # Server should be unhealthy when not running
        assert not server.is_healthy()

        # Server should be healthy when running
        server.start()
        assert server.is_healthy()

        # Server should be unhealthy after stopping
        server.stop()
        assert not server.is_healthy()


class TestWebChatSessionIntegration:
    """Test integration between WebChat components."""

    def test_session_server_integration(self):
        """Test session integration with server."""
        with mock_webchat_server() as server:
            # Create and configure session
            session_id = server.create_session("integration_test")
            session = server.get_session(session_id)

            assert_session_state(
                session,
                expected_messages=0,
                expected_branch="main",
                should_be_active=True,
            )

            # Simulate conversation activity
            session.add_message("user", "Hello")
            session.add_message("assistant", "Hi there!")

            assert_session_state(
                session,
                expected_messages=2,
                expected_branch="main",
                should_be_active=True,
            )

            # Test session persistence across server operations
            response = server.handle_rest_request("GET", f"/sessions/{session_id}")
            assert response["status"] == "ok"

    def test_multiple_sessions_isolation(self):
        """Test that multiple sessions are properly isolated."""
        with mock_webchat_server() as server:
            # Create multiple sessions
            session1_id = server.create_session("session1")
            session2_id = server.create_session("session2")

            session1 = server.get_session(session1_id)
            session2 = server.get_session(session2_id)

            # Add different conversations to each session
            session1.add_message("user", "Hello from session 1")
            session2.add_message("user", "Hello from session 2")
            session2.add_message("assistant", "Response from session 2")

            # Verify isolation
            assert len(session1.conversation_history) == 1
            assert len(session2.conversation_history) == 2
            assert session1.conversation_history[0]["content"] == "Hello from session 1"
            assert session2.conversation_history[0]["content"] == "Hello from session 2"

    def test_session_cleanup_integration(self):
        """Test integrated session cleanup process."""
        with mock_webchat_server() as server:
            # Create sessions with different activity levels
            active_id = server.create_session("active")
            idle_id = server.create_session("idle")
            expired_id = server.create_session("expired")

            active_session = server.get_session(active_id)
            idle_session = server.get_session(idle_id)
            expired_session = server.get_session(expired_id)

            # Simulate different activity patterns
            active_session.update_activity()  # Recently active
            idle_session.last_activity = time.time() - 1000  # Idle but not expired
            expired_session.last_activity = time.time() - 3600  # Expired

            # Perform cleanup
            server.cleanup_expired_sessions(timeout_seconds=1800)  # 30 min timeout

            # Verify cleanup results
            assert server.get_session(active_id) is not None
            assert server.get_session(idle_id) is not None  # Still within timeout
            assert server.get_session(expired_id) is None  # Should be cleaned up

    def test_configuration_integration(self):
        """Test configuration integration across components."""
        from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams

        config = InferenceConfig(
            model=ModelParams(
                model_name="test-model-123",
                model_max_length=512,
                torch_dtype_str="float16",
                trust_remote_code=True,
            ),
            generation=GenerationParams(max_new_tokens=256, temperature=0.7, seed=42),
        )

        with mock_webchat_server(config=config) as server:
            session_id = server.create_session()
            session = server.get_session(session_id)

            # Session should inherit server configuration
            assert session.command_context.config.model.model_name == "test-model-123"
            assert session.command_context.config.generation.max_new_tokens == 256
            assert session.command_context.config.generation.temperature == 0.7
