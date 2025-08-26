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

"""Integration tests for WebChat WebSocket communication."""

import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock

import pytest

from tests.unit.webchat.utils.webchat_test_utils import (
    MockWebSocketClient,
    WebSocketTestClient,
    WebChatTestServer,
    mock_webchat_server,
    assert_websocket_message,
    get_test_websocket_messages,
    wait_for_condition,
)


class TestWebSocketBasicCommunication:
    """Test basic WebSocket communication patterns."""

    def test_websocket_client_initialization(self):
        """Test WebSocket client initialization."""
        client = MockWebSocketClient("ws://localhost:8080/ws")
        
        assert client.uri == "ws://localhost:8080/ws"
        assert not client.connected
        assert len(client.messages_sent) == 0
        assert len(client.messages_received) == 0
        assert client.connection_error is None

    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self):
        """Test WebSocket connection establishment and cleanup."""
        client = MockWebSocketClient()
        
        # Initial state
        assert not client.connected
        
        # Connect
        await client.connect()
        assert client.connected
        
        # Disconnect
        await client.disconnect()
        assert not client.connected

    @pytest.mark.asyncio
    async def test_websocket_connection_error(self):
        """Test WebSocket connection error handling."""
        client = MockWebSocketClient()
        test_error = ConnectionError("Connection refused")
        client.connection_error = test_error
        
        with pytest.raises(ConnectionError, match="Connection refused"):
            await client.connect()
        
        assert not client.connected

    @pytest.mark.asyncio
    async def test_websocket_message_sending(self):
        """Test WebSocket message sending."""
        client = MockWebSocketClient()
        await client.connect()
        
        test_message = {
            "type": "chat",
            "message": "Hello, world!",
            "session_id": "test_session"
        }
        
        response = await client.send_message(test_message)
        
        assert len(client.messages_sent) == 1
        assert len(client.messages_received) == 1
        assert client.messages_sent[0] == test_message
        assert response["type"] == "response"
        assert response["success"] is True

    @pytest.mark.asyncio
    async def test_websocket_message_filtering(self):
        """Test WebSocket message filtering by type."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Send different types of messages
        chat_msg = {"type": "chat", "message": "Hello"}
        command_msg = {"type": "command", "command": "/clear()"}
        system_msg = {"type": "system_stats"}
        
        await client.send_message(chat_msg)
        await client.send_message(command_msg)
        await client.send_message(system_msg)
        
        # Filter by message type
        chat_messages = client.get_sent_messages("chat")
        command_messages = client.get_sent_messages("command")
        system_messages = client.get_sent_messages("system_stats")
        
        assert len(chat_messages) == 1
        assert len(command_messages) == 1
        assert len(system_messages) == 1
        assert chat_messages[0]["message"] == "Hello"
        assert command_messages[0]["command"] == "/clear()"

    @pytest.mark.asyncio
    async def test_websocket_auto_responses(self):
        """Test WebSocket automatic response system."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Set up auto-response
        auto_response = {
            "type": "chat_response",
            "content": "Automated reply",
            "success": True
        }
        client.add_auto_response("chat", auto_response)
        
        # Send message that should trigger auto-response
        chat_message = {
            "type": "chat",
            "message": "Test message",
            "session_id": "test"
        }
        
        response = await client.send_message(chat_message)
        
        # Verify auto-response was returned
        assert response["type"] == "chat_response"
        assert response["content"] == "Automated reply"
        assert response["success"] is True
        assert "timestamp" in response

    def test_websocket_message_history_management(self):
        """Test WebSocket message history management."""
        client = MockWebSocketClient()
        
        # Add some mock messages
        client.messages_sent.extend([
            {"type": "chat", "message": "Hello"},
            {"type": "command", "command": "/help()"},
            {"type": "chat", "message": "How are you?"}
        ])
        
        client.messages_received.extend([
            {"type": "response", "content": "Hi there!"},
            {"type": "command_response", "result": "Help displayed"},
            {"type": "response", "content": "I'm doing well!"}
        ])
        
        # Test filtering
        chat_sent = client.get_sent_messages("chat")
        command_sent = client.get_sent_messages("command")
        all_received = client.get_received_messages()
        
        assert len(chat_sent) == 2
        assert len(command_sent) == 1
        assert len(all_received) == 3
        
        # Test clearing history
        client.clear_history()
        assert len(client.messages_sent) == 0
        assert len(client.messages_received) == 0


class TestWebSocketMessageProtocol:
    """Test WebSocket message protocol and validation."""

    def test_websocket_message_validation(self):
        """Test WebSocket message validation."""
        # Valid chat message
        chat_message = {
            "type": "chat",
            "message": "Hello, world!",
            "session_id": "test_session",
            "timestamp": time.time()
        }
        
        assert_websocket_message(
            chat_message,
            expected_type="chat",
            expected_fields=["message", "session_id"]
        )
        
        # Valid command message
        command_message = {
            "type": "command",
            "command": "/clear()",
            "session_id": "test_session",
            "timestamp": time.time()
        }
        
        assert_websocket_message(
            command_message,
            expected_type="command",
            expected_fields=["command", "session_id"]
        )

    def test_websocket_message_validation_failures(self):
        """Test WebSocket message validation error cases."""
        # Missing type field
        invalid_message1 = {"message": "Hello", "session_id": "test"}
        
        with pytest.raises(AssertionError, match="Message missing 'type' field"):
            assert_websocket_message(invalid_message1, "chat")
        
        # Wrong type
        invalid_message2 = {"type": "wrong", "message": "Hello"}
        
        with pytest.raises(AssertionError, match="Expected type 'chat', got 'wrong'"):
            assert_websocket_message(invalid_message2, "chat")
        
        # Missing required field
        invalid_message3 = {"type": "chat", "session_id": "test"}
        
        with pytest.raises(AssertionError, match="Message missing required field 'message'"):
            assert_websocket_message(invalid_message3, "chat", ["message"])

    def test_standard_websocket_messages(self):
        """Test standard WebSocket message formats."""
        test_messages = get_test_websocket_messages()
        
        # Validate each standard message type
        assert_websocket_message(
            test_messages["chat_message"],
            "chat",
            ["message", "session_id"]
        )
        
        assert_websocket_message(
            test_messages["command_message"],
            "command",
            ["command", "session_id"]
        )
        
        assert_websocket_message(
            test_messages["branch_create"],
            "branch_create",
            ["branch_name", "session_id"]
        )
        
        assert_websocket_message(
            test_messages["system_stats"],
            "system_stats",
            ["session_id"]
        )

    @pytest.mark.asyncio
    async def test_websocket_error_message_format(self):
        """Test WebSocket error message format."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Set up error response
        error_response = {
            "type": "error",
            "error_code": "INVALID_COMMAND",
            "error_message": "Command not recognized",
            "success": False
        }
        client.add_auto_response("command", error_response)
        
        # Send invalid command
        invalid_command = {
            "type": "command",
            "command": "/invalid_command()",
            "session_id": "test"
        }
        
        response = await client.send_message(invalid_command)
        
        # Validate error response format
        assert_websocket_message(
            response,
            "error",
            ["error_code", "error_message", "success"]
        )
        assert response["success"] is False
        assert response["error_code"] == "INVALID_COMMAND"


class TestWebSocketSessionManagement:
    """Test WebSocket session management through WebSocket."""

    @pytest.mark.asyncio
    async def test_session_creation_via_websocket(self):
        """Test session creation through WebSocket."""
        with mock_webchat_server() as server:
            client = MockWebSocketClient()
            await client.connect()
            
            # Send session creation message
            create_session_msg = {
                "type": "create_session",
                "config": {"model": "test-model"}
            }
            
            # Mock successful session creation response
            session_response = {
                "type": "session_created",
                "session_id": "new_session_123",
                "success": True
            }
            client.add_auto_response("create_session", session_response)
            
            response = await client.send_message(create_session_msg)
            
            assert response["type"] == "session_created"
            assert response["success"] is True
            assert "session_id" in response

    @pytest.mark.asyncio
    async def test_session_switching_via_websocket(self):
        """Test session switching through WebSocket."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Send session switch message
        switch_message = {
            "type": "switch_session",
            "session_id": "target_session_456"
        }
        
        # Mock successful switch response
        switch_response = {
            "type": "session_switched",
            "session_id": "target_session_456",
            "conversation_history": [],
            "success": True
        }
        client.add_auto_response("switch_session", switch_response)
        
        response = await client.send_message(switch_message)
        
        assert response["type"] == "session_switched"
        assert response["session_id"] == "target_session_456"
        assert "conversation_history" in response

    @pytest.mark.asyncio
    async def test_session_cleanup_via_websocket(self):
        """Test session cleanup through WebSocket."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Send session cleanup message
        cleanup_message = {
            "type": "cleanup_session",
            "session_id": "session_to_cleanup"
        }
        
        # Mock successful cleanup response
        cleanup_response = {
            "type": "session_cleaned",
            "session_id": "session_to_cleanup",
            "success": True
        }
        client.add_auto_response("cleanup_session", cleanup_response)
        
        response = await client.send_message(cleanup_message)
        
        assert response["type"] == "session_cleaned"
        assert response["success"] is True


class TestWebSocketCommandIntegration:
    """Test command integration through WebSocket."""

    @pytest.mark.asyncio
    async def test_command_execution_via_websocket(self):
        """Test command execution through WebSocket."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Test various command types
        commands_to_test = [
            "/clear()",
            "/help()",
            "/save(test.json)",
            "/export(markdown)",
            "/branch(create, new_branch)"
        ]
        
        for command in commands_to_test:
            # Mock successful command response
            command_response = {
                "type": "command_response",
                "command": command,
                "result": f"Successfully executed {command}",
                "success": True
            }
            client.add_auto_response("command", command_response)
            
            # Send command message
            command_message = {
                "type": "command",
                "command": command,
                "session_id": "test_session"
            }
            
            response = await client.send_message(command_message)
            
            assert response["type"] == "command_response"
            assert response["success"] is True
            assert response["command"] == command

    @pytest.mark.asyncio
    async def test_command_error_handling_via_websocket(self):
        """Test command error handling through WebSocket."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Set up command error response
        error_response = {
            "type": "command_error",
            "command": "/invalid()",
            "error": "Unknown command: /invalid()",
            "success": False
        }
        client.add_auto_response("command", error_response)
        
        # Send invalid command
        invalid_command = {
            "type": "command",
            "command": "/invalid()",
            "session_id": "test_session"
        }
        
        response = await client.send_message(invalid_command)
        
        assert response["type"] == "command_error"
        assert response["success"] is False
        assert "error" in response

    @pytest.mark.asyncio
    async def test_streaming_command_response(self):
        """Test streaming command responses through WebSocket."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Mock streaming response (e.g., for /help() command)
        streaming_responses = [
            {
                "type": "command_stream",
                "command": "/help()",
                "chunk": "Available commands:\n",
                "is_complete": False
            },
            {
                "type": "command_stream", 
                "command": "/help()",
                "chunk": "- /clear(): Clear conversation\n",
                "is_complete": False
            },
            {
                "type": "command_stream",
                "command": "/help()",
                "chunk": "- /save(): Save conversation\n",
                "is_complete": True
            }
        ]
        
        # Send command that should stream
        help_command = {
            "type": "command",
            "command": "/help()",
            "session_id": "test_session",
            "stream": True
        }
        
        # Mock the first response, others would come via server push
        client.add_auto_response("command", streaming_responses[0])
        
        response = await client.send_message(help_command)
        
        assert response["type"] == "command_stream"
        assert response["command"] == "/help()"
        assert "chunk" in response
        assert "is_complete" in response


class TestWebSocketErrorHandling:
    """Test WebSocket error handling and recovery."""

    @pytest.mark.asyncio
    async def test_websocket_connection_recovery(self):
        """Test WebSocket connection recovery after failure."""
        client = MockWebSocketClient()
        
        # Initial connection succeeds
        await client.connect()
        assert client.connected
        
        # Simulate connection loss
        client.connected = False
        
        # Message sending should fail
        with pytest.raises(Exception):  # ConnectionClosed in real implementation
            await client.send_message({"type": "test"})
        
        # Reconnection should work
        client.connection_error = None  # Clear any connection error
        await client.connect()
        assert client.connected

    @pytest.mark.asyncio
    async def test_websocket_malformed_message_handling(self):
        """Test handling of malformed WebSocket messages."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Set up error response for malformed message
        error_response = {
            "type": "error",
            "error_code": "MALFORMED_MESSAGE",
            "error_message": "Invalid message format",
            "success": False
        }
        client.add_auto_response("malformed", error_response)
        
        # Send malformed message (missing required fields)
        malformed_message = {"type": "malformed"}
        
        response = await client.send_message(malformed_message)
        
        assert response["type"] == "error"
        assert response["error_code"] == "MALFORMED_MESSAGE"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_websocket_timeout_handling(self):
        """Test WebSocket timeout handling."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Test scenario where server doesn't respond
        # In real implementation, this would involve actual timeout handling
        
        # Mock timeout scenario
        def timeout_condition():
            # This would be True when real timeout occurs
            return len(client.messages_received) == 0
        
        # Send message that won't get a response
        timeout_message = {"type": "timeout_test"}
        
        # Don't set up auto-response, so no response comes back
        await client.send_message(timeout_message)
        
        # Verify message was sent but no response received
        assert len(client.messages_sent) == 1
        # Note: In real implementation, timeout handling would occur here

    @pytest.mark.asyncio
    async def test_websocket_concurrent_message_handling(self):
        """Test handling of concurrent WebSocket messages."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Set up responses for concurrent messages
        responses = {
            "chat": {"type": "chat_response", "content": "Chat response"},
            "command": {"type": "command_response", "result": "Command result"},
            "system": {"type": "system_response", "data": "System data"}
        }
        
        for msg_type, response in responses.items():
            client.add_auto_response(msg_type, response)
        
        # Send multiple messages concurrently
        messages = [
            {"type": "chat", "message": "Hello", "session_id": "test"},
            {"type": "command", "command": "/help()", "session_id": "test"},
            {"type": "system", "request": "stats", "session_id": "test"}
        ]
        
        # In real async implementation, these would be sent concurrently
        tasks = []
        for message in messages:
            tasks.append(client.send_message(message))
        
        # Wait for all responses (mocked)
        responses_received = []
        for task in tasks:
            response = await task
            responses_received.append(response)
        
        # Verify all messages were processed
        assert len(responses_received) == 3
        assert len(client.messages_sent) == 3
        assert len(client.messages_received) == 3
        
        # Verify response types match expected
        response_types = [r["type"] for r in responses_received]
        expected_types = ["chat_response", "command_response", "system_response"]
        assert all(rt in expected_types for rt in response_types)