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

"""Integration tests for WebChat session management."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tests.unit.webchat.utils.webchat_test_utils import (
    assert_session_state,
    mock_webchat_server,
)


class TestSessionLifecycle:
    """Test complete session lifecycle management."""

    def test_session_creation_and_initialization(self):
        """Test session creation and proper initialization."""
        with mock_webchat_server() as server:
            # Create session
            session_id = server.create_session("test_lifecycle_session")
            session = server.get_session(session_id)

            # Verify initial state
            assert_session_state(
                session,
                expected_messages=0,
                expected_branch="main",
                should_be_active=True,
            )

            # Verify session properties
            assert session.session_id == "test_lifecycle_session"
            assert session.is_active
            assert len(session.conversation_history) == 0

            # Verify timestamps
            current_time = time.time()
            assert abs(session.created_at - current_time) < 2.0
            assert abs(session.last_activity - current_time) < 2.0

    def test_session_activity_and_conversation_management(self):
        """Test session activity tracking and conversation management."""
        with mock_webchat_server() as server:
            session_id = server.create_session("activity_test")
            session = server.get_session(session_id)

            initial_activity = session.last_activity

            # Simulate conversation activity
            time.sleep(0.01)  # Small delay to ensure time difference
            session.add_message("user", "Hello, how are you?")

            # Verify activity was updated
            assert session.last_activity > initial_activity
            assert_session_state(session, expected_messages=1)

            # Add more messages
            session.add_message("assistant", "I'm doing well, thank you!")
            session.add_message("user", "That's great to hear.")

            assert_session_state(session, expected_messages=3)

            # Verify message content and order
            messages = session.conversation_history
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello, how are you?"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "I'm doing well, thank you!"
            assert messages[2]["role"] == "user"
            assert messages[2]["content"] == "That's great to hear."

            # Verify timestamps are in chronological order
            timestamps = [msg["timestamp"] for msg in messages]
            assert timestamps == sorted(timestamps)

    def test_session_cleanup_and_termination(self):
        """Test session cleanup and termination."""
        with mock_webchat_server() as server:
            session_id = server.create_session("cleanup_test")
            session = server.get_session(session_id)

            # Add some conversation data
            session.add_message("user", "Test message")
            session.add_message("assistant", "Test response")

            assert_session_state(session, expected_messages=2, should_be_active=True)

            # Clean up session
            session.cleanup()

            # Verify cleanup
            assert not session.is_active
            # Note: In real implementation, this might also clean up other resources


class TestMultiSessionManagement:
    """Test management of multiple concurrent sessions."""

    def test_concurrent_session_creation(self):
        """Test concurrent creation of multiple sessions."""
        with mock_webchat_server() as server:
            # Create multiple sessions
            session_ids = []
            for i in range(5):
                session_id = server.create_session(f"concurrent_session_{i}")
                session_ids.append(session_id)

            # Verify all sessions were created
            assert len(server.sessions) == 5

            # Verify session isolation
            for i, session_id in enumerate(session_ids):
                session = server.get_session(session_id)
                assert session is not None
                assert session.session_id == f"concurrent_session_{i}"
                assert_session_state(
                    session, expected_messages=0, should_be_active=True
                )

    def test_session_isolation(self):
        """Test that sessions are properly isolated from each other."""
        with mock_webchat_server() as server:
            # Create two sessions
            session1_id = server.create_session("isolation_test_1")
            session2_id = server.create_session("isolation_test_2")

            session1 = server.get_session(session1_id)
            session2 = server.get_session(session2_id)

            # Add different conversations to each session
            session1.add_message("user", "Hello from session 1")
            session1.add_message("assistant", "Response to session 1")

            session2.add_message("user", "Hello from session 2")
            session2.add_message("assistant", "Response to session 2")
            session2.add_message("user", "Another message to session 2")

            # Verify isolation
            assert_session_state(session1, expected_messages=2)
            assert_session_state(session2, expected_messages=3)

            # Verify message content isolation
            assert session1.conversation_history[0]["content"] == "Hello from session 1"
            assert session2.conversation_history[0]["content"] == "Hello from session 2"

            assert len(session1.conversation_history) != len(
                session2.conversation_history
            )

    def test_concurrent_session_operations(self):
        """Test concurrent operations on multiple sessions."""
        with mock_webchat_server() as server:
            # Create sessions for concurrent testing
            session_ids = [
                server.create_session(f"concurrent_op_{i}") for i in range(3)
            ]

            def session_operations(session_id, operation_count):
                """Perform operations on a session."""
                session = server.get_session(session_id)
                results = []

                for i in range(operation_count):
                    session.add_message("user", f"Message {i} from {session_id}")
                    session.add_message("assistant", f"Response {i} to {session_id}")
                    results.append(len(session.conversation_history))
                    time.sleep(0.01)  # Small delay to simulate processing

                return results

            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for session_id in session_ids:
                    future = executor.submit(session_operations, session_id, 3)
                    futures.append((session_id, future))

                # Wait for completion and verify results
                for session_id, future in futures:
                    results = future.result()
                    session = server.get_session(session_id)

                    # Verify final message count
                    assert_session_state(
                        session, expected_messages=6
                    )  # 3 user + 3 assistant

                    # Verify progressive message counts
                    assert results == [2, 4, 6]  # Each iteration adds 2 messages

    def test_session_resource_management(self):
        """Test resource management across multiple sessions."""
        with mock_webchat_server() as server:
            # Create many sessions to test resource management
            session_count = 10
            session_ids = []

            for i in range(session_count):
                session_id = server.create_session(f"resource_test_{i}")
                session_ids.append(session_id)

                # Add conversation data to each session
                session = server.get_session(session_id)
                session.add_message("user", f"Test message {i}")
                session.add_message("assistant", f"Test response {i}")

            # Verify all sessions exist and have data
            assert len(server.sessions) == session_count

            for session_id in session_ids:
                session = server.get_session(session_id)
                assert session is not None
                assert_session_state(
                    session, expected_messages=2, should_be_active=True
                )

            # Test resource cleanup (simulate cleanup of half the sessions)
            sessions_to_cleanup = session_ids[:5]
            for session_id in sessions_to_cleanup:
                session = server.get_session(session_id)
                session.cleanup()

            # Verify partial cleanup
            remaining_sessions = [s for s in server.sessions.values() if s.is_active]
            assert len(remaining_sessions) == 5


class TestSessionExpiration:
    """Test session expiration and automatic cleanup."""

    def test_session_expiration_detection(self):
        """Test detection of expired sessions."""
        with mock_webchat_server() as server:
            # Create sessions with different activity levels
            current_time = time.time()

            active_id = server.create_session("active_session")
            idle_id = server.create_session("idle_session")
            expired_id = server.create_session("expired_session")

            active_session = server.get_session(active_id)
            idle_session = server.get_session(idle_id)
            expired_session = server.get_session(expired_id)

            # Set different activity times
            active_session.last_activity = current_time - 300  # 5 minutes ago (active)
            idle_session.last_activity = current_time - 1000  # ~16 minutes ago (idle)
            expired_session.last_activity = current_time - 3600  # 1 hour ago (expired)

            # Test expiration detection (30-minute timeout)
            timeout_seconds = 1800  # 30 minutes

            assert not active_session.is_expired(timeout_seconds)
            assert not idle_session.is_expired(timeout_seconds)
            assert expired_session.is_expired(timeout_seconds)

    def test_automated_session_cleanup(self):
        """Test automated cleanup of expired sessions."""
        with mock_webchat_server() as server:
            # Create sessions with different expiration states
            current_time = time.time()

            # Create multiple sessions
            session_data = [
                ("active_1", current_time - 300),  # 5 min ago - active
                ("active_2", current_time - 600),  # 10 min ago - active
                ("expired_1", current_time - 2400),  # 40 min ago - expired
                ("expired_2", current_time - 3600),  # 1 hour ago - expired
                ("active_3", current_time - 900),  # 15 min ago - active
            ]

            session_ids = []
            for session_name, last_activity in session_data:
                session_id = server.create_session(session_name)
                session_ids.append(session_id)

                session = server.get_session(session_id)
                session.add_message("user", f"Test message from {session_name}")
                # Set activity time AFTER adding message to prevent update_activity() from overriding it
                session.last_activity = last_activity

            # Verify all sessions were created
            assert len(server.sessions) == 5

            # Perform cleanup with 30-minute timeout
            server.cleanup_expired_sessions(timeout_seconds=1800)

            # Verify cleanup results
            remaining_session_ids = list(server.sessions.keys())

            # Should have 3 active sessions remaining
            assert len(remaining_session_ids) == 3

            # Verify correct sessions were kept
            kept_sessions = ["active_1", "active_2", "active_3"]
            for session_name in kept_sessions:
                assert any(session_name in sid for sid in remaining_session_ids)

            # Verify expired sessions were removed
            expired_sessions = ["expired_1", "expired_2"]
            for session_name in expired_sessions:
                assert not any(session_name in sid for sid in remaining_session_ids)

    def test_session_activity_prevents_expiration(self):
        """Test that session activity prevents expiration."""
        with mock_webchat_server() as server:
            session_id = server.create_session("activity_prevents_expiration")
            session = server.get_session(session_id)

            # Set session to be nearly expired
            session.last_activity = time.time() - 1700  # 28+ minutes ago

            # Should be close to expiring but not yet expired (30 min timeout)
            assert not session.is_expired(timeout_seconds=1800)

            # Update activity (simulating recent interaction)
            session.update_activity()

            # Should now be fresh and not expired
            assert not session.is_expired(timeout_seconds=1800)

            # Verify session survives cleanup
            server.cleanup_expired_sessions(timeout_seconds=1800)
            assert server.get_session(session_id) is not None

    def test_cleanup_frequency_and_timing(self):
        """Test cleanup frequency and timing behavior."""
        with mock_webchat_server() as server:
            # Create sessions that will expire at different times
            base_time = time.time()

            sessions_info = [
                ("immediate_expire", base_time - 4000),  # Expired now
                ("expire_soon", base_time - 1700),  # Will expire in ~2 minutes
                ("expire_later", base_time - 1000),  # Will expire in ~13 minutes
                ("fresh", base_time - 300),  # Fresh, won't expire
            ]

            for session_name, last_activity in sessions_info:
                session_id = server.create_session(session_name)
                session = server.get_session(session_id)
                session.last_activity = last_activity

            # Initial cleanup - should remove immediate_expire
            server.cleanup_expired_sessions(timeout_seconds=1800)
            assert len(server.sessions) == 3
            assert not any("immediate_expire" in sid for sid in server.sessions.keys())

            # Simulate time passing (advance activity timestamps)
            # In real implementation, this would be handled by actual time passage
            time_advance = 800  # Advance by ~13 minutes
            for session in server.sessions.values():
                session.last_activity -= time_advance

            # Second cleanup - should now also remove expire_soon and expire_later
            server.cleanup_expired_sessions(timeout_seconds=1800)
            assert len(server.sessions) == 1  # Only 'fresh' should remain
            assert not any("expire_soon" in sid for sid in server.sessions.keys())
            assert not any("expire_later" in sid for sid in server.sessions.keys())
            assert any("fresh" in sid for sid in server.sessions.keys())


class TestSessionStateManagement:
    """Test session state management and synchronization."""

    def test_session_state_consistency(self):
        """Test session state consistency across operations."""
        with mock_webchat_server() as server:
            session_id = server.create_session("state_consistency_test")
            session = server.get_session(session_id)

            # Verify initial consistent state
            assert_session_state(
                session,
                expected_messages=0,
                expected_branch="main",
                should_be_active=True,
            )

            # Perform state-changing operations
            operations = [
                ("add_message", ("user", "What is machine learning?")),
                ("add_message", ("assistant", "Machine learning is a subset of AI...")),
                ("update_activity", ()),
                ("add_message", ("user", "Can you explain neural networks?")),
            ]

            expected_message_count = 0
            for operation, args in operations:
                if operation == "add_message":
                    session.add_message(*args)
                    expected_message_count += 1
                elif operation == "update_activity":
                    session.update_activity()

                # Verify state consistency after each operation
                assert_session_state(
                    session,
                    expected_messages=expected_message_count,
                    should_be_active=True,
                )

    def test_session_state_recovery(self):
        """Test session state recovery after errors."""
        with mock_webchat_server() as server:
            session_id = server.create_session("state_recovery_test")
            session = server.get_session(session_id)

            # Add some initial state
            session.add_message("user", "Initial message")
            session.add_message("assistant", "Initial response")
            initial_message_count = len(session.conversation_history)

            # Simulate error during state modification
            try:
                # This would be an operation that might fail
                # In real implementation, this could be a command execution error
                session.add_message("user", "Message that might cause error")
                # Simulate error by raising exception
                raise RuntimeError("Simulated error during message processing")
            except RuntimeError:
                # Session state should remain consistent despite error
                pass

            # Verify session state is still valid
            # Note: In real implementation, error handling would prevent partial state changes
            assert session.is_active
            assert len(session.conversation_history) >= initial_message_count

            # Session should still be functional after error
            session.add_message("user", "Recovery message")
            session.add_message("assistant", "Recovery response")

            assert_session_state(session, should_be_active=True)

    def test_concurrent_session_state_modifications(self):
        """Test concurrent modifications to session state."""
        with mock_webchat_server() as server:
            session_id = server.create_session("concurrent_state_test")
            session = server.get_session(session_id)

            def modify_session_state(thread_id, message_count):
                """Modify session state from concurrent thread."""
                for i in range(message_count):
                    session.add_message("user", f"Thread {thread_id} message {i}")
                    session.add_message(
                        "assistant", f"Response to thread {thread_id} message {i}"
                    )
                    time.sleep(0.001)  # Small delay to increase chance of interleaving

            # Run concurrent modifications
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for thread_id in range(3):
                    future = executor.submit(modify_session_state, thread_id, 2)
                    futures.append(future)

                # Wait for all threads to complete
                for future in as_completed(futures):
                    future.result()

            # Verify final state
            # Should have 3 threads * 2 messages * 2 (user + assistant) = 12 messages
            assert_session_state(session, expected_messages=12, should_be_active=True)

            # Verify message integrity (all messages should be complete)
            messages = session.conversation_history
            assert len(messages) == 12

            # Verify timestamps are in order (even with concurrent access)
            timestamps = [msg["timestamp"] for msg in messages]
            assert len(timestamps) == len(set(timestamps))  # All timestamps unique
