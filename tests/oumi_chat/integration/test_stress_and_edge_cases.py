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

"""Integration tests for stress scenarios and edge cases in chat sessions."""

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from oumi_chat.commands import CommandResult
from tests.oumi_chat.utils.chat_test_utils import (
    ChatTestSession,
    create_test_chat_config,
    temporary_test_files,
)


class TestConversationLimits:
    """Test suite for conversation length and memory limits."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_chat_config()
        return ChatTestSession(config)

    def test_very_long_conversation(self, chat_session):
        """Test handling of very long conversations."""
        chat_session.start_session()

        # Create a very long conversation
        base_message = "This is a test message to build up conversation length. " * 10

        message_count = 0
        max_messages = 100  # Reasonable limit for testing

        for i in range(max_messages):
            result = chat_session.send_message(f"Message {i + 1}: {base_message}")
            if result.success:
                message_count += 1
            else:
                # If it fails due to length limits, that's expected
                break

        assert message_count > 10, "Should handle at least 10 messages"

        # Test that session is still functional
        final_result = chat_session.send_message("Final test message")
        assert final_result.success

    def test_very_long_single_message(self, chat_session):
        """Test handling of extremely long single messages."""
        chat_session.start_session()

        # Create progressively longer messages
        base_text = "This is a long message test. "

        for multiplier in [1, 10, 100, 1000, 5000]:
            long_message = base_text * multiplier
            result = chat_session.send_message(long_message)

            if not result.success:
                # Hit a limit, which is expected
                break

            # Test that session is still responsive after long message
            followup_result = chat_session.send_message("Short followup")
            assert followup_result.success

    def test_rapid_fire_messages(self, chat_session):
        """Test sending many messages rapidly."""
        chat_session.start_session()

        # start_time = time.time()
        successful_messages = 0

        # Send 50 messages as quickly as possible
        for i in range(50):
            result = chat_session.send_message(f"Rapid message {i + 1}")
            if result.success:
                successful_messages += 1

        # elapsed_time = time.time() - start_time

        assert successful_messages > 25, (
            f"Should handle rapid messages (got {successful_messages}/50)"
        )

        # Test that session is still functional after rapid fire
        final_result = chat_session.send_message("Final message after rapid fire")
        assert final_result.success

    def test_conversation_with_many_branches(self, chat_session):
        """Test conversation with many branches."""
        chat_session.start_session()

        # Create base conversation
        chat_session.send_message("Base conversation for branching test")

        # Attempt to create many branches
        successful_branches = 0
        branch_limit = 20

        for i in range(branch_limit):
            branch_result = chat_session.execute_command(f"/branch(test_branch_{i})")
            if branch_result.success:
                successful_branches += 1
                # Add content to each branch
                chat_session.send_message(f"Content for branch {i}")

        if successful_branches == 0:
            pytest.skip("Branching not implemented")

        # Test branch listing
        list_result = chat_session.execute_command("/branches()")
        if list_result.success:
            # Should show multiple branches
            if list_result.message:
                assert (
                    str(successful_branches) in list_result.message
                    or "branches" in list_result.message.lower()
                )


class TestEdgeCaseInputs:
    """Test suite for edge case inputs and malformed commands."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_chat_config()
        return ChatTestSession(config)

    def test_empty_and_whitespace_inputs(self, chat_session):
        """Test handling of empty and whitespace-only inputs."""
        chat_session.start_session()

        edge_case_inputs = [
            "",  # Empty string
            " ",  # Single space
            "\n",  # Newline only
            "\t",  # Tab only
            "   \n\t  ",  # Mixed whitespace
        ]

        for input_text in edge_case_inputs:
            result = chat_session.send_message(input_text)
            # Should handle gracefully - either accept or reject with
            # informative message
            assert isinstance(result, CommandResult)

        # Session should still be functional
        normal_result = chat_session.send_message("Normal message after edge cases")
        assert normal_result.success

    def test_special_characters_and_unicode(self, chat_session):
        """Test handling of special characters and unicode."""
        chat_session.start_session()

        special_inputs = [
            "Hello! @#$%^&*()_+-={}[]|\\:;\"'<>?,./",  # Special characters
            "émojis: 😀🎉🚀💻🤖",  # Emojis
            "Ñiño café résumé naïve",  # Accented characters
            "中文测试 日本語 العربية русский",  # Multiple languages
            "Math: ∑∞∂∆∇∈∉∪∩⊂⊃",  # Mathematical symbols
            "Arrows: ←↑→↓↔↕⇐⇑⇒⇓⇔",  # Arrow symbols
        ]

        for special_input in special_inputs:
            result = chat_session.send_message(special_input)
            assert result.success, f"Should handle special input: {special_input}"

        # Test commands with special characters
        special_command_result = chat_session.execute_command("/help()")
        assert isinstance(special_command_result, CommandResult)

    def test_malformed_commands(self, chat_session):
        """Test handling of malformed commands."""
        chat_session.start_session()

        malformed_commands = [
            "/",  # Just slash
            "//help()",  # Double slash
            "/help",  # Missing parentheses
            "/help(())",  # Extra parentheses
            "/help(unclosed",  # Unclosed parentheses
            "/save(file.txt",  # Missing closing paren
            "/save()file.txt)",  # Extra closing paren
            '/save("unclosed)',  # Unclosed quote
            "/save('mixed\")",  # Mixed quotes
            "/command with spaces()",  # Spaces in command name
            "/ help()",  # Space after slash
        ]

        for malformed_cmd in malformed_commands:
            result = chat_session.execute_command(malformed_cmd)
            # Should handle gracefully without crashing
            assert isinstance(result, CommandResult)
            if result.success and result.message:
                # If command succeeded with a message, just continue
                pass
            elif not result.success:
                # If command failed, that's expected for malformed commands
                pass
            else:
                # Command succeeded with no message - that's also acceptable
                pass

        # Session should still work after malformed commands
        recovery_result = chat_session.send_message(
            "Testing recovery after malformed commands"
        )
        assert recovery_result.success

    def test_extremely_nested_structures(self, chat_session):
        """Test handling of extremely nested data structures."""
        chat_session.start_session()

        # Create deeply nested JSON structure
        nested_data: dict[str, Any] = {"level": 0}
        current_level: dict[str, Any] = nested_data

        # Create 50 levels of nesting
        for i in range(1, 50):
            next_level: dict[str, Any] = {"level": i, "data": f"Level {i} data"}
            current_level["next"] = next_level
            current_level = next_level

        nested_json = json.dumps(nested_data)

        with temporary_test_files({"deeply_nested.json": nested_json}) as temp_files:
            # Try to attach deeply nested file
            attach_result = chat_session.execute_command(
                f"/attach({temp_files['deeply_nested.json']})"
            )

            if attach_result.success:
                # Ask about the structure
                structure_result = chat_session.send_message(
                    "Can you analyze the structure of this nested data?"
                )
                assert structure_result.success
            else:
                # If attachment failed, continue with normal operation
                normal_result = chat_session.send_message(
                    "Let's continue without the nested file"
                )
                assert normal_result.success

    def test_binary_and_corrupted_data(self, chat_session):
        """Test handling of binary and corrupted data."""
        chat_session.start_session()

        # Create binary data file
        binary_data = bytes([i % 256 for i in range(1000)])  # Binary data
        corrupted_json = '{"key": "value", "incomplete": '  # Corrupted JSON

        test_files = {
            "binary.bin": binary_data,
            "corrupted.json": corrupted_json,
            "mixed.txt": "Normal text\x00Binary\xff\xfe\xfd",  # Mixed text/binary
        }

        temp_files = []
        try:
            for filename, content in test_files.items():
                temp_file = tempfile.NamedTemporaryFile(
                    mode="wb", suffix=f"_{filename}", delete=False
                )
                if isinstance(content, str):
                    temp_file.write(content.encode("utf-8", errors="ignore"))
                else:
                    temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)

                # Try to attach each file
                attach_result = chat_session.execute_command(
                    f"/attach({temp_file.name})"
                )
                # Should handle gracefully - either process or reject with
                # informative message
                assert isinstance(attach_result, CommandResult)

        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)

        # Session should still be functional
        recovery_result = chat_session.send_message(
            "Testing after binary/corrupted data"
        )
        assert recovery_result.success


class TestResourceExhaustion:
    """Test suite for resource exhaustion scenarios."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_chat_config()
        return ChatTestSession(config)

    def test_memory_intensive_operations(self, chat_session):
        """Test memory-intensive operations."""
        chat_session.start_session()

        # Create large text content
        large_content = "This is a large text block. " * 10000  # ~300KB

        with temporary_test_files({"large_file.txt": large_content}) as temp_files:
            # Try to attach large file
            large_file_result = chat_session.execute_command(
                f"/attach({temp_files['large_file.txt']})"
            )

            if large_file_result.success:
                # Ask for analysis of large content
                analysis_result = chat_session.send_message(
                    "Can you summarize this large document?"
                )
                assert isinstance(analysis_result, CommandResult)

            # Test multiple large operations
            for i in range(5):
                large_message = f"Processing iteration {i}: " + "data " * 1000
                result = chat_session.send_message(large_message)
                # Should handle gracefully
                assert isinstance(result, CommandResult)

    def test_concurrent_operations_simulation(self, chat_session):
        """Test simulation of concurrent operations."""
        chat_session.start_session()

        # Simulate rapid switching between operations
        operations = [
            "/help()",
            "/send(Message 1)",
            "/show(all)",
            "/send(Message 2)",
            "/branches()",
            "/send(Message 3)",
            "/save(test.json)",
            "/send(Message 4)",
        ]

        # Execute operations rapidly
        results = []
        for op in operations:
            if op.startswith("/send("):
                message = op[6:-1]
                result = chat_session.send_message(message)
            else:
                result = chat_session.execute_command(op)
            results.append(result)

        # At least some operations should succeed
        successful_ops = sum(1 for r in results if r.success)
        assert successful_ops >= len(operations) // 2, (
            f"Expected at least half operations to succeed, got "
            f"{successful_ops}/{len(operations)}"
        )

    def test_file_system_stress(self, chat_session):
        """Test file system stress scenarios."""
        chat_session.start_session()

        # Create multiple temporary files
        file_contents = {
            f"stress_test_{i}.txt": f"Content for file {i}\n" * 100 for i in range(10)
        }

        with temporary_test_files(file_contents) as temp_files:
            # Try to attach all files rapidly
            attachment_results = []
            for filename, filepath in temp_files.items():
                result = chat_session.execute_command(f"/attach({filepath})")
                attachment_results.append(result)

            # Process all attached files
            if any(r.success for r in attachment_results):
                analysis_result = chat_session.send_message(
                    "Please analyze all the attached files together"
                )
                assert isinstance(analysis_result, CommandResult)

            # Try to save conversation with all attachments
            save_result = chat_session.execute_command("/save(stress_test_output.json)")
            assert isinstance(save_result, CommandResult)


class TestErrorRecoveryAndResilience:
    """Test suite for error recovery and system resilience."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_chat_config()
        return ChatTestSession(config)

    def test_cascading_error_recovery(self, chat_session):
        """Test recovery from cascading errors."""
        chat_session.start_session()

        # Create a series of operations that might fail
        error_prone_sequence = [
            "/attach(/nonexistent/file1.txt)",  # File error
            "/branch(invalid*branch*name)",  # Invalid branch name
            "/set(invalid_parameter=bad_value)",  # Invalid parameter
            "/save(/root/no_permission.txt)",  # Permission error
            "/switch(nonexistent_branch)",  # Branch error
        ]

        error_count = 0
        for command in error_prone_sequence:
            result = chat_session.execute_command(command)
            if not result.success:
                error_count += 1

                # Test recovery with a simple message after each error
                recovery_result = chat_session.send_message(
                    f"Recovery test after error {error_count}"
                )
                assert recovery_result.success, (
                    f"Should recover after error {error_count}"
                )

        # Final comprehensive recovery test
        final_recovery_tests = [
            "Can you still help me?",
            "What is 2 + 2?",
            "Please tell me about yourself",
        ]

        for recovery_message in final_recovery_tests:
            recovery_result = chat_session.send_message(recovery_message)
            assert recovery_result.success, f"Final recovery failed: {recovery_message}"

    def test_state_corruption_resilience(self, chat_session):
        """Test resilience to state corruption scenarios."""
        chat_session.start_session()

        # Create conversation state
        chat_session.send_message("Building initial state")
        chat_session.send_message("Adding more content")

        # Attempt operations that might corrupt state
        potentially_corrupting_ops = [
            "/delete(999)",  # Delete invalid position
            "/show(-1)",  # Show negative position
            "/branch_from(test, -5)",  # Branch from negative position
            "/switch('')",  # Switch to empty branch name
        ]

        for op in potentially_corrupting_ops:
            chat_session.execute_command(op)
            # Command may fail, but should not corrupt session

            # Verify state is still intact
            state_test = chat_session.send_message("Testing state integrity")
            assert state_test.success, f"State corrupted after: {op}"

            # Verify conversation is still accessible
            show_result = chat_session.execute_command("/show(all)")
            assert isinstance(show_result, CommandResult)

    def test_resource_cleanup_on_errors(self, chat_session):
        """Test that resources are properly cleaned up after errors."""
        chat_session.start_session()

        # Create temporary files that might not be cleaned up properly
        temp_files = []
        try:
            for i in range(5):
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=f"_cleanup_test_{i}.txt"
                )
                temp_file.write(b"Test content for cleanup")
                temp_file.close()
                temp_files.append(temp_file.name)

                # Try to attach file and then cause an error
                chat_session.execute_command(f"/attach({temp_file.name})")

                # Cause a potential error
                chat_session.execute_command("/invalid_operation_after_attach()")

                # Session should still be functional
                function_test = chat_session.send_message(
                    f"Function test after file {i}"
                )
                assert function_test.success

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink()
                except FileNotFoundError:
                    pass  # Already cleaned up

    def test_infinite_loop_prevention(self, chat_session):
        """Test prevention of infinite loops in command processing."""
        chat_session.start_session()

        # Commands that might cause loops or recursive behavior
        potential_loop_commands = [
            "/branch(loop_test)",
            "/switch(loop_test)",
            "/branch(loop_test)",  # Same name again
            "/switch(loop_test)",
            "/switch(main)",
            "/switch(loop_test)",
            "/switch(main)",
        ]

        start_time = time.time()

        for command in potential_loop_commands:
            chat_session.execute_command(command)
            # Should complete quickly without hanging
            elapsed = time.time() - start_time
            assert elapsed < 30, f"Command took too long: {command} ({elapsed:.2f}s)"

        # Final functionality test
        final_result = chat_session.send_message(
            "Testing after potential loop scenarios"
        )
        assert final_result.success
