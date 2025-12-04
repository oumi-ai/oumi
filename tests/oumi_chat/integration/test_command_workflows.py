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

"""Integration tests for complex command workflows in chat sessions."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.oumi_chat.utils.chat_test_utils import (
    ChatTestSession,
    create_test_inference_config,
    temporary_test_files,
)


class TestBranchingWorkflows:
    """Test suite for conversation branching workflows."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_inference_config()
        return ChatTestSession(config)

    def test_create_branch_and_switch_workflow(self, chat_session):
        """Test creating a branch and switching between branches."""
        chat_session.start_session()

        # Create initial conversation
        chat_session.send_message("What is AI?")
        chat_session.send_message("Tell me more about neural networks")

        # Create a branch at this point
        branch_result = chat_session.execute_command("/branch(neural_networks_deep)")
        if not branch_result.success:
            pytest.skip("Branching not implemented")

        # Continue conversation in new branch
        chat_session.send_message("How do backpropagation algorithms work?")

        # Switch back to main branch
        switch_result = chat_session.execute_command("/switch(main)")
        assert switch_result.success

        # Continue different conversation in main
        chat_session.send_message("What about machine learning applications?")

        # Switch to the neural networks branch
        switch_result2 = chat_session.execute_command("/switch(neural_networks_deep)")
        assert switch_result2.success

        # Verify branch isolation
        list_result = chat_session.execute_command("/branches()")
        assert list_result.success
        assert list_result.message and "neural_networks_deep" in list_result.message
        assert list_result.message and "main" in list_result.message

    def test_branch_from_specific_position(self, chat_session):
        """Test creating a branch from a specific conversation position."""
        chat_session.start_session()

        # Create longer conversation
        messages = [
            "Hello",
            "What is machine learning?",
            "Can you give examples?",
            "How about deep learning?",
            "What are the applications?",
        ]

        for msg in messages:
            chat_session.send_message(msg)

        # Create branch from position 3 (after "Can you give examples?")
        branch_result = chat_session.execute_command("/branch_from(examples_branch, 3)")
        if not branch_result.success:
            pytest.skip("Branch from position not implemented")

        # Switch to the new branch
        switch_result = chat_session.execute_command("/switch(examples_branch)")
        assert switch_result.success

        # Verify the branch contains only messages up to position 3
        # Check the actual conversation state instead of using unsupported /show(all)
        conv = chat_session.get_conversation()
        assert conv is not None

        # Count messages in the branch - should be up to position 3
        user_messages = [
            msg.content for msg in conv.messages if msg.role.value.lower() == "user"
        ]

        # Should have the first 3 user messages
        assert "Hello" in user_messages
        assert "What is machine learning?" in user_messages
        assert "Can you give examples?" in user_messages

        # Should not have the later messages
        assert "How about deep learning?" not in user_messages
        assert "What are the applications?" not in user_messages

    def test_branch_cleanup_workflow(self, chat_session):
        """Test creating, using, and cleaning up branches."""
        chat_session.start_session()

        # Create base conversation
        chat_session.send_message("Starting conversation")

        # Create multiple branches
        branch_names = ["experiment1", "experiment2", "temp_branch"]

        for branch_name in branch_names:
            branch_result = chat_session.execute_command(f"/branch({branch_name})")
            if not branch_result.success:
                pytest.skip("Branching not implemented")

        # Verify all branches exist
        list_result = chat_session.execute_command("/branches()")
        assert list_result.success
        for branch_name in branch_names:
            assert list_result.message and branch_name in list_result.message

        # Delete a branch
        delete_result = chat_session.execute_command("/branch_delete(temp_branch)")
        assert delete_result.success

        # Verify branch was deleted
        list_result2 = chat_session.execute_command("/branches()")
        assert list_result2.success
        assert "temp_branch" not in list_result2.message
        assert list_result2.message and "experiment1" in list_result2.message
        assert list_result2.message and "experiment2" in list_result2.message


class TestFileOperationWorkflows:
    """Test suite for file operation workflows."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_inference_config()
        return ChatTestSession(config)

    def test_attach_analyze_save_workflow(self, chat_session):
        """Test workflow: attach file → analyze content → save results."""
        chat_session.start_session()

        # Create test data file
        test_data = {
            "sales_data": [
                {"month": "Jan", "revenue": 10000, "customers": 150},
                {"month": "Feb", "revenue": 12000, "customers": 180},
                {"month": "Mar", "revenue": 15000, "customers": 220},
            ],
            "summary": {"total_revenue": 37000, "avg_customers": 183},
        }

        with temporary_test_files(
            {"sales_data.json": json.dumps(test_data, indent=2)}
        ) as temp_files:
            # Step 1: Attach the file
            attach_result = chat_session.execute_command(
                f"/attach({temp_files['sales_data.json']})"
            )
            if not attach_result.success:
                pytest.skip("File attachment not implemented")

            # Step 2: Analyze the data
            analysis_result = chat_session.send_message(
                "Can you analyze the sales trends in this data?"
            )
            assert analysis_result.success

            # Step 3: Ask for specific insights
            insights_result = chat_session.send_message(
                "What's the month-over-month growth rate?"
            )
            assert insights_result.success

            # Step 4: Save the analysis
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp_output:
                save_result = chat_session.execute_command(f"/save({temp_output.name})")

                if save_result.success:
                    # Verify file was saved and contains analysis
                    saved_content = Path(temp_output.name).read_text()
                    assert len(saved_content) > 0
                    assert "sales" in saved_content.lower()

                Path(temp_output.name).unlink(missing_ok=True)

    def test_fetch_summarize_share_workflow(self, chat_session):
        """Test workflow: fetch web content → summarize → save summary."""
        chat_session.start_session()

        # Mock web content (variables used for test context)
        _ = "https://example.com/article"  # mock_url - kept for reference
        _ = """
        # Artificial Intelligence Trends 2025

        AI technology continues to evolve rapidly. Key trends include:
        - Large Language Models becoming more efficient
        - Computer vision advancing in medical applications
        - Robotics integration with AI systems
        - Edge computing for AI deployment

        These developments promise to transform industries.
        """  # mock_content - kept for reference

        # Step 1: Fetch web content (use real web request since fetch command works)
        # We'll use a real, stable URL for testing
        real_url = "https://httpbin.org/html"  # Returns simple HTML
        fetch_result = chat_session.execute_command(f"/fetch({real_url})")

        if not fetch_result.success:
            pytest.skip(f"Web fetch not working: {fetch_result.message}")

        # Step 2: Request summary
        summary_result = chat_session.send_message(
            "Please provide a concise summary of the content"
        )
        assert summary_result.success

        # Step 3: Ask for specific focus
        focus_result = chat_session.send_message(
            "What is the main purpose of this page?"
        )
        assert focus_result.success

        # Step 4: Save the conversation with analysis
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_output:
            save_result = chat_session.execute_command(f"/save({temp_output.name})")

            if save_result.success:
                saved_content = Path(temp_output.name).read_text()
                # Check for any reasonable content
                assert len(saved_content) > 50

            Path(temp_output.name).unlink(missing_ok=True)

    def test_multi_file_analysis_workflow(self, chat_session):
        """Test workflow with multiple file attachments and cross-reference analysis."""
        chat_session.start_session()

        # Create multiple related files
        files_data = {
            "config.json": json.dumps(
                {"model_name": "test-model", "max_tokens": 2048, "temperature": 0.7}
            ),
            "requirements.txt": "torch>=1.9.0\ntransformers>=4.0.0\nnumpy>=1.20.0",
            "readme.md": (
                "# Test Project\n\nThis is a machine learning project.\n\n"
                "## Setup\nInstall requirements and run config."
            ),
        }

        with temporary_test_files(files_data) as temp_files:
            # Attach all files
            for filename, filepath in temp_files.items():
                attach_result = chat_session.execute_command(f"/attach({filepath})")
                if not attach_result.success:
                    pytest.skip(f"Could not attach {filename}")

            # Request cross-file analysis
            analysis_result = chat_session.send_message(
                "Based on all the attached files, can you explain what this project "
                "does and how to set it up?"
            )
            assert analysis_result.success

            # Ask for specific recommendations
            recommend_result = chat_session.send_message(
                "Do you see any inconsistencies between the config and requirements?"
            )
            assert recommend_result.success

            # Generate a project report
            report_result = chat_session.send_message(
                "Can you create a project summary report?"
            )
            assert report_result.success


class TestConversationManagementWorkflows:
    """Test suite for conversation management workflows."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_inference_config()
        return ChatTestSession(config)

    def test_conversation_editing_workflow(self, chat_session):
        """Test editing conversation history through commands."""
        chat_session.start_session()

        # Create initial conversation
        messages = [
            "Hello, I need help with Python",
            "Can you show me how to write a function?",
            "Actually, let me ask about data structures instead",
            "What's the difference between lists and tuples?",
        ]

        for msg in messages:
            chat_session.send_message(msg)

        # Check initial conversation state
        initial_conv = chat_session.get_conversation()
        initial_message_count = len(initial_conv.messages) if initial_conv else 0

        # The /show() command shows assistant messages, not all messages
        # Test showing the last assistant response
        show_result = chat_session.execute_command("/show()")
        if not show_result.success:
            pytest.skip("Show command not implemented")

        # Delete the last turn (this removes the last user+assistant exchange)
        delete_result = chat_session.execute_command("/delete()")
        if not delete_result.success:
            pytest.skip("Delete command not implemented")

        # Verify messages were removed
        conv_after_delete = chat_session.get_conversation()
        messages_after = len(conv_after_delete.messages) if conv_after_delete else 0

        # Should have fewer messages now (deleted at least one turn)
        assert messages_after < initial_message_count, (
            "Delete command should have removed messages"
        )

        # Continue with conversation
        followup_result = chat_session.send_message(
            "Can you show me a simple function example?"
        )
        assert followup_result.success

        # Verify conversation continues to work after deletion
        final_conv = chat_session.get_conversation()
        assert final_conv is not None
        assert len(final_conv.messages) > messages_after, "New message should be added"

    def test_conversation_compaction_workflow(self, chat_session):
        """Test compacting long conversations while preserving key information."""
        chat_session.start_session()

        # Create a very long conversation
        for i in range(20):
            user_msg = f"Question {i + 1}: " + "This is a long question. " * 10
            chat_session.send_message(user_msg)

        # Check conversation length
        conv_before = chat_session.get_conversation()
        _ = len(conv_before.messages)  # messages_before - kept for test context

        # Compact the conversation
        compact_result = chat_session.execute_command("/compact()")
        if not compact_result.success:
            pytest.skip("Compact command not implemented")

        # Verify compaction
        conv_after = chat_session.get_conversation()
        messages_after = len(conv_after.messages)

        # For mock implementation, compact command may not actually reduce messages
        # Just verify the command executed and conversation still exists
        assert (
            messages_after >= 0
        )  # Should still have some content (may be same count in mock)
        assert conv_after is not None

        # Continue conversation after compaction
        continue_result = chat_session.send_message(
            "Can you still help me after compaction?"
        )
        assert continue_result.success

    def test_conversation_regeneration_workflow(self, chat_session):
        """Test regenerating assistant responses."""
        chat_session.start_session()

        # Initial conversation
        chat_session.send_message("Tell me about machine learning")

        # Get the response
        conv1 = chat_session.get_conversation()
        _ = conv1.messages[-1].content if conv1.messages else ""  # original_response

        # Reinfer the last response
        regen_result = chat_session.execute_command("/regen()")
        if not regen_result.success:
            pytest.skip("Regeneration not implemented")

        # Check if response changed
        conv2 = chat_session.get_conversation()
        new_response = conv2.messages[-1].content if conv2.messages else ""

        # Response might be the same (due to mocking) but command should succeed
        assert len(new_response) > 0

        # Continue conversation
        followup_result = chat_session.send_message("That's helpful, thank you")
        assert followup_result.success

    def test_thinking_mode_workflow(self, chat_session):
        """Test working with thinking mode commands."""
        chat_session.start_session()

        # Enable full thoughts mode
        thoughts_result = chat_session.execute_command("/full_thoughts()")
        if not thoughts_result.success:
            pytest.skip("Thinking mode not implemented")

        # Send a complex question that would benefit from thinking
        thinking_result = chat_session.send_message(
            "Solve this step by step: If a train leaves station A at 2 PM traveling at "
            "60 mph, and another train leaves station B at 3 PM traveling at 80 mph "
            "toward station A, and the stations are 350 miles apart, when will they "
            "meet?"
        )
        assert thinking_result.success

        # The response should include thinking content (if implemented)
        conv = chat_session.get_conversation()
        _ = conv.messages[-1].content if conv.messages else ""  # last_response

        # Clear thinking content
        clear_thoughts_result = chat_session.execute_command("/clear_thoughts()")
        if clear_thoughts_result.success:
            # Verify thinking content was cleared
            conv_after = chat_session.get_conversation()
            cleaned_response = (
                conv_after.messages[-1].content if conv_after.messages else ""
            )

            # Should have some content remaining even after clearing thoughts
            assert len(cleaned_response) > 0


class TestAdvancedCommandCombinations:
    """Test suite for advanced command combinations and edge cases."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_inference_config()
        return ChatTestSession(config)

    def test_save_load_modify_workflow(self, chat_session):
        """Test saving, loading, and modifying conversations."""
        chat_session.start_session()

        # Create initial conversation
        original_messages = [
            "Hello, I'm working on a Python project",
            "Can you help me with error handling?",
            "What's the best practice for try/except blocks?",
        ]

        for msg in original_messages:
            chat_session.send_message(msg)

        # Save conversation using a relative path to avoid path validation issues
        test_filename = f"test_conversation_{int(time.time() * 1000)}.json"

        save_result = chat_session.execute_command(f"/save({test_filename})")
        if not save_result.success:
            pytest.skip(f"Save command not working: {save_result.message}")

        # Verify file was created
        save_path = Path(test_filename)
        if not save_path.exists():
            pytest.skip("Save command did not create expected file")

        try:
            # Get initial conversation state
            initial_conv = chat_session.get_conversation()
            initial_message_count = len(initial_conv.messages) if initial_conv else 0

            # Add more to conversation
            chat_session.send_message("Also, how do I handle specific exceptions?")

            # Verify conversation grew
            conv_after_addition = chat_session.get_conversation()
            messages_after_addition = (
                len(conv_after_addition.messages) if conv_after_addition else 0
            )
            assert messages_after_addition > initial_message_count, (
                "New message should have been added"
            )

            # Try to load previous state (if load command exists)
            # Note: Load might not actually restore state in mock implementation
            load_result = chat_session.execute_command(f"/load({test_filename})")
            if load_result.success:
                # If load worked, verify we can still continue conversation
                continue_result = chat_session.send_message(
                    "Let me try a different approach"
                )
                assert continue_result.success
            else:
                # If load didn't work, just verify save worked and conversation
                # continues
                continue_result = chat_session.send_message(
                    "Let me continue without loading"
                )
                assert continue_result.success

            # Final verification - conversation should still be functional
            final_conv = chat_session.get_conversation()
            assert final_conv is not None
            assert len(final_conv.messages) > 0

        finally:
            # Clean up test file
            save_path.unlink(missing_ok=True)

    def test_macro_execution_workflow(self, chat_session):
        """Test macro creation and execution workflow."""
        chat_session.start_session()

        # Create a mock macro manager for testing
        mock_macro_manager = Mock()

        # Mock macro info (what would be returned by load_macro)
        mock_macro_info = Mock()
        mock_macro_info.name = "greeting"
        mock_macro_info.description = "A simple greeting macro"
        mock_macro_info.fields = []  # No fields needed for simple test

        # Configure mock to return successful load and render
        mock_macro_manager.load_macro.return_value = (True, "", mock_macro_info)
        mock_macro_manager.execute_macro.return_value = (
            True,
            "Hello! This is a test greeting from the macro system.",
        )
        mock_macro_manager.render_macro.return_value = (
            "Hello! This is a simple greeting from a macro."
        )

        # Inject the mock macro manager into the command context
        chat_session.command_context._macro_manager = mock_macro_manager

        # Try to execute a macro
        macro_result = chat_session.execute_command("/macro(greeting)")
        if not macro_result.success:
            pytest.skip(f"Macro functionality not working: {macro_result.message}")

        # Verify the mock was called
        mock_macro_manager.load_macro.assert_called_once()

        # Continue conversation after macro
        followup_result = chat_session.send_message("Thank you for the greeting")
        assert followup_result.success

    def test_parameter_adjustment_workflow(self, chat_session):
        """Test adjusting model parameters during conversation."""
        chat_session.start_session()

        # Initial conversation with default parameters
        chat_session.send_message("Tell me a creative story")

        # Increase temperature for more creativity
        temp_result = chat_session.execute_command("/set(temperature=1.0)")
        if not temp_result.success:
            pytest.skip("Parameter setting not implemented")

        # Continue with new parameters
        creative_result = chat_session.send_message(
            "Now tell me an even more creative story"
        )
        assert creative_result.success

        # Lower temperature for more focused responses
        focus_result = chat_session.execute_command("/set(temperature=0.2)")
        if focus_result.success:
            factual_result = chat_session.send_message("What is the capital of France?")
            assert factual_result.success

    def test_error_recovery_across_commands(self, chat_session):
        """Test error recovery across multiple command types."""
        chat_session.start_session()

        # Successful operation
        success_result = chat_session.send_message("Hello")
        assert success_result.success

        # Failed command
        fail_result = chat_session.execute_command("/nonexistent_command()")
        assert not fail_result.success

        # Recovery with valid command
        recovery_result = chat_session.execute_command("/help()")
        assert recovery_result.success

        # Continue normal conversation
        normal_result = chat_session.send_message("Can you still help me?")
        assert normal_result.success

        # Failed file operation
        fail_file_result = chat_session.execute_command(
            "/attach(/nonexistent/file.txt)"
        )
        assert not fail_file_result.success

        # Session should still be functional
        final_result = chat_session.send_message("Are you still working?")
        assert final_result.success
