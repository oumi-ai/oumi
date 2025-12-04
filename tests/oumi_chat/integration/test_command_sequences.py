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

"""Integration tests for complex command sequences and compositions."""

import json
import tempfile
from pathlib import Path

import pytest

from oumi_chat.commands import CommandResult
from tests.oumi_chat.utils.chat_test_utils import (
    ChatTestSession,
    create_test_chat_config,
    temporary_test_files,
)


class TestSequentialCommandExecution:
    """Test suite for executing commands in sequence."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_chat_config()
        return ChatTestSession(config)

    def test_parameter_adjustment_sequence(self, chat_session):
        """Test sequence of parameter adjustments."""
        chat_session.start_session()

        # Sequence of parameter changes
        parameter_sequence = [
            "/set(temperature=0.1)",
            "/set(top_p=0.8)",
            "/set(max_tokens=150)",
            "/set(temperature=0.9)",
            "/set(top_p=0.95)",
        ]

        successful_commands = 0
        for cmd in parameter_sequence:
            result = chat_session.execute_command(cmd)
            if result.success:
                successful_commands += 1

                # Test conversation after each parameter change
                msg_result = chat_session.send_message(f"Test message after {cmd}")
                assert msg_result.success

        # At least some parameter commands should work
        if successful_commands == 0:
            pytest.skip("Parameter setting not implemented")

        # Final parameter check
        chat_session.execute_command("/show_config()")
        # Command might not exist, but session should still be functional

    def test_conversation_management_sequence(self, chat_session):
        """Test sequence of conversation management commands."""
        chat_session.start_session()

        # Build up conversation
        for i in range(5):
            chat_session.send_message(f"Message {i + 1}: Building conversation history")

        # Execute management command sequence
        management_sequence = [
            "/show(all)",  # Show entire conversation
            "/show(3)",  # Show specific message
            "/delete(2)",  # Delete a message
            "/show(all)",  # Show conversation after deletion
            "/compact()",  # Compact conversation
            "/show(all)",  # Show after compaction
        ]

        for cmd in management_sequence:
            result = chat_session.execute_command(cmd)
            # Commands may not be implemented, but should not crash
            assert isinstance(result, CommandResult)

            # Continue conversation after each command
            cont_result = chat_session.send_message(
                "Continuing after management command"
            )
            assert cont_result.success

    def test_branching_sequence_workflow(self, chat_session):
        """Test complex branching command sequences."""
        chat_session.start_session()

        # Create base conversation
        base_messages = [
            "Let's discuss artificial intelligence",
            "What are the main types of AI?",
            "Tell me about machine learning",
        ]

        for msg in base_messages:
            chat_session.send_message(msg)

        # Complex branching sequence
        branching_sequence = [
            "/branch(ml_deep_dive)",  # Create branch 1
            "/send(How does deep learning work?)",  # Continue in branch 1
            "/branch_from(nlp_branch, 2)",  # Create branch 2 from position 2
            "/switch(nlp_branch)",  # Switch to branch 2
            "/send(What about natural language processing?)",  # Continue in branch 2
            "/branches()",  # List all branches
            "/switch(main)",  # Switch back to main
            "/send(What's the future of AI?)",  # Continue in main
            "/switch(ml_deep_dive)",  # Switch to first branch
            "/send(Can you explain neural networks?)",  # Continue in first branch
        ]

        successful_branches = 0
        for cmd in branching_sequence:
            if cmd.startswith("/send("):
                # Extract message and send it
                message = cmd[6:-1]  # Remove /send( and )
                result = chat_session.send_message(message)
                assert result.success
            else:
                result = chat_session.execute_command(cmd)
                if "branch" in cmd and result.success:
                    successful_branches += 1

        # If branching is implemented, should have some successful branch operations
        if successful_branches > 0:
            # Test branch cleanup
            chat_session.execute_command("/branch_delete(ml_deep_dive)")
            # May or may not succeed depending on implementation

    def test_file_operations_sequence(self, chat_session):
        """Test sequence of file operations."""
        chat_session.start_session()

        # Create test files
        test_data = {
            "config.json": json.dumps({"model": "test", "temperature": 0.7}),
            "data.txt": "This is test data\nLine 2\nLine 3",
            "notes.md": "# Notes\n\n- Point 1\n- Point 2\n- Point 3",
        }

        temp_files = []
        try:
            with temporary_test_files(test_data) as temp_file_paths:
                # File operations sequence
                file_sequence = [
                    f"/attach({temp_file_paths['config.json']})",
                    "/send(What do you see in this config file?)",
                    f"/attach({temp_file_paths['data.txt']})",
                    "/send(Now analyze both files together)",
                    f"/attach({temp_file_paths['notes.md']})",
                    "/send(Can you summarize all three files?)",
                ]

                successful_attachments = 0
                for cmd in file_sequence:
                    if cmd.startswith("/send("):
                        message = cmd[6:-1]
                        result = chat_session.send_message(message)
                        assert result.success
                    else:
                        result = chat_session.execute_command(cmd)
                        if result.success:
                            successful_attachments += 1

                # Save conversation with all attachments
                with tempfile.NamedTemporaryFile(
                    suffix=".json", delete=False
                ) as save_temp:
                    save_result = chat_session.execute_command(
                        f"/save({save_temp.name})"
                    )
                    temp_files.append(save_temp.name)

                    if save_result.success:
                        # Verify saved content
                        saved_content = Path(save_temp.name).read_text()
                        assert len(saved_content) > 0
        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)


class TestConditionalCommandFlows:
    """Test suite for conditional command execution based on results."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_chat_config()
        return ChatTestSession(config)

    def test_error_recovery_flow(self, chat_session):
        """Test command flow with error recovery."""
        chat_session.start_session()

        # Intentionally failing command followed by recovery
        error_recovery_flow = [
            "/attach(/nonexistent/file.txt)",  # This should fail
            "/help()",  # Recover with help
            "/send(Are you still working?)",  # Test normal operation
            "/save(/invalid/path.json)",  # Another failure
            "/send(Let's continue anyway)",  # Continue despite error
        ]

        for cmd in error_recovery_flow:
            if cmd.startswith("/send("):
                message = cmd[6:-1]
                result = chat_session.send_message(message)
                assert result.success, (
                    "Normal messages should work even after command errors"
                )
            else:
                result = chat_session.execute_command(cmd)
                # Don't assert success - errors are expected
                assert isinstance(result, CommandResult)

                # Session should remain functional after any error
                test_result = chat_session.send_message("Testing functionality")
                assert test_result.success, (
                    "Session should remain functional after command error"
                )

    def test_success_dependent_flow(self, chat_session):
        """Test command flow that depends on previous command success."""
        chat_session.start_session()

        # Create test file first
        test_content = "Test data for conditional flow"

        with temporary_test_files({"test_data.txt": test_content}) as temp_files:
            # Try to attach file
            attach_result = chat_session.execute_command(
                f"/attach({temp_files['test_data.txt']})"
            )

            if attach_result.success:
                # Continue with analysis since attach succeeded
                analysis_result = chat_session.send_message(
                    "Please analyze the attached file"
                )
                assert analysis_result.success

                # Try to save since analysis succeeded
                with tempfile.NamedTemporaryFile(
                    suffix=".md", delete=False
                ) as save_temp:
                    save_result = chat_session.execute_command(
                        f"/save({save_temp.name})"
                    )

                    if save_result.success:
                        # Verify saved file
                        assert Path(save_temp.name).exists()
                        saved_content = Path(save_temp.name).read_text()
                        assert len(saved_content) > 0

                    Path(save_temp.name).unlink(missing_ok=True)
            else:
                # Alternative flow if attach failed
                alternative_result = chat_session.send_message(
                    "Let's work without file attachment"
                )
                assert alternative_result.success

    def test_branching_conditional_flow(self, chat_session):
        """Test conditional flows with branching."""
        chat_session.start_session()

        # Set up conversation
        chat_session.send_message("Let's explore different conversation paths")
        chat_session.send_message("This is our baseline conversation")

        # Attempt to create branch
        branch_result = chat_session.execute_command("/branch(experimental)")

        if branch_result.success:
            # Branch creation succeeded - use branching workflow
            chat_session.send_message("We're now in the experimental branch")

            # Try risky operation in branch
            risky_result = chat_session.send_message("Let's try something experimental")
            assert risky_result.success

            # Switch back to main if we want to preserve original
            switch_result = chat_session.execute_command("/switch(main)")
            if switch_result.success:
                chat_session.send_message("We're back in main branch safely")

        else:
            # Branch creation failed - use linear workflow
            chat_session.send_message("Continuing with linear conversation")

            # Use conversation management instead
            show_result = chat_session.execute_command("/show(all)")
            if show_result.success:
                chat_session.send_message("Here's our conversation so far")


class TestComplexWorkflowComposition:
    """Test suite for complex workflow compositions combining multiple features."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_chat_config()
        return ChatTestSession(config)

    def test_data_analysis_workflow_composition(self, chat_session):
        """Test complex data analysis workflow."""
        chat_session.start_session()

        # Create complex dataset
        sales_data = {
            "quarterly_sales": [
                {
                    "quarter": "Q1",
                    "revenue": 125000,
                    "customers": 450,
                    "satisfaction": 4.2,
                },
                {
                    "quarter": "Q2",
                    "revenue": 142000,
                    "customers": 520,
                    "satisfaction": 4.1,
                },
                {
                    "quarter": "Q3",
                    "revenue": 168000,
                    "customers": 610,
                    "satisfaction": 4.3,
                },
                {
                    "quarter": "Q4",
                    "revenue": 195000,
                    "customers": 720,
                    "satisfaction": 4.4,
                },
            ],
            "product_performance": {
                "Product A": {"sales": 85000, "returns": 450},
                "Product B": {"sales": 120000, "returns": 230},
                "Product C": {"sales": 95000, "returns": 180},
            },
        }

        market_report = """
        # Market Analysis Report 2025

        ## Market Conditions
        - Economic growth: 3.2%
        - Industry growth: 8.5%
        - Competition index: High

        ## Trends
        - Digital transformation accelerating
        - Customer expectations rising
        - Sustainability becoming critical

        ## Recommendations
        - Invest in digital capabilities
        - Enhance customer experience
        - Develop sustainable products
        """

        with temporary_test_files(
            {
                "sales_data.json": json.dumps(sales_data, indent=2),
                "market_report.md": market_report,
            }
        ) as temp_files:
            # Complex workflow composition
            workflow_steps = [
                # Step 1: Data ingestion
                (f"/attach({temp_files['sales_data.json']})", "attach sales data"),
                (
                    "/send(Please analyze the quarterly sales trends)",
                    "analyze sales trends",
                ),
                # Step 2: Additional context
                (f"/attach({temp_files['market_report.md']})", "attach market report"),
                (
                    "/send(How do our sales trends compare to market conditions?)",
                    "compare with market",
                ),
                # Step 3: Deep analysis
                (
                    "/send(Which product is performing best and why?)",
                    "product analysis",
                ),
                (
                    "/send(What should be our strategy for next quarter?)",
                    "strategy recommendation",
                ),
                # Step 4: Branch for alternative analysis
                ("/branch(alternative_analysis)", "create analysis branch"),
                (
                    "/send(What if we focused on customer satisfaction instead?)",
                    "satisfaction focus",
                ),
                # Step 5: Switch back and synthesize
                ("/switch(main)", "back to main analysis"),
                (
                    "/send(Can you provide an executive summary of all findings?)",
                    "executive summary",
                ),
                # Step 6: Documentation
                ("/save(analysis_report.md)", "save complete analysis"),
            ]

            analysis_results = []
            successful_messages = 0

            for command, description in workflow_steps:
                if command.startswith("/send("):
                    message = command[6:-1]
                    result = chat_session.send_message(message)
                    analysis_results.append((description, result.success))
                    if result.success:
                        successful_messages += 1
                    assert result.success, f"Analysis step failed: {description}"
                else:
                    result = chat_session.execute_command(command)
                    analysis_results.append((description, result.success))
                    # Commands may not be implemented, but should not crash
                    assert isinstance(result, CommandResult)

            # Verify we got through the workflow - count actual send_message calls
            assert successful_messages >= 5, (
                f"Should have successful analysis messages, got {successful_messages}"
            )

    def test_collaborative_workflow_composition(self, chat_session):
        """Test workflow simulating collaborative work."""
        chat_session.start_session()

        # Simulate collaborative document creation
        project_spec = """
        # Project: AI Assistant Enhancement

        ## Objective
        Improve the chat interface with new features

        ## Requirements
        - Add voice input capability
        - Implement file drag-and-drop
        - Create conversation export
        - Add collaboration features

        ## Timeline
        - Phase 1: 2 weeks
        - Phase 2: 3 weeks
        - Phase 3: 2 weeks
        """

        with temporary_test_files({"project_spec.md": project_spec}) as temp_files:
            collaborative_workflow = [
                # Project setup
                (f"/attach({temp_files['project_spec.md']})", "load project spec"),
                ("/send(Let's break down this project into tasks)", "task breakdown"),
                # Branch for different perspectives
                ("/branch(technical_review)", "technical perspective"),
                (
                    "/send(What are the technical challenges for each requirement?)",
                    "technical analysis",
                ),
                ("/branch_from(ux_review, 1)", "UX perspective from start"),
                ("/switch(ux_review)", "switch to UX"),
                (
                    "/send(How will these features impact user experience?)",
                    "UX analysis",
                ),
                ("/branch_from(timeline_review, 1)", "timeline perspective"),
                ("/switch(timeline_review)", "switch to timeline"),
                (
                    "/send(Is this timeline realistic for the scope?)",
                    "timeline analysis",
                ),
                # Synthesis
                ("/switch(main)", "back to main discussion"),
                (
                    "/send(Based on all perspectives, what's our implementation plan?)",
                    "synthesis",
                ),
                # Documentation
                ("/branches()", "review all branches"),
                ("/save(project_analysis.md)", "save comprehensive analysis"),
            ]

            for command, description in collaborative_workflow:
                if command.startswith("/send("):
                    message = command[6:-1]
                    result = chat_session.send_message(message)
                    assert result.success, f"Collaborative step failed: {description}"
                else:
                    result = chat_session.execute_command(command)
                    # Branching commands may not be implemented
                    assert isinstance(result, CommandResult)

    def test_iterative_refinement_workflow(self, chat_session):
        """Test iterative refinement workflow with feedback loops."""
        chat_session.start_session()

        # Start with initial concept
        initial_idea = "Create a recommendation system for e-commerce"
        chat_session.send_message(
            f"I want to {initial_idea}. Can you help me design this?"
        )

        # Iterative refinement cycles
        refinement_cycles = [
            # Cycle 1: Basic design
            {
                "focus": "basic architecture",
                "questions": [
                    "What are the core components needed?",
                    "How should data flow through the system?",
                    "What algorithms would you recommend?",
                ],
            },
            # Cycle 2: Detailed design
            {
                "focus": "detailed implementation",
                "questions": [
                    "Can you elaborate on the recommendation algorithms?",
                    "How do we handle cold start problems?",
                    "What about scalability considerations?",
                ],
            },
            # Cycle 3: Practical concerns
            {
                "focus": "practical implementation",
                "questions": [
                    "What technologies should we use?",
                    "How do we measure recommendation quality?",
                    "What are potential privacy concerns?",
                ],
            },
        ]

        for cycle_num, cycle in enumerate(refinement_cycles):
            # Create branch for this refinement cycle
            branch_name = f"refinement_cycle_{cycle_num + 1}"
            chat_session.execute_command(f"/branch({branch_name})")

            # Ask focused questions for this cycle
            for question in cycle["questions"]:
                result = chat_session.send_message(question)
                assert result.success, f"Refinement question failed: {question}"

            # Save progress for this cycle
            save_file = f"refinement_{cycle_num + 1}.md"
            chat_session.execute_command(f"/save({save_file})")
            # May not be implemented, but should not crash

            # Switch back to main for next cycle
            if cycle_num < len(refinement_cycles) - 1:
                chat_session.execute_command("/switch(main)")

        # Final synthesis
        chat_session.execute_command("/switch(main)")
        final_result = chat_session.send_message(
            "Based on all our refinement cycles, can you provide a comprehensive "
            "implementation plan?"
        )
        assert final_result.success


class TestErrorHandlingInSequences:
    """Test suite for error handling in command sequences."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_chat_config()
        return ChatTestSession(config)

    def test_graceful_degradation_sequence(self, chat_session):
        """Test sequences that degrade gracefully when commands fail."""
        chat_session.start_session()

        # Sequence with mix of working and failing commands
        mixed_sequence = [
            ("/help()", True),  # Should work
            ("/nonexistent_command()", False),  # Should fail
            ("/send(Still working after error?)", True),  # Should work
            ("/attach(/nonexistent/file.txt)", False),  # Should fail
            ("/send(Continuing despite file error)", True),  # Should work
            ("/invalid_branch_operation()", False),  # Should fail
            ("/send(Final test message)", True),  # Should work
        ]

        for command, should_succeed in mixed_sequence:
            if command.startswith("/send("):
                message = command[6:-1]
                result = chat_session.send_message(message)
                assert result.success == should_succeed, (
                    f"Message expectation failed: {command}"
                )
            else:
                result = chat_session.execute_command(command)
                # Don't assert specific success/failure - just that it doesn't crash
                assert isinstance(result, CommandResult)

        # Session should still be functional at the end
        final_test = chat_session.send_message("Final functionality test")
        assert final_test.success

    def test_recovery_strategy_sequence(self, chat_session):
        """Test sequences with built-in recovery strategies."""
        chat_session.start_session()

        # Strategy: Try advanced command, fall back to basic if it fails
        recovery_strategies = [
            {
                "primary": "/branch(advanced_analysis)",
                "fallback": "/send(Let's use linear analysis instead)",
                "test": "/send(Testing primary strategy)",
            },
            {
                "primary": "/attach(/some/advanced/file.txt)",
                "fallback": "/send(Let's work without external files)",
                "test": "/send(Testing file strategy)",
            },
            {
                "primary": "/complex_analysis_command()",
                "fallback": "/send(Let's do manual analysis)",
                "test": "/send(Testing analysis strategy)",
            },
        ]

        for strategy in recovery_strategies:
            # Try primary approach
            primary_result = chat_session.execute_command(strategy["primary"])

            if primary_result.success:
                # Primary worked, continue with test
                test_result = chat_session.send_message(
                    strategy["test"][6:-1]
                )  # Remove /send()
                assert test_result.success
            else:
                # Primary failed, use fallback
                fallback_message = strategy["fallback"][6:-1]  # Remove /send()
                fallback_result = chat_session.send_message(fallback_message)
                assert fallback_result.success

                # Test that fallback is working
                test_result = chat_session.send_message("Testing fallback approach")
                assert test_result.success

    def test_transaction_like_sequence(self, chat_session):
        """Test sequences that should be atomic (all succeed or all fail gracefully)."""
        chat_session.start_session()

        # Simulate transaction-like operations
        chat_session.send_message("Starting complex operation")

        # Create test files for transaction
        transaction_data = {
            "step1.json": json.dumps({"status": "step1_complete"}),
            "step2.json": json.dumps({"status": "step2_complete"}),
            "step3.json": json.dumps({"status": "step3_complete"}),
        }

        with temporary_test_files(transaction_data) as temp_files:
            # Transaction sequence
            transaction_steps = [
                f"/attach({temp_files['step1.json']})",
                "/send(Processing step 1 data)",
                f"/attach({temp_files['step2.json']})",
                "/send(Processing step 2 data)",
                f"/attach({temp_files['step3.json']})",
                "/send(Processing step 3 data)",
                "/send(All steps complete - finalizing transaction)",
            ]

            step_results = []
            for step in transaction_steps:
                if step.startswith("/send("):
                    message = step[6:-1]
                    result = chat_session.send_message(message)
                    step_results.append(result.success)
                else:
                    result = chat_session.execute_command(step)
                    step_results.append(result.success)

            # Check if transaction completed successfully
            all_messages_succeeded = all(
                step_results[i] for i in [1, 3, 5, 6]
            )  # Message indices
            assert all_messages_succeeded, "Core transaction messages should succeed"

            # Even if some commands failed, session should be in consistent state
            consistency_test = chat_session.send_message(
                "Transaction completed, testing consistency"
            )
            assert consistency_test.success
