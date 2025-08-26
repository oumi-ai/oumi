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

"""Chat fuzzing tests for stress testing and edge case discovery."""

import random
import time
from pathlib import Path
from typing import List
import tempfile

import pytest

from oumi.core.types.conversation import Role
from tests.markers import requires_cuda_initialized, requires_gpus
from tests.utils.chat_real_model_utils import (
    RealModelChatSession,
    create_real_model_inference_config,
    create_fuzzing_conversation_prompts,
    temporary_chat_files,
    ChatPerformanceMonitor
)


@pytest.mark.chat_fuzzing
@pytest.mark.e2e
class TestChatFuzzingBasic:
    """Basic fuzzing tests that can run on any system."""

    def test_rapid_message_sequence(self):
        """Test sending many messages rapidly to find race conditions."""
        config = create_real_model_inference_config(
            max_new_tokens=10,  # Keep responses short for speed
            temperature=0.0     # Deterministic for reproducibility
        )
        chat_session = RealModelChatSession(config)
        
        prompts = [
            "Hello",
            "What is 2+2?", 
            "Tell me about science",
            "Help me understand",
            "Explain briefly"
        ]
        
        with chat_session.real_inference_session():
            chat_session.start_session()
            
            responses = []
            for i in range(20):  # Rapid sequence
                prompt = prompts[i % len(prompts)]
                result = chat_session.send_message_with_real_inference(f"{prompt} (round {i+1})")
                responses.append(result)
                
                # Very brief pause to avoid overwhelming
                time.sleep(0.01)
            
            # Validate that most responses succeeded
            successful_responses = [r for r in responses if r.success]
            assert len(successful_responses) >= len(responses) * 0.8  # At least 80% success rate
            
            # Check conversation state integrity
            conversation = chat_session.get_conversation()
            assert conversation is not None
            assert len(conversation.messages) >= len(successful_responses)

    def test_command_chaos_sequence(self):
        """Test mixing random commands with conversation."""
        config = create_real_model_inference_config(max_new_tokens=10)
        chat_session = RealModelChatSession(config)
        
        commands = [
            "/help()",
            "/clear()", 
            "/show()",
            "/invalid_command()",
        ]
        
        with temporary_chat_files({"fuzz_test.txt": "test content"}) as temp_files:
            temp_path = temp_files["fuzz_test.txt"]
            save_commands = [
                f"/save({temp_path}.1.json)",
                f"/save({temp_path}.2.txt)", 
            ]
            
            with chat_session.real_inference_session():
                chat_session.start_session()
                
                actions_taken = []
                for i in range(30):
                    if i % 3 == 0:
                        # Send a message
                        result = chat_session.send_message_with_real_inference(f"Message {i}")
                        actions_taken.append(("message", result.success))
                    elif i % 3 == 1:
                        # Execute a random command
                        if i < 20:  # Don't save too many files
                            cmd = random.choice(commands + save_commands)
                        else:
                            cmd = random.choice(commands)
                        result = chat_session.inject_command(cmd)
                        actions_taken.append(("command", result.success))
                    else:
                        # Brief pause
                        time.sleep(0.05)
                        actions_taken.append(("pause", True))
                
                # Verify session remained functional
                final_result = chat_session.send_message_with_real_inference("Final test message")
                assert final_result.success or "no active session" in final_result.message.lower()
                
                # Check that we had a good mix of actions
                messages = [a for a in actions_taken if a[0] == "message"]
                commands = [a for a in actions_taken if a[0] == "command"]
                assert len(messages) >= 8
                assert len(commands) >= 8

    def test_edge_case_inputs(self):
        """Test various edge case inputs that might break the chat system."""
        config = create_real_model_inference_config(max_new_tokens=15)
        chat_session = RealModelChatSession(config)
        
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "Hello" * 100,  # Very long repetition
            "What is 2+2?" + "\n" * 50,  # Lots of newlines
            "Special chars: !@#$%^&*()_+{}|:<>?[]\\;'\",./-=~`",
            "Unicode: 你好 🌟 🤖 café naïve résumé",
            "Mixed: Hello 世界 🎉 How are you? 🤔",
            "/fake/command/that/looks/like/path",
            "SELECT * FROM users; -- SQL injection attempt",
            "<script>alert('xss')</script>",  # XSS-like
        ]
        
        with chat_session.real_inference_session():
            chat_session.start_session()
            
            results = []
            for i, edge_case in enumerate(edge_cases):
                try:
                    result = chat_session.send_message_with_real_inference(
                        f"Test {i}: {edge_case}"
                    )
                    results.append(("success" if result.success else "graceful_failure", edge_case))
                except Exception as e:
                    results.append(("exception", str(e)))
                    
                # Brief pause between edge cases
                time.sleep(0.1)
            
            # Verify system handled edge cases gracefully (no crashes)
            exceptions = [r for r in results if r[0] == "exception"]
            assert len(exceptions) == 0, f"Unexpected exceptions: {exceptions}"
            
            # Verify session is still functional
            final_test = chat_session.send_message_with_real_inference("Are you still working?")
            assert final_test.success

    def test_branch_explosion(self):
        """Test creating many conversation branches rapidly."""
        config = create_real_model_inference_config(max_new_tokens=8)
        chat_session = RealModelChatSession(config)
        
        with chat_session.real_inference_session():
            chat_session.start_session()
            
            # Initial conversation
            result = chat_session.send_message_with_real_inference("Hello, let's start a conversation")
            assert result.success
            
            # Attempt to create multiple branches
            branch_results = []
            for i in range(10):
                try:
                    # Try to branch (this might not work if branching isn't implemented)
                    branch_result = chat_session.inject_command(f"/branch(branch_{i})")
                    branch_results.append(branch_result.success)
                    
                    # Send a message in this context
                    msg_result = chat_session.send_message_with_real_inference(f"Branch {i} message")
                    
                    time.sleep(0.05)
                except Exception:
                    # Branching might not be implemented, which is fine for this test
                    pass
            
            # The main goal is to ensure the system doesn't crash with branch attempts
            final_result = chat_session.send_message_with_real_inference("Final message after branch attempts")
            assert final_result.success


@pytest.mark.chat_fuzzing 
@pytest.mark.e2e
@pytest.mark.single_gpu
@requires_cuda_initialized()
class TestChatFuzzingGPU:
    """GPU-accelerated fuzzing tests for more intensive scenarios."""

    def test_extended_conversation_stress(self):
        """Test very long conversation to stress memory and context management."""
        config = create_real_model_inference_config(
            engine_type="VLLM",  # Use vLLM for better performance
            max_new_tokens=15,
            model_max_length=2048  # Reasonable context window
        )
        chat_session = RealModelChatSession(config)
        
        try:
            # Skip if vLLM not available
            import vllm
        except ImportError:
            pytest.skip("vLLM not available for extended stress testing")
        
        prompts = create_fuzzing_conversation_prompts(100)
        monitor = ChatPerformanceMonitor()
        
        with chat_session.real_inference_session():
            monitor.start_session_monitoring()
            chat_session.start_session()
            
            successful_exchanges = 0
            memory_errors = 0
            
            for i, prompt in enumerate(prompts[:50]):  # Test 50 exchanges
                try:
                    result = chat_session.send_message_with_real_inference(
                        f"Turn {i+1}: {prompt}"
                    )
                    
                    if result.success:
                        successful_exchanges += 1
                        
                        # Validate response periodically
                        if i % 10 == 0:
                            validation = chat_session.validate_last_response()
                            assert validation.get("basic_validation", False)
                    
                    # Brief pause to allow monitoring
                    time.sleep(0.1)
                    
                except Exception as e:
                    if "memory" in str(e).lower() or "cuda" in str(e).lower():
                        memory_errors += 1
                        if memory_errors > 5:
                            break  # Stop if too many memory errors
                    else:
                        raise
            
            # End monitoring
            metrics = monitor.end_session_monitoring(chat_session)
            
            # Verify reasonable success rate
            assert successful_exchanges >= 30, f"Only {successful_exchanges} successful exchanges"
            
            # Check performance metrics
            assert metrics["session_duration"] > 0
            assert "avg_response_time" in chat_session.get_performance_summary()
            
            # Verify conversation integrity
            conversation = chat_session.get_conversation()
            if conversation:
                # Should have many messages
                assert len(conversation.messages) >= successful_exchanges

    def test_concurrent_session_simulation(self):
        """Simulate multiple concurrent chat sessions (sequential for testing)."""
        configs = [
            create_real_model_inference_config(max_new_tokens=10, temperature=0.0),
            create_real_model_inference_config(max_new_tokens=12, temperature=0.1),
            create_real_model_inference_config(max_new_tokens=8, temperature=0.2),
        ]
        
        sessions = []
        session_results = []
        
        # Create and run multiple sessions
        for i, config in enumerate(configs):
            session = RealModelChatSession(config)
            sessions.append(session)
            
            with session.real_inference_session():
                session.start_session()
                
                # Send messages in this session
                session_messages = []
                for j in range(5):
                    result = session.send_message_with_real_inference(
                        f"Session {i+1}, message {j+1}: Tell me about topic {j}"
                    )
                    session_messages.append(result.success)
                    time.sleep(0.1)
                
                session_results.append(session_messages)
                
                # Get performance data
                perf = session.get_performance_summary()
                assert "total_responses" in perf
        
        # Verify all sessions worked independently
        for i, results in enumerate(session_results):
            successful_in_session = sum(1 for r in results if r)
            assert successful_in_session >= 3, f"Session {i} had too many failures"

    def test_file_system_stress(self):
        """Test heavy file operations during chat."""
        config = create_real_model_inference_config(max_new_tokens=10)
        chat_session = RealModelChatSession(config)
        
        # Create many temporary files for testing
        temp_dir = Path(tempfile.mkdtemp(prefix="chat_fuzz_"))
        try:
            with chat_session.real_inference_session():
                chat_session.start_session()
                
                files_created = []
                save_successes = 0
                
                for i in range(20):
                    # Send a message
                    result = chat_session.send_message_with_real_inference(
                        f"Message {i}: Tell me something interesting."
                    )
                    
                    if result.success:
                        # Try to save the conversation frequently
                        save_path = temp_dir / f"chat_save_{i}.json"
                        save_result = chat_session.inject_command(f"/save({save_path})")
                        
                        if save_result.success:
                            files_created.append(save_path)
                            save_successes += 1
                        
                        # Also test other file commands
                        if i % 5 == 0:
                            export_path = temp_dir / f"export_{i}.txt"
                            export_result = chat_session.inject_command(f"/save({export_path})")
                            if export_result.success:
                                files_created.append(export_path)
                    
                    time.sleep(0.05)
                
                # Verify files were created
                existing_files = [f for f in files_created if f.exists()]
                assert len(existing_files) >= save_successes * 0.8  # Most files should exist
                
                # Verify file contents are reasonable
                for file_path in existing_files[:5]:  # Check first few files
                    content = file_path.read_text()
                    assert len(content) > 10  # Should have substantial content
        
        finally:
            # Cleanup temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors


@pytest.mark.chat_fuzzing
@pytest.mark.e2e_eternal
class TestChatFuzzingExtreme:
    """Extremely long-running fuzzing tests for comprehensive stress testing."""

    def test_marathon_conversation(self):
        """Test extremely long conversation (hundreds of turns)."""
        config = create_real_model_inference_config(
            max_new_tokens=5,  # Very short responses for speed
            temperature=0.0    # Deterministic
        )
        chat_session = RealModelChatSession(config)
        
        monitor = ChatPerformanceMonitor()
        prompts = create_fuzzing_conversation_prompts(500)
        
        with chat_session.real_inference_session():
            monitor.start_session_monitoring()
            chat_session.start_session()
            
            milestone_intervals = [50, 100, 200, 300]
            milestones_reached = []
            
            for i, prompt in enumerate(prompts[:300]):  # Up to 300 turns
                try:
                    result = chat_session.send_message_with_real_inference(
                        f"Turn {i+1}: {prompt[:50]}"  # Truncate prompts
                    )
                    
                    # Track milestones
                    if (i + 1) in milestone_intervals:
                        milestones_reached.append(i + 1)
                        
                        # Validate system health at milestones
                        validation = chat_session.validate_last_response()
                        assert validation.get("basic_validation", True)  # Should still work
                        
                        # Check memory usage
                        perf = chat_session.get_performance_summary()
                        if "peak_memory_mb" in perf:
                            assert perf["peak_memory_mb"] < 8000  # Reasonable memory limit
                    
                    # Very brief pause
                    time.sleep(0.02)
                    
                except Exception as e:
                    # Stop on serious errors but log how far we got
                    if "memory" in str(e).lower() or "cuda" in str(e).lower():
                        break
                    else:
                        # For other errors, continue
                        continue
            
            metrics = monitor.end_session_monitoring(chat_session)
            
            # Verify we reached significant milestones
            assert len(milestones_reached) >= 2, f"Only reached milestones: {milestones_reached}"
            
            # Final system health check
            final_result = chat_session.send_message_with_real_inference("Final health check")
            assert final_result.success or "memory" in final_result.message.lower()

    def test_chaos_monkey_simulation(self):
        """Simulate chaotic user behavior with random actions."""
        config = create_real_model_inference_config(max_new_tokens=8)
        chat_session = RealModelChatSession(config)
        
        # Define possible actions with weights
        actions = [
            ("message", 0.4),      # 40% chance of sending message
            ("command", 0.3),      # 30% chance of command
            ("pause", 0.2),        # 20% chance of pause
            ("restart", 0.1),      # 10% chance of restarting session
        ]
        
        commands = [
            "/help()",
            "/clear()", 
            "/show()",
            "/invalid_test()",
        ]
        
        with temporary_chat_files({"chaos.txt": "chaos content"}) as temp_files:
            save_path = temp_files["chaos.txt"] + ".save.json"
            
            total_actions = 0
            successful_messages = 0
            session_restarts = 0
            
            with chat_session.real_inference_session():
                chat_session.start_session()
                
                for round_num in range(100):  # 100 rounds of chaos
                    # Choose random action
                    rand_val = random.random()
                    cumulative_weight = 0
                    chosen_action = "message"  # default
                    
                    for action, weight in actions:
                        cumulative_weight += weight
                        if rand_val <= cumulative_weight:
                            chosen_action = action
                            break
                    
                    total_actions += 1
                    
                    try:
                        if chosen_action == "message":
                            prompts = ["Hello", "What?", "Tell me", "Help", "Explain"]
                            msg = f"{random.choice(prompts)} (round {round_num})"
                            result = chat_session.send_message_with_real_inference(msg)
                            if result.success:
                                successful_messages += 1
                        
                        elif chosen_action == "command":
                            cmd = random.choice(commands + [f"/save({save_path})"])
                            chat_session.inject_command(cmd)
                        
                        elif chosen_action == "pause":
                            time.sleep(random.uniform(0.01, 0.1))
                        
                        elif chosen_action == "restart":
                            if chat_session.is_active():
                                chat_session.end_session()
                            chat_session.start_session()
                            session_restarts += 1
                        
                        # Random micro-pause
                        if random.random() < 0.3:
                            time.sleep(0.01)
                    
                    except Exception as e:
                        # Log but continue with chaos
                        if "memory" in str(e).lower():
                            break  # Stop on memory issues
                        continue
                
                # Verify system survived the chaos
                assert total_actions >= 80
                assert successful_messages >= 20  # At least some messages worked
                
                # Final health check
                if not chat_session.is_active():
                    chat_session.start_session()
                
                final_result = chat_session.send_message_with_real_inference("Chaos test complete")
                assert final_result.success or "session" in final_result.message.lower()


class TestFuzzingUtilities:
    """Test the fuzzing utility functions themselves."""
    
    def test_fuzzing_prompt_generation(self):
        """Test the fuzzing prompt generation utility."""
        prompts = create_fuzzing_conversation_prompts(20)
        
        assert len(prompts) == 20
        assert all(isinstance(p, str) for p in prompts)
        assert all(len(p.strip()) > 0 for p in prompts)
        
        # Check for diversity
        unique_prompts = set(prompts)
        assert len(unique_prompts) >= len(prompts) * 0.7  # At least 70% unique
        
        # Check for keyword instructions in some prompts
        keyword_prompts = [p for p in prompts if "include the word" in p.lower()]
        assert len(keyword_prompts) >= 3  # Should have some keyword instructions
    
    def test_fuzzing_prompt_generation_large(self):
        """Test generating a large number of fuzzing prompts."""
        prompts = create_fuzzing_conversation_prompts(100)
        
        assert len(prompts) == 100
        
        # Check that we have good variety even with large numbers
        unique_base_topics = set()
        for prompt in prompts:
            if "science" in prompt.lower():
                unique_base_topics.add("science")
            elif "computer" in prompt.lower():
                unique_base_topics.add("computers")
            elif "mathematics" in prompt.lower():
                unique_base_topics.add("math")
            # Add more topic detection as needed
        
        assert len(unique_base_topics) >= 3  # Should have variety in topics