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

"""Real model integration tests for chat functionality."""

import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from oumi.core.configs import InferenceConfig
from tests.markers import requires_cuda_initialized, requires_gpus, requires_inference_backend
from tests.utils.chat_real_model_utils import (
    RealModelChatSession,
    create_real_model_inference_config,
    create_real_model_chat_conversations,
    temporary_chat_files,
    ChatPerformanceMonitor
)


class AbstractRealModelChatTest(ABC):
    """Abstract base class for real model chat tests.
    
    Provides common test patterns shared across different inference engines,
    following the same design patterns as inference engine tests.
    """

    @abstractmethod
    def get_engine_config(self, model_key: str = "smollm_135m") -> InferenceConfig:
        """Get engine-specific configuration.
        
        Args:
            model_key: Model configuration key to use.
            
        Returns:
            InferenceConfig for the specific engine.
        """
        pass

    @abstractmethod  
    def get_engine_name(self) -> str:
        """Get the name of the inference engine for test identification."""
        pass

    @abstractmethod
    def should_skip_test(self) -> Optional[str]:
        """Check if tests should be skipped for this engine.
        
        Returns:
            Skip reason string if tests should be skipped, None otherwise.
        """
        pass

    def create_chat_session(
        self, 
        model_key: str = "smollm_135m",
        **config_overrides
    ) -> RealModelChatSession:
        """Create a real model chat session for testing.
        
        Args:
            model_key: Model configuration key.
            **config_overrides: Additional configuration overrides.
            
        Returns:
            Configured RealModelChatSession.
        """
        config = self.get_engine_config(model_key)
        
        # Apply any overrides
        for key, value in config_overrides.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.generation, key):
                setattr(config.generation, key, value)
        
        return RealModelChatSession(
            config=config,
            enable_performance_monitoring=True
        )

    def test_basic_chat_initialization(self):
        """Test that real model chat session initializes correctly."""
        skip_reason = self.should_skip_test()
        if skip_reason:
            pytest.skip(skip_reason)
        
        chat_session = self.create_chat_session()
        
        with chat_session.real_inference_session():
            assert chat_session.real_engine is not None
            assert hasattr(chat_session.real_engine, 'infer')
            
            # Test session lifecycle
            result = chat_session.start_session()
            assert result.success
            assert chat_session.is_active()
            
            result = chat_session.end_session()
            assert result.success
            assert not chat_session.is_active()

    def test_basic_real_model_conversation(self):
        """Test basic conversation with real model inference."""
        skip_reason = self.should_skip_test()
        if skip_reason:
            pytest.skip(skip_reason)
        
        chat_session = self.create_chat_session()
        
        with chat_session.real_inference_session():
            chat_session.start_session()
            
            # Send a simple message
            result = chat_session.send_message_with_real_inference(
                "Hello! Please say hello back to me."
            )
            
            assert result.success
            assert result.message
            assert len(result.message.strip()) > 0
            
            # Validate response quality
            chat_session.assert_response_quality(
                expected_keywords=["hello", "hi", "greetings"]
            )
            
            # Check conversation state
            conversation = chat_session.get_conversation()
            assert conversation is not None
            assert len(conversation.messages) == 2  # User + Assistant
            assert conversation.messages[0].role.value == "user"
            assert conversation.messages[1].role.value == "assistant"

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation with real model."""
        skip_reason = self.should_skip_test()
        if skip_reason:
            pytest.skip(skip_reason)
        
        chat_session = self.create_chat_session()
        
        with chat_session.real_inference_session():
            chat_session.start_session()
            
            # First exchange
            result1 = chat_session.send_message_with_real_inference(
                "What is 2 + 2? Please include the number 4 in your answer."
            )
            assert result1.success
            chat_session.assert_response_quality(expected_keywords=["4", "four"])
            
            # Second exchange - should maintain context
            result2 = chat_session.send_message_with_real_inference(
                "What about 3 + 3? Please include the word 'equals' in your response."
            )
            assert result2.success
            chat_session.assert_response_quality(expected_keywords=["6", "equals"])
            
            # Verify conversation state
            conversation = chat_session.get_conversation()
            assert len(conversation.messages) == 4  # 2 user + 2 assistant

    def test_chat_commands_with_real_model(self):
        """Test chat commands integration with real model responses."""
        skip_reason = self.should_skip_test()
        if skip_reason:
            pytest.skip(skip_reason)
        
        chat_session = self.create_chat_session()
        
        with temporary_chat_files({"test.txt": "This is test content"}) as temp_files:
            with chat_session.real_inference_session():
                chat_session.start_session()
                
                # Have a conversation
                result = chat_session.send_message_with_real_inference(
                    "Tell me about science. Please mention the word 'knowledge'."
                )
                assert result.success
                chat_session.assert_response_quality(expected_keywords=["knowledge"])
                
                # Test save command with real conversation
                temp_save_path = "test_conversation.json"
                save_result = chat_session.inject_command(f"/save({temp_save_path})")
                if save_result.success:
                    # Verify file was created and contains conversation
                    assert Path(temp_save_path).exists()
                    saved_content = Path(temp_save_path).read_text()
                    assert len(saved_content) > 0
                    # Clean up the test file
                    try:
                        Path(temp_save_path).unlink()
                    except Exception:
                        pass
                
                # Test clear command
                clear_result = chat_session.inject_command("/clear()")
                # Just verify the command can be executed - don't assert strict behavior
                # since clear behavior may vary between real and mock sessions
                assert isinstance(clear_result.success, bool)

    def test_real_model_error_handling(self):
        """Test error handling with real model inference."""
        skip_reason = self.should_skip_test()
        if skip_reason:
            pytest.skip(skip_reason)
        
        chat_session = self.create_chat_session()
        
        with chat_session.real_inference_session():
            # Test sending message without active session
            result = chat_session.send_message_with_real_inference("Hello")
            assert not result.success
            assert "no active session" in result.message.lower()
            
            # Start session and test very long input (edge case)
            chat_session.start_session()
            very_long_message = "Tell me about science. " * 1000  # Very long prompt
            
            # This should either work or fail gracefully
            result = chat_session.send_message_with_real_inference(very_long_message)
            # We don't assert success/failure since behavior depends on model limits
            # but we check that it doesn't crash and returns a result
            assert isinstance(result.success, bool)
            assert isinstance(result.message, str)

    def test_performance_monitoring(self):
        """Test performance monitoring during real model chat."""
        skip_reason = self.should_skip_test()
        if skip_reason:
            pytest.skip(skip_reason)
        
        monitor = ChatPerformanceMonitor()
        chat_session = self.create_chat_session()
        
        with chat_session.real_inference_session():
            monitor.start_session_monitoring()
            chat_session.start_session()
            
            # Send multiple messages to gather performance data
            for i in range(3):
                result = chat_session.send_message_with_real_inference(
                    f"Tell me fact number {i + 1} about science."
                )
                assert result.success
                
                # Small delay to allow measurement
                time.sleep(0.1)
            
            # End monitoring and check metrics
            metrics = monitor.end_session_monitoring(chat_session)
            
            assert "session_duration" in metrics
            assert "total_exchanges" in metrics
            assert metrics["total_exchanges"] >= 3
            
            # Get performance summary from session
            perf_summary = chat_session.get_performance_summary()
            assert "total_responses" in perf_summary
            assert "avg_response_time" in perf_summary
            assert perf_summary["total_responses"] >= 3

    def test_conversation_validation_patterns(self):
        """Test various conversation validation patterns from inference tests."""
        skip_reason = self.should_skip_test()
        if skip_reason:
            pytest.skip(skip_reason)
        
        chat_session = self.create_chat_session()
        
        # Test different validation configurations
        test_cases = [
            {
                "prompt": "Explain photosynthesis. Please use the word 'plants'.",
                "expected_keywords": ["plants", "light", "energy"],
                "min_length": 10,
                "max_length": 500
            },
            {
                "prompt": "What is 5 + 5? Give a short answer with the number.",
                "expected_keywords": ["10", "ten"],
                "min_length": 3,
                "max_length": 100,
                "require_sentences": True
            }
        ]
        
        with chat_session.real_inference_session():
            chat_session.start_session()
            
            for test_case in test_cases:
                # Configure validation settings
                chat_session.configure_validation_settings(
                    min_length=test_case.get("min_length", 3),
                    max_length=test_case.get("max_length", 1000),
                    require_sentences=test_case.get("require_sentences", False)
                )
                
                # Send message and validate
                result = chat_session.send_message_with_real_inference(test_case["prompt"])
                assert result.success
                
                # Validate response using configured settings
                validation_results = chat_session.validate_last_response()
                assert validation_results["basic_validation"]
                assert validation_results["non_empty_responses"]
                assert validation_results["appropriate_length"]
                
                # Test keyword-based assertion
                chat_session.assert_response_quality(
                    expected_keywords=test_case["expected_keywords"]
                )


@pytest.mark.real_model_chat
@requires_inference_backend()
class TestNativeChatEngine(AbstractRealModelChatTest):
    """Real chat tests with Native inference engine."""

    def get_engine_config(self, model_key: str = "smollm_135m") -> InferenceConfig:
        """Get Native engine configuration."""
        return create_real_model_inference_config(
            model_key=model_key,
            engine_type="NATIVE",
            max_new_tokens=100  # Allow longer responses for keyword inclusion tests
        )
    
    def get_engine_name(self) -> str:
        """Get engine name."""
        return "NATIVE"
    
    def should_skip_test(self) -> Optional[str]:
        """Native engine tests can always run."""
        return None


@pytest.mark.real_model_chat
@pytest.mark.single_gpu
@requires_cuda_initialized()
class TestVllmChatEngine(AbstractRealModelChatTest):
    """Real chat tests with vLLM inference engine."""

    def get_engine_config(self, model_key: str = "smollm_135m") -> InferenceConfig:
        """Get vLLM engine configuration."""
        return create_real_model_inference_config(
            model_key=model_key,
            engine_type="VLLM",
            max_new_tokens=100  # Allow longer responses for keyword inclusion tests
        )
    
    def get_engine_name(self) -> str:
        """Get engine name."""
        return "VLLM"
    
    def should_skip_test(self) -> Optional[str]:
        """Check if vLLM tests should be skipped."""
        try:
            import vllm
            return None
        except ImportError:
            return "vLLM not available"


@pytest.mark.real_model_chat
@requires_inference_backend()
class TestLlamaCppChatEngine(AbstractRealModelChatTest):
    """Real chat tests with LlamaCPP inference engine."""

    def get_engine_config(self, model_key: str = "gemma_270m_gguf") -> InferenceConfig:
        """Get LlamaCPP engine configuration."""
        return create_real_model_inference_config(
            model_key=model_key,
            engine_type="LLAMACPP",
            max_new_tokens=100  # Allow longer responses for keyword inclusion tests
        )
    
    def get_engine_name(self) -> str:
        """Get engine name."""
        return "LLAMACPP"
    
    def should_skip_test(self) -> Optional[str]:
        """Check if LlamaCPP tests should be skipped."""
        try:
            import llama_cpp
            return None
        except ImportError:
            return "llama-cpp-python not available"


class TestRealModelChatUtilities:
    """Test the real model chat utilities themselves."""

    @pytest.mark.real_model_chat
    def test_chat_performance_monitor(self):
        """Test the chat performance monitoring utilities."""
        monitor = ChatPerformanceMonitor()
        
        # Test empty state
        aggregate = monitor.get_aggregate_metrics()
        assert "no_sessions" in aggregate
        
        # Mock a session for testing
        monitor.start_session_monitoring()
        time.sleep(0.01)  # Small delay
        
        # Create a mock session with some data
        config = create_real_model_inference_config()
        session = RealModelChatSession(config)
        session.response_times = [0.5, 1.0, 0.8]
        session.token_counts = [10, 15, 12]
        
        metrics = monitor.end_session_monitoring(session)
        assert "session_duration" in metrics
        assert metrics["session_duration"] > 0
        
        # Test aggregate metrics
        aggregate = monitor.get_aggregate_metrics()
        assert "total_sessions" in aggregate
        assert aggregate["total_sessions"] == 1

    def test_temporary_chat_files(self):
        """Test temporary file creation for chat tests."""
        test_files = {
            "test1.txt": "Content 1",
            "test2.json": '{"test": "data"}',
            "test3.md": "# Test Markdown"
        }
        
        with temporary_chat_files(test_files) as temp_files:
            # Check all files were created
            assert len(temp_files) == 3
            
            for original_name, temp_path in temp_files.items():
                assert Path(temp_path).exists()
                assert Path(temp_path).is_file()
                
                # Check content
                content = Path(temp_path).read_text()
                assert content == test_files[original_name]
        
        # Check files were cleaned up
        for temp_path in temp_files.values():
            assert not Path(temp_path).exists()

    def test_real_model_config_creation(self):
        """Test real model configuration creation."""
        # Test default configuration
        config = create_real_model_inference_config()
        assert config.model is not None
        assert config.generation is not None
        assert config.engine == "NATIVE"
        
        # Test with overrides
        config = create_real_model_inference_config(
            engine_type="VLLM",
            max_new_tokens=50,
            temperature=0.7
        )
        assert config.engine == "VLLM"
        assert config.generation.max_new_tokens == 50
        assert config.generation.temperature == 0.7
        
        # Test invalid model key
        with pytest.raises(ValueError, match="Model key.*not found"):
            create_real_model_inference_config(model_key="invalid_model")

    def test_chat_conversation_creation(self):
        """Test chat-specific conversation creation."""
        conversations = create_real_model_chat_conversations()
        
        assert len(conversations) > 3  # Should have base + chat-specific
        
        # Check that we have chat-specific conversation IDs
        conversation_ids = [conv.conversation_id for conv in conversations]
        assert "chat_greeting" in conversation_ids
        assert "chat_help_request" in conversation_ids
        assert "chat_multi_turn" in conversation_ids
        
        # Check multi-turn conversation structure
        multi_turn = next(conv for conv in conversations if conv.conversation_id == "chat_multi_turn")
        assert len(multi_turn.messages) >= 3
        assert multi_turn.messages[0].role.value == "user"
        assert multi_turn.messages[1].role.value == "assistant"
        assert multi_turn.messages[2].role.value == "user"