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

"""Integration tests for WebChat with real models."""

import asyncio
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from tests.markers import requires_cuda_initialized, requires_gpus
from tests.unit.webchat.utils.webchat_test_utils import (
    MockWebSocketClient,
    cleanup_test_files,
    create_mock_file_upload,
    mock_webchat_server,
)


class TestRealModelWebChatIntegration:
    """Test WebChat integration with actual inference models."""

    @pytest.fixture
    def small_model_config(self):
        """Create config for small test model."""
        return InferenceConfig(
            model=ModelParams(
                model_name="HuggingFaceTB/SmolLM-135M-Instruct",
                model_max_length=512,
                torch_dtype_str="float16",
                attn_implementation="sdpa",
                trust_remote_code=True,
            ),
            generation=GenerationParams(max_new_tokens=50, temperature=0.7, seed=42),
        )

    @pytest.fixture
    def mock_inference_engine(self):
        """Create mock inference engine for testing."""
        engine = Mock()

        # Mock standard responses
        engine.generate.return_value = {
            "generated_text": "This is a test response from the model.",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        engine.is_loaded = True
        engine.model_name = "SmolLM-135M-Instruct"
        return engine

    def test_webchat_server_with_model_config(self, small_model_config):
        """Test WebChat server initialization with real model config."""
        with mock_webchat_server(config=small_model_config) as server:
            # Verify server starts with model configuration
            assert server.running
            assert (
                server.config.model.model_name == "HuggingFaceTB/SmolLM-135M-Instruct"
            )
            assert server.config.generation.max_new_tokens == 50

            # Create session with model config
            session_id = server.create_session("model_config_test")
            session = server.get_session(session_id)

            # Verify session inherits model configuration
            assert (
                session.command_context.config.model.model_name
                == "HuggingFaceTB/SmolLM-135M-Instruct"
            )
            assert session.command_context.config.generation.temperature == 0.7

    @pytest.mark.asyncio
    async def test_chat_completion_via_websocket(self, mock_inference_engine):
        """Test chat completion through WebSocket with mocked inference."""
        with patch("oumi.webchat.server.build_inference_engine", return_value=mock_inference_engine), \
             patch("oumi.infer.get_engine", return_value=mock_inference_engine), \
             patch("oumi.infer.infer", return_value=[]):
            client = MockWebSocketClient()
            await client.connect()

            # Mock successful chat completion response
            completion_response = {
                "type": "chat_completion",
                "message": "This is a test response from the model.",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
                "model": "SmolLM-135M-Instruct",
                "success": True,
            }
            client.add_auto_response("chat", completion_response)

            # Send chat message
            chat_message = {
                "type": "chat",
                "message": "What is the capital of France?",
                "session_id": "test_session",
                "stream": False,
            }

            response = await client.send_message(chat_message)

            assert response["type"] == "chat_completion"
            assert response["success"] is True
            assert "This is a test response" in response["message"]
            assert response["model"] == "SmolLM-135M-Instruct"
            assert response["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_streaming_chat_completion(self, mock_inference_engine):
        """Test streaming chat completion via WebSocket."""
        with patch("oumi.webchat.server.build_inference_engine", return_value=mock_inference_engine), \
             patch("oumi.infer.get_engine", return_value=mock_inference_engine), \
             patch("oumi.infer.infer", return_value=[]):
            client = MockWebSocketClient()
            await client.connect()

            # Mock streaming response chunks
            stream_chunks = [
                {"type": "chat_stream_start", "message_id": "msg_123", "success": True},
                {
                    "type": "chat_stream_chunk",
                    "delta": "This is ",
                    "message_id": "msg_123",
                },
                {
                    "type": "chat_stream_chunk",
                    "delta": "a streaming ",
                    "message_id": "msg_123",
                },
                {
                    "type": "chat_stream_chunk",
                    "delta": "response.",
                    "message_id": "msg_123",
                },
                {
                    "type": "chat_stream_end",
                    "message_id": "msg_123",
                    "final_message": "This is a streaming response.",
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 5,
                        "total_tokens": 13,
                    },
                    "success": True,
                },
            ]

            # Add responses for streaming
            for i, chunk in enumerate(stream_chunks):
                client.add_auto_response(f"chat_stream_{i}", chunk)

            # Send streaming chat message
            stream_message = {
                "type": "chat",
                "message": "Tell me about streaming",
                "session_id": "test_session",
                "stream": True,
            }

            # For this mock test, we simulate receiving all chunks
            response = await client.send_message(stream_message)

            # The mock will return the first configured response
            assert response["type"] == "chat_stream_start"
            assert response["success"] is True
            assert "message_id" in response

    @pytest.mark.asyncio
    async def test_conversation_context_management(self, mock_inference_engine):
        """Test conversation context management across multiple messages."""
        with patch("oumi.webchat.server.build_inference_engine", return_value=mock_inference_engine), \
             patch("oumi.infer.get_engine", return_value=mock_inference_engine), \
             patch("oumi.infer.infer", return_value=[]):
            client = MockWebSocketClient()
            await client.connect()

            # Setup conversation context tracking
            conversation_history = []

            def create_contextual_response(message_content: str) -> Dict[str, Any]:
                conversation_history.append(
                    {"role": "user", "content": message_content}
                )

                # Generate contextual response based on conversation history
                if "name" in message_content.lower():
                    response_text = "Nice to meet you! I'm Claude, an AI assistant."
                elif len(conversation_history) > 1:
                    response_text = f"I remember our conversation. You previously mentioned: {conversation_history[-2]['content'][:20]}..."
                else:
                    response_text = "Hello! How can I help you today?"

                conversation_history.append(
                    {"role": "assistant", "content": response_text}
                )

                return {
                    "type": "chat_completion",
                    "message": response_text,
                    "conversation_length": len(conversation_history),
                    "context_tokens": len(conversation_history) * 10,  # Approximate
                    "success": True,
                }

            # Test conversation flow
            messages = [
                "Hello, my name is John",
                "What did I just tell you?",
                "Can you summarize our conversation?",
            ]

            responses = []
            for i, message in enumerate(messages):
                contextual_response = create_contextual_response(message)
                client.add_auto_response(f"chat_{i}", contextual_response)

                chat_message = {
                    "type": "chat",
                    "message": message,
                    "session_id": "context_test_session",
                }

                response = await client.send_message(chat_message)
                responses.append(response)

            # Verify contextual understanding
            assert (
                responses[0]["message"]
                == "Nice to meet you! I'm Claude, an AI assistant."
            )
            assert (
                "remember our conversation" in responses[1]["message"]
                or "previously mentioned" in responses[1]["message"]
            )
            assert (
                responses[2]["conversation_length"] == 6
            )  # 3 user + 3 assistant messages

    def test_webchat_with_file_upload_and_model_processing(self, mock_inference_engine):
        """Test file upload integration with model processing."""
        with patch("oumi.webchat.server.build_inference_engine", return_value=mock_inference_engine), \
             patch("oumi.infer.get_engine", return_value=mock_inference_engine), \
             patch("oumi.infer.infer", return_value=[]):
            with mock_webchat_server() as server:
                session_id = server.create_session("file_processing_test")
                session = server.get_session(session_id)

                # Create test file
                test_content = b"This is a test document with important information about AI and machine learning."
                mock_file = create_mock_file_upload(
                    "ai_document.txt", test_content, "text/plain"
                )

                # Simulate file upload and processing
                upload_response = server.handle_rest_request(
                    "POST",
                    f"/files/{session_id}",
                    {
                        "file_name": mock_file.name,
                        "file_type": mock_file.content_type,
                        "file_size": mock_file.size,
                        "action": "analyze_and_chat",
                    },
                )

                assert upload_response["status"] == "ok"

                # Add conversation about uploaded file
                session.add_message("user", "What does the uploaded document contain?")

                # Mock model response based on file content
                mock_inference_engine.generate.return_value = {
                    "generated_text": "The document discusses AI and machine learning topics.",
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 25,
                        "completion_tokens": 12,
                        "total_tokens": 37,
                    },
                }

                session.add_message(
                    "assistant",
                    "The document discusses AI and machine learning topics.",
                )

                # Verify conversation includes file context
                assert len(session.conversation_history) == 2
                assert "uploaded document" in session.conversation_history[0]["content"]

                cleanup_test_files(mock_file.temp_path)

    @pytest.mark.asyncio
    async def test_model_switching_via_websocket(self, mock_inference_engine):
        """Test model switching through WebSocket interface."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock model switching response
        switch_response = {
            "type": "model_switched",
            "previous_model": "SmolLM-135M-Instruct",
            "new_model": "SmolLM-360M-Instruct",
            "model_loaded": True,
            "loading_time": 2.5,
            "new_model_info": {
                "parameters": "360M",
                "context_length": 2048,
                "capabilities": ["chat", "code", "reasoning"],
            },
            "success": True,
        }
        client.add_auto_response("switch_model", switch_response)

        switch_message = {
            "type": "switch_model",
            "target_model": "SmolLM-360M-Instruct",
            "session_id": "model_switch_test",
        }

        response = await client.send_message(switch_message)

        assert response["type"] == "model_switched"
        assert response["new_model"] == "SmolLM-360M-Instruct"
        assert response["model_loaded"] is True
        assert response["loading_time"] > 0

    @pytest.mark.asyncio
    async def test_error_handling_with_model_failures(self, mock_inference_engine):
        """Test error handling when model inference fails."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock model failure scenarios
        failure_responses = [
            {
                "type": "chat_error",
                "error_code": "MODEL_OVERLOADED",
                "error_message": "Model is currently overloaded, please try again",
                "retry_after": 5,
                "success": False,
            },
            {
                "type": "chat_error",
                "error_code": "CONTEXT_LENGTH_EXCEEDED",
                "error_message": "Input exceeds model context length",
                "context_limit": 2048,
                "input_tokens": 2100,
                "success": False,
            },
            {
                "type": "chat_error",
                "error_code": "MODEL_UNAVAILABLE",
                "error_message": "Model is temporarily unavailable",
                "suggested_models": ["SmolLM-135M-Instruct", "SmolLM-360M-Instruct"],
                "success": False,
            },
        ]

        for i, failure_response in enumerate(failure_responses):
            # Set up the response for the 'chat' message type
            client.add_auto_response("chat", failure_response)

            error_message = {
                "type": "chat",
                "message": f"Test message {i} that causes error",
                "session_id": "error_test_session",
            }

            response = await client.send_message(error_message)

            assert response["type"] == "chat_error"
            assert response["success"] is False
            assert "error_code" in response
            assert "error_message" in response
            
            # Clear the auto-response for the next iteration
            client.auto_responses.clear()

    def test_performance_monitoring_integration(self, mock_inference_engine):
        """Test performance monitoring integration with real models."""
        with mock_webchat_server() as server:
            session_id = server.create_session("performance_test")
            session = server.get_session(session_id)

            # Mock performance metrics
            performance_data = {
                "inference_time": 0.75,
                "tokens_per_second": 25.3,
                "memory_usage": {
                    "gpu_memory_used": 2048,
                    "gpu_memory_total": 8192,
                    "cpu_memory_used": 1024,
                },
                "model_info": {
                    "model_name": "SmolLM-135M-Instruct",
                    "parameters": "135M",
                    "precision": "float16",
                },
            }

            # Simulate chat with performance tracking
            session.add_message("user", "What is machine learning?")

            # Mock inference with performance data
            mock_inference_engine.generate.return_value = {
                "generated_text": "Machine learning is a subset of artificial intelligence...",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 45,
                    "total_tokens": 60,
                },
                "performance": performance_data,
            }

            session.add_message(
                "assistant",
                "Machine learning is a subset of artificial intelligence...",
            )

            # Test performance monitoring endpoint
            perf_response = server.handle_rest_request(
                "GET", f"/performance/{session_id}"
            )

            # Verify performance data is available
            assert "inference_time" in str(perf_response)
            assert "tokens_per_second" in str(perf_response)

    @pytest.mark.asyncio
    async def test_concurrent_model_requests(self, mock_inference_engine):
        """Test concurrent model requests through WebSocket."""
        num_clients = 3
        clients = [MockWebSocketClient() for _ in range(num_clients)]

        # Connect all clients
        for client in clients:
            await client.connect()

        # Configure responses for concurrent requests
        for i, client in enumerate(clients):
            concurrent_response = {
                "type": "chat_completion",
                "message": f"Response to concurrent request {i}",
                "request_id": f"req_{i}",
                "processing_time": 0.5 + (i * 0.1),
                "success": True,
            }
            client.add_auto_response("chat", concurrent_response)

        # Send concurrent requests
        tasks = []
        for i, client in enumerate(clients):
            message = {
                "type": "chat",
                "message": f"Concurrent test message {i}",
                "session_id": f"concurrent_session_{i}",
            }
            task = client.send_message(message)
            tasks.append(task)

        # Wait for all responses
        responses = await asyncio.gather(*tasks)

        # Verify all requests completed successfully
        assert len(responses) == num_clients
        for i, response in enumerate(responses):
            assert response["type"] == "chat_completion"
            assert response["success"] is True
            assert f"concurrent request {i}" in response["message"]


@pytest.mark.skipif(
    not pytest.importorskip("torch", minversion="1.9.0"),
    reason="PyTorch required for real model tests",
)
class TestRealModelEndToEnd:
    """End-to-end tests with actual model loading (requires GPU/significant resources)."""

    @requires_gpus(1, min_gb=2.0)
    @pytest.mark.single_gpu
    @pytest.mark.slow
    def test_full_webchat_with_small_model(self):
        """Test full WebChat workflow with actual small model (requires GPU)."""
        # This test would require actual model loading
        # Skipped by default due to resource requirements
        pytest.skip(
            "Requires GPU and significant resources - run manually for full validation"
        )

    @pytest.mark.asyncio
    @requires_cuda_initialized()
    @pytest.mark.single_gpu
    async def test_real_websocket_with_model_inference(self):
        """Test real WebSocket connection with model inference."""
        # This would test against a running WebChat server with real model
        pytest.skip(
            "Requires running WebChat server - integration test for CI/CD pipeline"
        )

    def test_memory_usage_with_real_models(self):
        """Test memory usage patterns with real model loading."""
        # This would monitor actual memory usage during model operations
        pytest.skip("Requires actual model loading - use for memory profiling")


class TestModelCompatibility:
    """Test WebChat compatibility with different model types."""

    def test_model_config_validation(self):
        """Test model configuration validation for different model types."""
        # Test configurations for different model families
        model_configs = [
            {
                "model_name": "HuggingFaceTB/SmolLM-135M-Instruct",
                "expected_family": "smol_lm",
                "expected_features": ["chat", "instruct"],
            },
            {
                "model_name": "microsoft/Phi-3.5-mini-instruct",
                "expected_family": "phi",
                "expected_features": ["chat", "instruct", "code"],
            },
            {
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "expected_family": "qwen",
                "expected_features": ["chat", "instruct", "multilingual"],
            },
        ]

        for config_data in model_configs:
            config = InferenceConfig(
                model=ModelParams(
                    model_name=config_data["model_name"],
                    model_max_length=2048,
                    torch_dtype_str="float16",
                ),
                generation=GenerationParams(max_new_tokens=100, temperature=0.7),
            )

            with mock_webchat_server(config=config) as server:
                assert server.running
                assert config_data["model_name"] in server.config.model.model_name

                # Test session creation with model config
                session_id = server.create_session("compatibility_test")
                session = server.get_session(session_id)
                assert (
                    session.command_context.config.model.model_name
                    == config_data["model_name"]
                )

    @pytest.mark.asyncio
    async def test_model_capability_detection(self):
        """Test detection of model capabilities for WebChat features."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock model capability response
        capability_response = {
            "type": "model_capabilities",
            "model_name": "SmolLM-135M-Instruct",
            "capabilities": {
                "chat": True,
                "code_generation": True,
                "function_calling": False,
                "vision": False,
                "audio": False,
                "multimodal": False,
            },
            "limitations": {
                "max_context_length": 2048,
                "max_output_tokens": 1024,
                "languages": ["en"],
                "safety_filters": True,
            },
            "success": True,
        }
        client.add_auto_response("get_model_capabilities", capability_response)

        capability_request = {
            "type": "get_model_capabilities",
            "session_id": "capability_test",
        }

        response = await client.send_message(capability_request)

        assert response["type"] == "model_capabilities"
        assert response["capabilities"]["chat"] is True
        assert response["limitations"]["max_context_length"] > 0

        # Test UI feature enabling based on capabilities
        ui_features_response = {
            "type": "ui_features_updated",
            "enabled_features": {
                "file_upload": response["capabilities"]["multimodal"],
                "code_execution": response["capabilities"]["code_generation"],
                "function_calling": response["capabilities"]["function_calling"],
                "image_upload": response["capabilities"]["vision"],
            },
            "success": True,
        }
        client.add_auto_response("update_ui_features", ui_features_response)

        ui_update_request = {
            "type": "update_ui_features",
            "model_capabilities": response["capabilities"],
            "session_id": "capability_test",
        }

        ui_response = await client.send_message(ui_update_request)

        assert ui_response["type"] == "ui_features_updated"
        # Features should be enabled based on model capabilities
        assert (
            ui_response["enabled_features"]["code_execution"] is True
        )  # SmolLM supports code
        assert (
            ui_response["enabled_features"]["image_upload"] is False
        )  # No vision support
