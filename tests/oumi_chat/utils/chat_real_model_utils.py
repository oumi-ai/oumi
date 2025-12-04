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

"""Real model utilities for comprehensive chat testing."""

import resource
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi_chat.commands import CommandResult
from tests.integration.infer.test_inference_test_utils import (
    assert_performance_requirements,
    assert_response_properties,
    count_response_tokens,
    create_test_conversations,
    get_test_generation_params,
    get_test_models,
    validate_generation_output,
    validate_response_performance,
    validate_response_properties,
)
from tests.oumi_chat.utils.chat_test_utils import ChatTestSession


class RealModelChatSession(ChatTestSession):
    """Chat session that uses actual model inference instead of mocks.

    Inherits chat utilities but connects to real inference engines,
    applying the same validation patterns used in inference tests.
    """

    def __init__(
        self,
        config: InferenceConfig,
        mock_inputs: Optional[list[str]] = None,
        capture_output: bool = True,
        enable_performance_monitoring: bool = True,
    ):
        """Initialize real model chat session.

        Args:
            config: Inference configuration for real model.
            mock_inputs: Predefined inputs for testing.
            capture_output: Whether to capture output.
            enable_performance_monitoring: Track performance metrics.
        """
        super().__init__(config, mock_inputs, capture_output)

        # Replace mock engine with real inference engine
        self.real_engine: Optional[BaseInferenceEngine] = None
        self.enable_performance_monitoring = enable_performance_monitoring

        # Performance tracking
        self.response_times: list[float] = []
        self.token_counts: list[int] = []
        self.memory_usage: list[float] = []

        # Validation settings
        self.validation_settings = {
            "min_length": 3,
            "max_length": 1000,
            "require_sentences": False,
            "max_response_time": 60.0,
            "min_throughput": 0.5,  # tokens per second
        }

    def initialize_real_engine(self) -> bool:
        """Initialize the real inference engine.

        When CUDA is not available but MPS is, automatically fallback to LlamaCPP.

        Returns:
            True if engine initialized successfully, False otherwise.
        """
        try:
            # First attempt with original configuration
            from oumi.core.configs.inference_engine_type import InferenceEngineType

            engine_type = getattr(self.config, "engine", "NATIVE")
            if isinstance(engine_type, str):
                engine_type = InferenceEngineType(engine_type)

            self.real_engine = build_inference_engine(
                engine_type=engine_type,
                model_params=self.config.model,
                generation_params=self.config.generation,
            )
            # Update command context to use real engine
            if hasattr(self, "command_context") and self.command_context:
                self.command_context.inference_engine = self.real_engine
            return True
        except Exception as e:
            # Check if we can fallback to LlamaCPP on MPS
            import torch

            if (
                not torch.cuda.is_available()
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                try:
                    # Create LlamaCPP fallback config
                    fallback_config = self._create_llamacpp_fallback_config()
                    if fallback_config:
                        from oumi.core.configs.inference_engine_type import (
                            InferenceEngineType,
                        )

                        fallback_engine_type = getattr(
                            fallback_config, "engine", "LLAMACPP"
                        )
                        if isinstance(fallback_engine_type, str):
                            fallback_engine_type = InferenceEngineType(
                                fallback_engine_type
                            )

                        self.real_engine = build_inference_engine(
                            engine_type=fallback_engine_type,
                            model_params=fallback_config.model,
                            generation_params=fallback_config.generation,
                        )
                        # Update command context to use real engine
                        if hasattr(self, "command_context") and self.command_context:
                            self.command_context.inference_engine = self.real_engine
                        # Store that we're using fallback
                        self._using_fallback = True
                        return True
                except Exception as fallback_e:
                    self._initialization_error = (
                        f"Original: {str(e)}. Fallback: {str(fallback_e)}"
                    )
                    return False

            # Don't print errors in tests - they can clutter output
            # Store error for debugging if needed
            self._initialization_error = str(e)
            return False

    def _create_llamacpp_fallback_config(self) -> Optional[InferenceConfig]:
        """Create a LlamaCPP fallback configuration for MPS systems.

        Returns:
            LlamaCPP-compatible InferenceConfig or None if not possible.
        """
        try:
            from oumi.core.configs import InferenceConfig, ModelParams

            # Try to find a suitable GGUF model for testing
            # Use models that are available in configs/recipes
            test_models = [
                {
                    "model_name": "unsloth/gemma-3n-E4B-it-GGUF",
                    "tokenizer_name": "google/gemma-3n-E4B-it",
                    "filename": "gemma-3n-E4B-it-UD-Q5_K_XL.gguf",
                },
                {
                    "model_name": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                    "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
                    "filename": "Qwen3-4B-Instruct-2507-UD-Q5_K_XL.gguf",
                },
            ]

            for model_config in test_models:
                try:
                    # Create LlamaCPP configuration
                    model_params = ModelParams(
                        model_name=model_config["model_name"],
                        tokenizer_name=model_config["tokenizer_name"],
                        model_max_length=512,  # Keep small for testing
                        torch_dtype_str="float16",
                        trust_remote_code=True,
                        model_kwargs={"filename": model_config["filename"]},
                    )

                    # Use the same generation params as original config
                    fallback_config = InferenceConfig(
                        model=model_params,
                        generation=self.config.generation,
                        engine=InferenceEngineType.LLAMACPP,  # Force LlamaCPP engine
                    )

                    return fallback_config

                except Exception:
                    continue  # Try next model

            return None

        except ImportError:
            return None

    def _get_gguf_filename(self, model_name: str) -> str:
        """Get appropriate GGUF filename for a model.

        Args:
            model_name: HuggingFace model name.

        Returns:
            GGUF filename to use.
        """
        # Map models to their GGUF filenames
        filename_map = {
            "unsloth/SmolLM-135M-Instruct-GGUF": "SmolLM-135M-Instruct-Q4_K_M.gguf",
            "unsloth/Llama-3.2-1B-Instruct-GGUF": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "microsoft/Phi-3-mini-4k-instruct-gguf": "Phi-3-mini-4k-instruct-q4.gguf",
        }

        return filename_map.get(model_name, "model-Q4_K_M.gguf")

    def cleanup_real_engine(self):
        """Clean up the real inference engine resources."""
        if self.real_engine:
            try:
                # Attempt to cleanup if method exists
                cleanup_method = getattr(self.real_engine, "cleanup", None)
                if cleanup_method is not None:
                    cleanup_method()
                del self.real_engine
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self.real_engine = None

    @contextmanager
    def real_inference_session(self):
        """Context manager for real inference session with cleanup."""
        engine_initialized = self.initialize_real_engine()
        if not engine_initialized:
            # Import pytest here to avoid circular imports
            import pytest

            error_msg = getattr(self, "_initialization_error", "Unknown error")
            pytest.skip(f"Real inference engine not available: {error_msg}")

        try:
            yield self
        finally:
            self.cleanup_real_engine()

    def send_message_with_real_inference(self, message: str) -> CommandResult:
        """Send message and get real model response.

        Args:
            message: User message to send.

        Returns:
            Command result with real model response.
        """
        if not self.real_engine:
            return CommandResult(
                success=False, message="Real inference engine not initialized"
            )

        if not self._session_active:
            return CommandResult(
                success=False, message="No active session. Start a session first."
            )

        try:
            start_time = time.time()

            # Create conversation for inference
            user_message = Message(role=Role.USER, content=message)
            conversation = Conversation(
                conversation_id=f"real_inference_{len(self.conversation_history)}",
                messages=list(self._current_conversation.messages) + [user_message]
                if self._current_conversation
                else [user_message],
            )

            # Perform real inference
            conversations = self.real_engine.infer([conversation])
            elapsed_time = time.time() - start_time

            if conversations and conversations[0].messages:
                # Get the assistant's response
                assistant_messages = [
                    msg
                    for msg in conversations[0].messages
                    if msg.role == Role.ASSISTANT
                ]

                if assistant_messages:
                    assistant_response = assistant_messages[
                        -1
                    ].compute_flattened_text_content()

                    # Update conversation state
                    if not self._current_conversation:
                        self._current_conversation = Conversation(
                            conversation_id=f"real_session_{len(self.conversation_history)}",
                            messages=[],
                        )

                    self._current_conversation.messages.append(user_message)
                    self._current_conversation.messages.append(assistant_messages[-1])

                    # Sync conversation to command context so commands can access it
                    self._sync_to_command_context()

                    # Track performance if enabled
                    if self.enable_performance_monitoring:
                        token_count = count_response_tokens([conversations[0]])
                        self._track_performance(elapsed_time, token_count)

                    return CommandResult(success=True, message=assistant_response)

            return CommandResult(
                success=False, message="No response generated from model"
            )

        except Exception as e:
            return CommandResult(
                success=False, message=f"Real inference failed: {str(e)}"
            )

    def _track_performance(self, response_time: float, token_count: int):
        """Track performance metrics for analysis.

        Args:
            response_time: Time taken for inference.
            token_count: Number of tokens generated.
        """
        self.response_times.append(response_time)
        self.token_counts.append(token_count)

        # Track memory usage if available
        try:
            memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.memory_usage.append(memory_kb / 1024)  # Convert to MB
        except Exception:
            pass

    def validate_last_response(self) -> dict[str, bool]:
        """Validate the last model response using inference test patterns.

        Returns:
            Dictionary with validation results.
        """
        if not self._current_conversation or not self._current_conversation.messages:
            return {"valid_conversation": False}

        # Create conversation list for validation
        conversations = [self._current_conversation]

        # Basic generation validation
        basic_valid = validate_generation_output(conversations)

        # Property-based validation
        properties = validate_response_properties(
            conversations,
            min_length=self.validation_settings["min_length"],
            max_length=self.validation_settings["max_length"],
            require_complete_sentences=self.validation_settings["require_sentences"],
        )

        # Performance validation if we have metrics
        performance_results = {"completed_in_time": True, "adequate_throughput": True}
        if self.response_times and self.token_counts:
            performance_results = validate_response_performance(
                self.response_times[-1],
                self.token_counts[-1],
                max_time_seconds=self.validation_settings["max_response_time"],
                min_throughput=self.validation_settings["min_throughput"],
            )

        # Combine all validation results
        return {"basic_validation": basic_valid, **properties, **performance_results}

    def assert_response_quality(
        self,
        expected_keywords: Optional[list[str]] = None,
        forbidden_patterns: Optional[list[str]] = None,
    ):
        """Assert that the last response meets quality requirements.

        Args:
            expected_keywords: Keywords that should appear in response.
            forbidden_patterns: Patterns that should not appear.

        Raises:
            AssertionError: If validation fails.
        """
        if not self._current_conversation:
            raise AssertionError("No conversation to validate")

        conversations = [self._current_conversation]

        # Use inference test assertion patterns
        assert_response_properties(
            conversations,
            min_length=self.validation_settings["min_length"],
            max_length=self.validation_settings["max_length"],
            expected_keywords=expected_keywords,
            forbidden_patterns=forbidden_patterns,
            require_sentences=self.validation_settings["require_sentences"],
        )

        # Performance assertions if we have data
        if self.response_times and self.token_counts:
            assert_performance_requirements(
                self.response_times[-1],
                self.token_counts[-1],
                max_time_seconds=self.validation_settings["max_response_time"],
                min_throughput=self.validation_settings["min_throughput"],
            )

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of performance metrics.

        Returns:
            Dictionary with performance statistics.
        """
        if not self.response_times:
            return {"no_data": True}

        return {
            "total_responses": len(self.response_times),
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "max_response_time": max(self.response_times),
            "min_response_time": min(self.response_times),
            "total_tokens": sum(self.token_counts) if self.token_counts else 0,
            "avg_tokens_per_response": (
                sum(self.token_counts) / len(self.token_counts)
                if self.token_counts
                else 0
            ),
            "avg_throughput": (
                sum(self.token_counts) / sum(self.response_times)
                if self.token_counts and self.response_times
                else 0
            ),
            "peak_memory_mb": max(self.memory_usage) if self.memory_usage else 0,
        }

    def configure_validation_settings(self, **kwargs):
        """Update validation settings.

        Args:
            **kwargs: Validation settings to update.
        """
        self.validation_settings.update(kwargs)


def create_real_model_inference_config(
    model_key: str = "smollm_135m", engine_type: str = "NATIVE", **overrides
) -> InferenceConfig:
    """Create inference config for real model chat testing.

    Args:
        model_key: Key from test models configuration.
        engine_type: Inference engine type to use.
        **overrides: Additional config overrides.

    Returns:
        Inference configuration for real model testing.
    """
    test_models = get_test_models()
    if model_key not in test_models:
        raise ValueError(f"Model key '{model_key}' not found in test models")

    model_params = test_models[model_key]
    generation_params = get_test_generation_params()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(model_params, key):
            setattr(model_params, key, value)
        elif hasattr(generation_params, key):
            setattr(generation_params, key, value)

    return InferenceConfig(
        model=model_params,
        generation=generation_params,
        engine=InferenceEngineType(engine_type),
    )


def create_real_model_chat_conversations() -> list[Conversation]:
    """Create chat-specific test conversations for real model testing.

    Returns:
        List of conversations designed for chat testing.
    """
    # Start with inference test conversations
    base_conversations = create_test_conversations()

    # Add chat-specific conversation patterns
    chat_conversations = [
        Conversation(
            conversation_id="chat_greeting",
            messages=[
                Message(
                    role=Role.USER,
                    content=(
                        "Hello! Can you have a friendly conversation with me? "
                        "Please say hello back."
                    ),
                )
            ],
        ),
        Conversation(
            conversation_id="chat_help_request",
            messages=[
                Message(
                    role=Role.USER,
                    content=(
                        "I need help with a task. Can you assist me? "
                        "Please mention the word 'help' in your response."
                    ),
                )
            ],
        ),
        Conversation(
            conversation_id="chat_multi_turn",
            messages=[
                Message(role=Role.USER, content="What's the weather like?"),
                Message(
                    role=Role.ASSISTANT,
                    content=(
                        "I don't have access to current weather data, but I can "
                        "help you think about weather-related topics."
                    ),
                ),
                Message(
                    role=Role.USER,
                    content=(
                        "That's okay. Can you tell me about different types of "
                        "weather? Please mention 'rain' or 'sun'."
                    ),
                ),
            ],
        ),
    ]

    return base_conversations + chat_conversations


def create_fuzzing_conversation_prompts(count: int = 50) -> list[str]:
    """Create diverse prompts for chat fuzzing tests.

    Args:
        count: Number of prompts to generate.

    Returns:
        List of diverse prompts for stress testing.
    """
    base_prompts = [
        "Tell me about science",
        "Explain how computers work",
        "What is artificial intelligence?",
        "Help me understand mathematics",
        "Describe the ocean",
        "Tell me a short story",
        "What are the benefits of reading?",
        "How do plants grow?",
        "What is music theory?",
        "Explain the solar system",
        "Discuss climate change",
        "What is quantum physics?",
        "How does the internet work?",
        "What are black holes?",
        "Explain photosynthesis",
        "What is machine learning?",
        "Describe the human brain",
        "How do ecosystems work?",
        "What is renewable energy?",
        "Explain DNA and genetics",
    ]

    # Generate variations and combinations
    prompts = []
    for i in range(count):
        base = base_prompts[i % len(base_prompts)]

        if i % 5 == 0:
            # Add keyword instruction
            prompts.append(
                f"{base} Please include the word 'interesting' in your response."
            )
        elif i % 5 == 1:
            # Add length instruction
            prompts.append(f"{base} Give me a brief answer.")
        elif i % 5 == 2:
            # Add follow-up question
            prompts.append(f"{base} Also, why is this topic important?")
        elif i % 5 == 3:
            # Add context
            prompts.append(f"I'm curious about learning. {base}")
        else:
            # Original prompt with variation
            variations = [
                f"{base} Please explain in simple terms.",
                f"Can you help me understand {base.lower()}?",
                f"I need to learn about {base.lower()}.",
                base,  # Keep some original
            ]
            prompts.append(variations[i % len(variations)])

    return prompts


@contextmanager
def temporary_chat_files(file_contents: dict[str, str]):
    """Create temporary files for chat testing with cleanup.

    Args:
        file_contents: Mapping of filename to content.

    Yields:
        Dictionary mapping filenames to temporary file paths.
    """
    temp_files = {}
    temp_paths = []

    try:
        for filename, content in file_contents.items():
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=Path(filename).suffix, prefix="chat_test_"
            )
            with open(temp_fd, "w", encoding="utf-8") as f:
                f.write(content)

            temp_files[filename] = temp_path
            temp_paths.append(temp_path)

        yield temp_files

    finally:
        # Clean up temporary files
        for temp_path in temp_paths:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors


class ChatPerformanceMonitor:
    """Monitor performance during chat sessions."""

    def __init__(self):
        self.session_metrics: list[dict[str, Any]] = []
        self.current_session_start: Optional[float] = None

    def start_session_monitoring(self):
        """Start monitoring a chat session."""
        self.current_session_start = time.time()

    def end_session_monitoring(self, session: RealModelChatSession) -> dict[str, Any]:
        """End session monitoring and record metrics.

        Args:
            session: The chat session to analyze.

        Returns:
            Session performance metrics.
        """
        if self.current_session_start is None:
            return {}

        session_duration = time.time() - self.current_session_start
        performance_summary = session.get_performance_summary()

        # Count exchanges from both conversation history and current active conversation
        total_exchanges = len(session.conversation_history)
        if hasattr(session, "_current_conversation") and session._current_conversation:
            # Count user/assistant message pairs in the current conversation
            messages = session._current_conversation.messages
            user_messages = sum(1 for msg in messages if msg.role.value == "user")
            total_exchanges += user_messages

        metrics = {
            "session_duration": session_duration,
            "total_exchanges": total_exchanges,
            **performance_summary,
        }

        self.session_metrics.append(metrics)
        self.current_session_start = None

        return metrics

    def get_aggregate_metrics(self) -> dict[str, Any]:
        """Get aggregate performance metrics across all sessions.

        Returns:
            Aggregate performance statistics.
        """
        if not self.session_metrics:
            return {"no_sessions": True}

        return {
            "total_sessions": len(self.session_metrics),
            "avg_session_duration": sum(
                m.get("session_duration", 0) for m in self.session_metrics
            )
            / len(self.session_metrics),
            "total_exchanges": sum(
                m.get("total_exchanges", 0) for m in self.session_metrics
            ),
            "avg_response_time": sum(
                m.get("avg_response_time", 0) for m in self.session_metrics
            )
            / len(self.session_metrics),
            "peak_memory_across_sessions": max(
                m.get("peak_memory_mb", 0) for m in self.session_metrics
            ),
        }
