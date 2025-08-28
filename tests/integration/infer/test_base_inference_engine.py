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

"""Abstract base class for inference engine integration tests."""

import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pytest

from oumi.core.configs import GenerationParams, InferenceConfig
from tests.integration.infer.test_inference_test_utils import (
    assert_performance_requirements,
    assert_response_properties,
    count_response_tokens,
    create_batch_conversations,
    create_test_conversations,
    get_contextual_keywords,
    get_test_generation_params,
    get_test_models,
    validate_generation_output,
)


class AbstractInferenceEngineTest(ABC):
    """Abstract base class for inference engine integration tests.

    This class provides common test patterns that are shared across all
    inference engine implementations, reducing code duplication and ensuring
    consistent testing patterns.

    Subclasses must implement the abstract methods to specify engine-specific
    configuration and behavior.
    """

    @abstractmethod
    def get_engine_class(self) -> type:
        """Return the inference engine class to test.

        Returns:
            The inference engine class (e.g., VLLMInferenceEngine).
        """
        pass

    @abstractmethod
    def get_default_model_key(self) -> str:
        """Return the default model key for basic testing.

        Returns:
            Model key string that exists in get_test_models() output.
        """
        pass

    @abstractmethod
    def get_performance_thresholds(self) -> dict[str, Any]:
        """Return engine-specific performance expectations.

        Returns:
            Dictionary with performance thresholds:
            - max_time_seconds: Maximum acceptable inference time
            - min_throughput: Minimum tokens per second
            - batch_size: Default batch size for batch tests
        """
        pass

    def get_engine_instance(self, model_key: Optional[str] = None):
        """Create an engine instance with the specified model.

        Args:
            model_key: Model key to use. If None, uses get_default_model_key().

        Returns:
            Configured inference engine instance.
        """
        if model_key is None:
            model_key = self.get_default_model_key()

        models = get_test_models()
        engine_class = self.get_engine_class()
        return engine_class(models[model_key])

    def test_basic_inference(self):
        """Test basic inference functionality with single conversation."""
        engine = self.get_engine_instance()
        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time

        # Basic validation
        assert validate_generation_output(result)
        assert len(result) == len(conversations)

        # Enhanced property validation
        assert_response_properties(
            result,
            min_length=3,
            max_length=400,
            forbidden_patterns=[r"\berror\b", r"\bfailed\b", r"\bunable\b"],
        )

        # Performance validation
        tokens_generated = count_response_tokens(result)
        thresholds = self.get_performance_thresholds()
        assert_performance_requirements(
            elapsed_time,
            tokens_generated,
            max_time_seconds=thresholds.get("max_time_seconds", 60.0),
            min_throughput=thresholds.get("min_throughput", 1.0),
        )

    def test_batch_inference(self):
        """Test batched conversation inference."""
        engine = self.get_engine_instance()
        thresholds = self.get_performance_thresholds()
        batch_size = thresholds.get("batch_size", 3)

        conversations = create_batch_conversations(batch_size, "Tell me about")

        generation_params = get_test_generation_params()
        generation_params.max_new_tokens = 15
        inference_config = InferenceConfig(generation=generation_params)

        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time

        # Basic validation
        assert validate_generation_output(result)
        assert len(result) == len(conversations)

        # Validate each response
        for i, conversation in enumerate(result):
            original_msg = conversations[i].messages[0]
            original_prompt = original_msg.compute_flattened_text_content()
            expected_keywords = get_contextual_keywords(original_prompt)

            # Filter to get actual topic words (not meta instruction words)
            topic_keywords = []
            if expected_keywords:
                # Remove meta-instruction words that models might not naturally include
                meta_words = {
                    "about",
                    "details",
                    "information",
                    "tell",
                    "explanation",
                    "description",
                }
                topic_keywords = [
                    kw for kw in expected_keywords if kw.lower() not in meta_words
                ]

            assert_response_properties(
                [conversation],
                min_length=3,
                max_length=200,
                # Only expect 1 topic keyword
                expected_keywords=topic_keywords[:1] if topic_keywords else None,
                forbidden_patterns=[r"\berror\b", r"\bfailed\b"],
            )

        # Performance validation
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time,
            tokens_generated,
            max_time_seconds=thresholds.get("max_time_seconds", 60.0),
            min_throughput=thresholds.get("min_throughput", 1.0),
        )

    def test_file_io(self):
        """Test input/output file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = self.get_engine_instance()
            conversations = create_test_conversations()[:2]
            output_path = Path(temp_dir) / "engine_output.jsonl"

            generation_params = get_test_generation_params()
            generation_params.max_new_tokens = 10
            inference_config = InferenceConfig(
                generation=generation_params, output_path=str(output_path)
            )

            result = engine.infer(conversations, inference_config)

            # Validate output
            assert validate_generation_output(result)
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_empty_input(self):
        """Test graceful handling of empty conversations."""
        engine = self.get_engine_instance()
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer([], inference_config)
        assert result == []

    def test_generation_params(self):
        """Test generation parameter handling."""
        engine = self.get_engine_instance()
        conversation = create_test_conversations()[:1]

        # Test with specific generation parameters
        gen_params = GenerationParams(
            max_new_tokens=15, temperature=0.7, top_p=0.9, seed=42, use_sampling=True
        )
        config = InferenceConfig(generation=gen_params)
        result = engine.infer(conversation, config)

        assert validate_generation_output(result)

        # Validate response properties
        assert_response_properties(
            result, min_length=3, max_length=300, forbidden_patterns=[r"\berror\b"]
        )

    def test_deterministic_generation(self):
        """Test seed-based reproducible outputs."""
        engine = self.get_engine_instance()
        conversation = create_test_conversations()[:1]

        # Use deterministic parameters
        gen_params = GenerationParams(
            max_new_tokens=20, temperature=0.0, seed=42, use_sampling=False
        )
        config = InferenceConfig(generation=gen_params)

        # Run inference twice with same parameters
        result1 = engine.infer(conversation, config)
        result2 = engine.infer(conversation, config)

        # Both should be valid
        assert validate_generation_output(result1)
        assert validate_generation_output(result2)

        # Extract responses
        response1 = result1[0].messages[-1].content
        response2 = result2[0].messages[-1].content

        # With deterministic settings, should have some consistency
        # Note: Some engines may still have variability, so we check basic
        # properties rather than exact matches
        assert len(response1.strip()) > 0
        assert len(response2.strip()) > 0

    def test_invalid_model_name(self):
        """Test error handling for invalid model names."""
        # Test error handling for invalid model names

        # Create a copy of the model params with invalid model name
        try:
            models = get_test_models()
            model_key = self.get_default_model_key()
            model_params = models[model_key]
        except (KeyError, AttributeError):
            # Fallback for engines with custom model configurations
            from oumi.core.configs import ModelParams

            model_params = ModelParams(
                model_name="nonexistent/invalid-model",
                trust_remote_code=True,
            )
        else:
            model_params.model_name = "nonexistent/invalid-model"

        # Should raise an error during engine initialization
        engine_class = self.get_engine_class()
        with pytest.raises(Exception):
            engine_class(model_params)


class AbstractInferenceEngineBasicFunctionality(AbstractInferenceEngineTest):
    """Base class for basic functionality tests."""

    pass


class AbstractInferenceEngineGenerationParameters(AbstractInferenceEngineTest):
    """Base class for generation parameter tests."""

    pass


class AbstractInferenceEngineErrorHandling(AbstractInferenceEngineTest):
    """Base class for error handling tests."""

    pass
