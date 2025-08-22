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

"""Live integration tests for VLLMInferenceEngine with real models."""

import time
from typing import Any

import pytest

from oumi.core.configs import GenerationParams, InferenceConfig
from oumi.inference.vllm_inference_engine import VLLMInferenceEngine
from tests.integration.infer.inference_test_utils import (
    assert_performance_requirements,
    assert_response_properties,
    assert_response_relevance,
    count_response_tokens,
    create_batch_conversations,
    create_test_conversations,
    get_test_generation_params,
    get_test_models,
    validate_generation_output,
)
from tests.integration.infer.test_base_inference_engine import (
    AbstractInferenceEngineBasicFunctionality,
    AbstractInferenceEngineErrorHandling,
    AbstractInferenceEngineGenerationParameters,
)
from tests.markers import requires_cuda_initialized, requires_gpus

# Skip all tests if vLLM is not available
try:
    import vllm  # noqa: F401

    vllm_available = True
except ImportError:
    vllm_available = False

pytestmark = [
    pytest.mark.skipif(not vllm_available, reason="vLLM not available"),
    pytest.mark.requires_vllm,
]


class TestVLLMBasicFunctionality(AbstractInferenceEngineBasicFunctionality):
    """Test core VLLM inference functionality using abstract base class.

    This class inherits 7 comprehensive test methods:
    - test_basic_inference() - Single conversation inference
    - test_batch_inference() - Multiple conversation batch processing
    - test_file_io() - Input/output file handling
    - test_empty_input() - Edge case: empty conversation list
    - test_generation_params() - Parameter validation and handling
    - test_deterministic_generation() - Seed-based reproducibility testing
    - test_invalid_model_name() - Error handling for invalid models
    """

    def get_engine_class(self) -> type:
        """Return the VLLM inference engine class."""
        return VLLMInferenceEngine

    def get_default_model_key(self) -> str:
        """Return the default model key for VLLM testing."""
        return "gemma_270m"

    def get_performance_thresholds(self) -> dict[str, Any]:
        """Return VLLM-specific performance expectations."""
        return {
            "max_time_seconds": 60.0,
            "min_throughput": 10.0,  # VLLM should achieve higher throughput
            "batch_size": 8,  # VLLM handles larger batches well
        }

    # Additional VLLM-specific tests can be added here
    @pytest.mark.memory_intensive  # Requires >6GB RAM
    def test_vllm_specific_gemma_270m(self):
        """Test VLLM-specific features with Gemma-3-270m model."""
        # This is an example of an additional VLLM-specific test
        # The basic functionality is already covered by inherited methods
        import torch

        _use_gpu = torch.cuda.is_available()

        models = get_test_models()
        engine = VLLMInferenceEngine(models["gemma_270m"])

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)

        # Basic validation (detailed validation is in inherited methods)
        assert validate_generation_output(result)

        # VLLM-specific validation could go here
        assert_response_properties(
            result,
            min_length=5,
            max_length=500,
            expected_keywords=["Hello"],
            forbidden_patterns=[r"\berror\b", r"\bfailed\b", r"\bunable\b"],
        )


class TestVLLMGenerationParameters(AbstractInferenceEngineGenerationParameters):
    """Test VLLM generation parameter handling using abstract base class.

    This class inherits comprehensive parameter testing from the base class
    and adds VLLM-specific parameter variations.
    """

    def get_engine_class(self) -> type:
        """Return the VLLM inference engine class."""
        return VLLMInferenceEngine

    def get_default_model_key(self) -> str:
        """Return the default model key for VLLM testing."""
        return "smollm_135m"

    def get_performance_thresholds(self) -> dict[str, Any]:
        """Return VLLM-specific performance expectations."""
        return {
            "max_time_seconds": 30.0,
            "min_throughput": 5.0,
            "batch_size": 4,
        }

    # VLLM-specific parameter tests
    def test_vllm_temperature_variation(self):
        """Test VLLM-specific temperature parameter effects."""
        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])
        conversation = create_test_conversations()[0:1]

        # Test with temperature=0.0 (deterministic)
        gen_params_deterministic = GenerationParams(
            max_new_tokens=20, temperature=0.0, seed=42, use_sampling=False
        )
        config_det = InferenceConfig(generation=gen_params_deterministic)
        result_det = engine.infer(conversation, config_det)

        # Test with higher temperature (more random)
        gen_params_random = GenerationParams(
            max_new_tokens=20, temperature=0.8, seed=42, use_sampling=True
        )
        config_random = InferenceConfig(generation=gen_params_random)
        result_random = engine.infer(conversation, config_random)

        # Both should generate valid responses
        assert validate_generation_output(result_det)
        assert validate_generation_output(result_random)

        # Enhanced validation for temperature variations
        assert_response_properties(
            result_det,
            min_length=5,
            max_length=400,
            forbidden_patterns=[r"\berror\b", r"\bfailed\b"],
            require_sentences=True,
        )

        assert_response_properties(
            result_random,
            min_length=5,
            max_length=400,
            forbidden_patterns=[r"\berror\b", r"\bfailed\b"],
        )

        # Responses should address the greeting appropriately
        assert_response_relevance(
            result_det + result_random,
            expected_topics=["hello", "greeting", "conversation"],
        )

    def test_vllm_top_p_parameter(self):
        """Test VLLM-specific top_p nucleus sampling parameter."""
        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])
        conversation = create_test_conversations()[0:1]

        gen_params = GenerationParams(
            max_new_tokens=20, temperature=0.7, top_p=0.9, seed=42, use_sampling=True
        )
        config = InferenceConfig(generation=gen_params)

        start_time = time.time()
        result = engine.infer(conversation, config)
        elapsed_time = time.time() - start_time

        # Basic validation
        assert validate_generation_output(result)

        # Enhanced validation for nucleus sampling
        assert_response_properties(
            result,
            min_length=5,
            max_length=400,
            forbidden_patterns=[r"\berror\b", r"\bfailed\b"],
        )

        # Should address the greeting appropriately
        assert_response_relevance(result)

        # Performance validation
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time, tokens_generated, max_time_seconds=25.0, min_throughput=2.0
        )


class TestVLLMSpecificFeatures:
    """Test VLLM-specific features and optimizations."""

    def test_vllm_tensor_parallel_single_gpu(self):
        """Test tensor parallelism configuration with single GPU."""

        models = get_test_models()
        model_params = models["smollm_135m"]

        # Add VLLM-specific configuration for single GPU
        model_params.model_kwargs = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
        }

        engine = VLLMInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)

    def test_vllm_memory_optimization(self):
        """Test GPU memory utilization settings."""

        models = get_test_models()
        model_params = models["smollm_135m"]

        # Test with conservative memory usage
        model_params.model_kwargs = {
            "gpu_memory_utilization": 0.6,
            "max_num_seqs": 16,  # Limit concurrent sequences
        }

        engine = VLLMInferenceEngine(model_params)

        conversations = create_test_conversations()[:2]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)

    def test_vllm_block_size_configuration(self):
        """Test attention block size configuration."""

        models = get_test_models()
        model_params = models["smollm_135m"]

        # Test with specific block size
        model_params.model_kwargs = {
            "block_size": 16,  # Smaller block size
            "gpu_memory_utilization": 0.7,
        }

        engine = VLLMInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)


class TestVLLMErrorHandling(AbstractInferenceEngineErrorHandling):
    """Test VLLM error handling and edge cases using abstract base class.

    This class inherits standard error handling tests and adds VLLM-specific
    error scenarios.
    """

    def get_engine_class(self) -> type:
        """Return the VLLM inference engine class."""
        return VLLMInferenceEngine

    def get_default_model_key(self) -> str:
        """Return the default model key for VLLM testing."""
        return "smollm_135m"

    def get_performance_thresholds(self) -> dict[str, Any]:
        """Return VLLM-specific performance expectations."""
        return {
            "max_time_seconds": 30.0,
            "min_throughput": 2.0,
            "batch_size": 2,
        }

    # VLLM-specific error handling tests

    def test_vllm_invalid_generation_params(self):
        """Test error handling for invalid generation parameters."""

        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])

        conversations = create_test_conversations()[:1]

        # Test with invalid temperature
        invalid_gen_params = GenerationParams(
            max_new_tokens=10,
            temperature=-1.0,  # Invalid negative temperature
            seed=42,
        )
        invalid_config = InferenceConfig(generation=invalid_gen_params)

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, Exception)):
            engine.infer(conversations, invalid_config)

    def test_vllm_extremely_long_sequence(self):
        """Test handling of very long input sequences."""

        models = get_test_models()
        model_params = models["smollm_135m"]

        # Set reasonable context window
        model_params.model_max_length = 1024

        engine = VLLMInferenceEngine(model_params)

        # Create very long input that approaches context limit
        long_prompt = "Tell me about " + " ".join(["artificial intelligence"] * 100)
        from oumi.core.types.conversation import Conversation, Message, Role

        long_conversation = [
            Conversation(
                conversation_id="long_test",
                messages=[Message(content=long_prompt, role=Role.USER)],
            )
        ]

        inference_config = InferenceConfig(
            generation=GenerationParams(
                max_new_tokens=10,  # Keep response short
                temperature=0.0,
            )
        )

        # Should handle gracefully (truncate or error appropriately)
        result = engine.infer(long_conversation, inference_config)
        # If it succeeds, validate the output
        assert validate_generation_output(result)


class TestVLLMPerformance:
    """Test VLLM performance characteristics."""

    @requires_cuda_initialized()
    @requires_gpus(1, min_gb=5.0)  # Need 5GB VRAM
    @pytest.mark.slow_integration
    def test_vllm_concurrent_requests(self):
        """Test handling of concurrent inference requests."""

        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])

        # Test with multiple separate inference calls
        conversations1 = create_test_conversations()[:2]
        conversations2 = create_batch_conversations(3, "Describe")

        inference_config = InferenceConfig(generation=get_test_generation_params())

        # Run concurrent inferences
        result1 = engine.infer(conversations1, inference_config)
        result2 = engine.infer(conversations2, inference_config)

        # Both should succeed
        assert validate_generation_output(result1)
        assert validate_generation_output(result2)
        assert len(result1) == len(conversations1)
        assert len(result2) == len(conversations2)
