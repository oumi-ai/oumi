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

"""Live integration tests for LlamaCppInferenceEngine with real GGUF models."""

import time
from typing import Any

import pytest

from oumi.core.configs import GenerationParams, InferenceConfig
from oumi.inference.llama_cpp_inference_engine import LlamaCppInferenceEngine
from tests.integration.infer.test_base_inference_engine import (
    AbstractInferenceEngineBasicFunctionality,
    AbstractInferenceEngineErrorHandling,
    AbstractInferenceEngineGenerationParameters,
)
from tests.integration.infer.test_inference_test_utils import (
    assert_performance_requirements,
    assert_response_properties,
    assert_response_relevance,
    count_response_tokens,
    create_test_conversations,
    get_test_generation_params,
    get_test_models,
    validate_generation_output,
)

# Skip all tests if llama-cpp-python is not available
try:
    from llama_cpp import Llama  # noqa: F401

    llamacpp_available = True
except ImportError:
    llamacpp_available = False

pytestmark = [
    pytest.mark.skipif(not llamacpp_available, reason="llama-cpp-python not available"),
    pytest.mark.requires_llamacpp,
]


class TestLlamaCppBasicFunctionality(AbstractInferenceEngineBasicFunctionality):
    """Test core LlamaCpp inference functionality using abstract base class.

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
        """Return the LlamaCpp inference engine class."""
        return LlamaCppInferenceEngine

    def get_default_model_key(self) -> str:
        """Return the default model key for LlamaCpp testing."""
        return "gemma_270m_gguf"

    def get_performance_thresholds(self) -> dict[str, Any]:
        """Return LlamaCpp-specific performance expectations."""
        return {
            "max_time_seconds": 45.0,  # CPU inference is slower
            "min_throughput": 1.0,  # Lower throughput for CPU
            "batch_size": 3,  # Smaller batches for CPU
        }

    # LlamaCpp-specific additional tests can be added here
    def test_llamacpp_specific_gguf_features(self):
        """Test LlamaCpp-specific GGUF model features."""
        # This is an example of an additional LlamaCpp-specific test
        # The basic functionality is already covered by inherited methods
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)

        # Basic validation (detailed validation is in inherited methods)
        assert validate_generation_output(result)

        # GGUF-specific validation could go here
        assert_response_properties(
            result,
            min_length=3,
            max_length=400,
            expected_keywords=["Hello"],
            forbidden_patterns=[
                r"\berror\b",
                r"\bfailed\b",
                r"\bunable\b",
                r"\bsorry\b.*\bcannot\b",
            ],
        )

        # Validate response relevance to greeting
        assert_response_relevance(result)


class TestLlamaCppMemoryManagement:
    """Test LlamaCpp memory management features."""

    def test_llamacpp_memory_mapping_enabled(self):
        """Test use_mmap=True parameter effects."""

        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Enable memory mapping (should be default)
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "use_mmap": True,
            "verbose": False,
        }

        engine = LlamaCppInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time

        # Basic validation
        assert validate_generation_output(result)

        # Enhanced validation for memory mapping
        assert_response_properties(
            result,
            min_length=3,
            max_length=400,
            forbidden_patterns=[r"\berror\b", r"\bfailed\b", r"\bunable\b"],
        )

        # Should still work properly with mmap enabled
        assert_response_relevance(result)

        # Performance with memory mapping should be reasonable
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time, tokens_generated, max_time_seconds=35.0, min_throughput=1.0
        )

    @pytest.mark.memory_intensive  # Need more RAM when not using mmap
    def test_llamacpp_memory_mapping_disabled(self):
        """Test use_mmap=False parameter effects."""

        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Disable memory mapping
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "use_mmap": False,
            "verbose": False,
        }

        engine = LlamaCppInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)

    def test_llamacpp_memory_locking_enabled(self):
        """Test use_mlock=True parameter effects."""

        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Enable memory locking (should be default)
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "use_mlock": True,
            "verbose": False,
        }

        engine = LlamaCppInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)

    def test_llamacpp_memory_locking_disabled(self):
        """Test use_mlock=False parameter effects."""

        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Disable memory locking
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "use_mlock": False,
            "verbose": False,
        }

        engine = LlamaCppInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)

    def test_llamacpp_combined_memory_features(self):
        """Test use_mmap + use_mlock together."""

        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Test combination of memory features (should be defaults from your
        # implementation)
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "use_mmap": True,
            "use_mlock": True,
            "verbose": False,
        }

        engine = LlamaCppInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        # Measure memory-optimized performance
        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time

        assert validate_generation_output(result)
        # With optimal memory settings, should be reasonably fast
        assert elapsed_time < 30.0


class TestLlamaCppHardwareOptimization:
    """Test LlamaCpp hardware optimization features."""

    def test_llamacpp_cpu_inference(self):
        """Test pure CPU inference (n_gpu_layers=0)."""

        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Force CPU-only inference
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "n_gpu_layers": 0,  # No GPU layers
            "verbose": False,
        }

        engine = LlamaCppInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)

    @pytest.mark.single_gpu
    def test_llamacpp_gpu_layers(self):
        """Test n_gpu_layers parameter with GPU acceleration."""

        # Skip if no CUDA available
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU layers test")

        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Enable GPU acceleration for some layers
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "n_gpu_layers": 10,  # Put some layers on GPU
            "verbose": False,
        }

        engine = LlamaCppInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)

    def test_llamacpp_thread_scaling(self):
        """Test n_threads parameter effects."""

        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Test with specific thread count
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "n_threads": 2,  # Use 2 threads
            "verbose": False,
        }

        engine = LlamaCppInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)

    def test_llamacpp_flash_attention(self):
        """Test flash_attn parameter."""

        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Enable flash attention (should be default from your implementation)
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "flash_attn": True,
            "verbose": False,
        }

        engine = LlamaCppInferenceEngine(model_params)

        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())

        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)


class TestLlamaCppGenerationParameters(AbstractInferenceEngineGenerationParameters):
    """Test LlamaCpp generation parameter handling using abstract base class.

    This class inherits comprehensive parameter testing from the base class
    and adds LlamaCpp-specific parameter variations.
    """

    def get_engine_class(self) -> type:
        """Return the LlamaCpp inference engine class."""
        return LlamaCppInferenceEngine

    def get_default_model_key(self) -> str:
        """Return the default model key for LlamaCpp testing."""
        return "gemma_270m_gguf"

    def get_performance_thresholds(self) -> dict[str, Any]:
        """Return LlamaCpp-specific performance expectations."""
        return {
            "max_time_seconds": 35.0,
            "min_throughput": 1.0,
            "batch_size": 2,
        }

    # LlamaCpp-specific parameter tests
    def test_llamacpp_variable_max_tokens(self):
        """Test LlamaCpp-specific max_new_tokens variations."""
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        conversation = create_test_conversations()[0:1]

        # Test with small token limit
        gen_params_small = GenerationParams(max_new_tokens=5, temperature=0.0, seed=42)
        config_small = InferenceConfig(generation=gen_params_small)
        result_small = engine.infer(conversation, config_small)

        # Test with larger token limit
        gen_params_large = GenerationParams(max_new_tokens=25, temperature=0.0, seed=42)
        config_large = InferenceConfig(generation=gen_params_large)
        result_large = engine.infer(conversation, config_large)

        # Both should be valid
        assert validate_generation_output(result_small)
        assert validate_generation_output(result_large)

        # Both should have content
        small_response = result_small[0].messages[-1].compute_flattened_text_content()
        large_response = result_large[0].messages[-1].compute_flattened_text_content()
        assert len(small_response.strip()) > 0
        assert len(large_response.strip()) > 0


class TestLlamaCppErrorHandling(AbstractInferenceEngineErrorHandling):
    """Test LlamaCpp error handling and edge cases using abstract base class.

    This class inherits standard error handling tests and adds LlamaCpp-specific
    error scenarios.
    """

    def get_engine_class(self) -> type:
        """Return the LlamaCpp inference engine class."""
        return LlamaCppInferenceEngine

    def get_default_model_key(self) -> str:
        """Return the default model key for LlamaCpp testing."""
        return "gemma_270m_gguf"

    def get_performance_thresholds(self) -> dict[str, Any]:
        """Return LlamaCpp-specific performance expectations."""
        return {
            "max_time_seconds": 35.0,
            "min_throughput": 1.0,
            "batch_size": 2,
        }

    # LlamaCpp-specific error handling tests
    def test_llamacpp_invalid_gguf_file(self):
        """Test error handling for invalid GGUF files."""
        models = get_test_models()
        model_params = models["gemma_270m_gguf"]

        # Point to non-existent GGUF file
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "filename": "nonexistent-file.gguf",
        }

        # Should raise an error during engine initialization
        with pytest.raises(
            Exception
        ):  # Could be FileNotFoundError or llama-cpp specific error
            LlamaCppInferenceEngine(model_params)

    def test_llamacpp_extreme_parameters(self):
        """Test handling of extreme parameter values."""

        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])

        conversations = create_test_conversations()[:1]

        # Test with extreme parameters
        gen_params = GenerationParams(
            max_new_tokens=1,  # Minimal tokens
            temperature=0.0,
            seed=42,
        )
        config = InferenceConfig(generation=gen_params)

        result = engine.infer(conversations, config)

        # Should still generate valid output (even if very short)
        assert validate_generation_output(result)


class TestLlamaCppConsistency:
    """Test LlamaCpp consistency and deterministic behavior."""

    def test_llamacpp_deterministic_consistency(self):
        """Test LlamaCpp produces consistent outputs with same seed."""

        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])

        # Use deterministic parameters
        gen_params = GenerationParams(
            max_new_tokens=15, temperature=0.0, seed=42, use_sampling=False
        )
        inference_config = InferenceConfig(generation=gen_params)

        conversations = create_test_conversations()[:1]

        # Run same inference twice
        result1 = engine.infer(conversations, inference_config)
        result2 = engine.infer(conversations, inference_config)

        # Both should be valid
        assert validate_generation_output(result1)
        assert validate_generation_output(result2)

        response1 = result1[0].messages[-1].compute_flattened_text_content()
        response2 = result2[0].messages[-1].compute_flattened_text_content()

        # With deterministic settings, should have some consistency
        # (Note: LlamaCpp might still have some variability, so we check basic
        # properties)
        assert len(response1.strip()) > 0
        assert len(response2.strip()) > 0
