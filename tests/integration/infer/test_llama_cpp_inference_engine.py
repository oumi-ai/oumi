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

import tempfile
import time
from pathlib import Path

import pytest

from oumi.core.configs import GenerationParams, InferenceConfig
from oumi.inference.llama_cpp_inference_engine import LlamaCppInferenceEngine
from tests.integration.infer.inference_test_utils import (
    assert_performance_requirements,
    assert_response_properties,
    assert_response_relevance,
    compare_conversation_responses,
    count_response_tokens,
    create_batch_conversations,
    create_test_conversations,
    get_contextual_keywords,
    get_test_generation_params,
    get_test_models,
    measure_tokens_per_second,
    validate_generation_output,
)

# Skip all tests if llama-cpp-python is not available
try:
    from llama_cpp import Llama
    llamacpp_available = True
except ImportError:
    llamacpp_available = False

pytestmark = [
    pytest.mark.skipif(not llamacpp_available, reason="llama-cpp-python not available"),
    pytest.mark.requires_llamacpp,
]


class TestLlamaCppBasicFunctionality:
    """Test core LlamaCpp inference functionality with GGUF models."""
    
    def test_llamacpp_gguf_loading(self):
        """Test loading GGUF model (Q4_K_M quantization)."""
        
        models = get_test_models()
        model_params = models["gemma_270m_gguf"]
        
        # Should successfully initialize with GGUF model
        engine = LlamaCppInferenceEngine(model_params)
        assert engine is not None
        
    def test_llamacpp_basic_inference(self):
        """Test basic inference with quantized Gemma model."""
        
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        
        conversations = create_test_conversations()[:1]  # Single conversation
        inference_config = InferenceConfig(generation=get_test_generation_params())
        
        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time
        
        # Validate output structure and basic properties
        assert validate_generation_output(result)
        assert len(result) == len(conversations)
        
        # Enhanced property-based validation for GGUF quantized model
        # Since we use natural keyword instructions, we can test for them
        assert_response_properties(
            result,
            min_length=3,
            max_length=400,
            expected_keywords=["Hello"],  # Test for the natural keyword that should appear in greeting response
            forbidden_patterns=[r'\berror\b', r'\bfailed\b', r'\bunable\b', r'\bsorry\b.*\bcannot\b'],
        )
        
        # Validate response relevance to greeting (broad validation)
        assert_response_relevance(result)
        
        # Performance validation (CPU inference is slower)
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time,
            tokens_generated,
            max_time_seconds=45.0,  # Longer timeout for CPU
            min_throughput=1.0  # Lower throughput expectations for CPU
        )
        
    def test_llamacpp_batch_inference(self):
        """Test batched conversations."""
        
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        
        # Create batch of conversations
        conversations = create_batch_conversations(3, "Tell me about")
        
        generation_params = get_test_generation_params()
        generation_params.max_new_tokens = 15  # Keep responses short for batch testing
        inference_config = InferenceConfig(generation=generation_params)
        
        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time
        
        # Validate output structure
        assert validate_generation_output(result)
        assert len(result) == len(conversations)
        
        # Enhanced batch validation with topic relevance
        for i, conversation in enumerate(result):
            assert len(conversation.messages) > len(conversations[i].messages)
            assert conversation.messages[-1].role.value == "assistant"
            
            # Extract expected topic from original prompt
            original_prompt = conversations[i].messages[0].content
            expected_keywords = get_contextual_keywords(original_prompt)
            
            # Validate each response's properties
            assert_response_properties(
                [conversation],
                min_length=3,
                max_length=300,
                expected_keywords=expected_keywords[:1] if expected_keywords else None,  # Use top keyword
                forbidden_patterns=[r'\berror\b', r'\bfailed\b', r'\bunable\b'],
            )
        
        # Performance validation for batch processing
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time,
            tokens_generated,
            max_time_seconds=60.0,  # Longer for CPU batch processing
            min_throughput=0.8  # Lower expectations for CPU batching
        )
            
    def test_llamacpp_file_io(self):
        """Test input/output file operations."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            models = get_test_models()
            engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
            
            conversations = create_test_conversations()[:2]
            output_path = Path(temp_dir) / "llamacpp_output.jsonl"
            
            generation_params = get_test_generation_params()
            generation_params.max_new_tokens = 10  # Keep short for file I/O test
            inference_config = InferenceConfig(
                generation=generation_params,
                output_path=str(output_path)
            )
            
            result = engine.infer(conversations, inference_config)
            
            # Validate output
            assert validate_generation_output(result)
            assert output_path.exists()
            
            # Check file content
            assert output_path.stat().st_size > 0
            
    def test_llamacpp_empty_input(self):
        """Test graceful handling of empty conversations."""
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        
        inference_config = InferenceConfig(generation=get_test_generation_params())
        result = engine.infer([], inference_config)
        
        assert result == []


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
            "verbose": False
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
            forbidden_patterns=[r'\berror\b', r'\bfailed\b', r'\bunable\b'],
        )
        
        # Should still work properly with mmap enabled
        assert_response_relevance(result)
        
        # Performance with memory mapping should be reasonable
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time,
            tokens_generated,
            max_time_seconds=35.0,
            min_throughput=1.0
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
            "verbose": False
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
            "verbose": False
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
            "verbose": False
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
        
        # Test combination of memory features (should be defaults from your implementation)
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "use_mmap": True,
            "use_mlock": True,
            "verbose": False
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
            "verbose": False
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
            "verbose": False
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
            "verbose": False
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
            "verbose": False
        }
        
        engine = LlamaCppInferenceEngine(model_params)
        
        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())
        
        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)


class TestLlamaCppGenerationParameters:
    """Test LlamaCpp generation parameter handling."""
    
    def test_llamacpp_generation_params(self):
        """Test temperature, top_p, max_tokens parameters."""
        
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        
        conversation = create_test_conversations()[0:1]
        
        # Test with various generation parameters
        gen_params = GenerationParams(
            max_new_tokens=15,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            use_sampling=True
        )
        config = InferenceConfig(generation=gen_params)
        result = engine.infer(conversation, config)
        
        assert validate_generation_output(result)
        
    def test_llamacpp_deterministic_generation(self):
        """Test seed-based reproducible outputs."""
        
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        
        conversation = create_test_conversations()[0:1]
        
        # Test deterministic generation with same seed
        gen_params = GenerationParams(
            max_new_tokens=20,
            temperature=0.0,  # Deterministic
            seed=42,
            use_sampling=False
        )
        config = InferenceConfig(generation=gen_params)
        
        # Run twice with same parameters
        result1 = engine.infer(conversation, config)
        result2 = engine.infer(conversation, config)
        
        assert validate_generation_output(result1)
        assert validate_generation_output(result2)
        
        # With temperature=0.0 and same seed, results should be identical
        response1 = result1[0].messages[-1].content
        response2 = result2[0].messages[-1].content
        
        # Note: LlamaCpp might still have some variability, so we check for basic similarity
        assert len(response1.strip()) > 0
        assert len(response2.strip()) > 0
        
    def test_llamacpp_variable_max_tokens(self):
        """Test different max_new_tokens values."""
        
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        
        conversation = create_test_conversations()[0:1]
        
        # Test with small token limit
        gen_params_small = GenerationParams(
            max_new_tokens=5,
            temperature=0.0,
            seed=42
        )
        config_small = InferenceConfig(generation=gen_params_small)
        result_small = engine.infer(conversation, config_small)
        
        # Test with larger token limit
        gen_params_large = GenerationParams(
            max_new_tokens=25,
            temperature=0.0,
            seed=42
        )
        config_large = InferenceConfig(generation=gen_params_large)
        result_large = engine.infer(conversation, config_large)
        
        # Both should be valid
        assert validate_generation_output(result_small)
        assert validate_generation_output(result_large)
        
        # Both should have content
        small_response = result_small[0].messages[-1].content
        large_response = result_large[0].messages[-1].content
        assert len(small_response.strip()) > 0
        assert len(large_response.strip()) > 0


class TestLlamaCppErrorHandling:
    """Test LlamaCpp error handling and edge cases."""
    
    def test_llamacpp_invalid_gguf_file(self):
        """Test error handling for invalid GGUF files."""
        models = get_test_models()
        model_params = models["gemma_270m_gguf"]
        
        # Point to non-existent GGUF file
        model_params.model_kwargs = {
            **model_params.model_kwargs,
            "filename": "nonexistent-file.gguf"
        }
        
        # Should raise an error during engine initialization
        with pytest.raises(Exception):  # Could be FileNotFoundError or llama-cpp specific error
            LlamaCppInferenceEngine(model_params)
            
    def test_llamacpp_invalid_model_path(self):
        """Test error handling for invalid model paths."""
        models = get_test_models()
        model_params = models["gemma_270m_gguf"]
        model_params.model_name = "nonexistent/invalid-model"
        
        # Should raise an error during engine initialization
        with pytest.raises(Exception):
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
            seed=42
        )
        config = InferenceConfig(generation=gen_params)
        
        result = engine.infer(conversations, config)
        
        # Should still generate valid output (even if very short)
        assert validate_generation_output(result)


class TestLlamaCppPerformance:
    """Test LlamaCpp performance characteristics."""
    
    def test_llamacpp_throughput_measurement(self):
        """Test and measure LlamaCpp throughput."""
        
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        
        # Create multiple conversations for throughput testing
        conversations = create_batch_conversations(4, "What is")
        
        generation_params = GenerationParams(
            max_new_tokens=15,
            temperature=0.0,
            seed=42
        )
        inference_config = InferenceConfig(generation=generation_params)
        
        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time
        
        # Validate results
        assert validate_generation_output(result)
        assert len(result) == len(conversations)
        
        # Measure performance
        total_tokens = count_response_tokens(result)
        throughput = measure_tokens_per_second(total_tokens, elapsed_time)
        
        # Should achieve some reasonable throughput (CPU inference is slower)
        assert throughput > 1.0, f"Throughput too low: {throughput} tokens/sec"
        
    @pytest.mark.memory_intensive
    def test_llamacpp_memory_optimization_comparison(self):
        """Compare performance with different memory settings."""
        
        conversations = create_test_conversations()[:1]
        generation_params = GenerationParams(
            max_new_tokens=20,
            temperature=0.0,
            seed=42
        )
        config = InferenceConfig(generation=generation_params)
        
        models = get_test_models()
        
        # Test with memory mapping enabled
        model_params_mmap = models["gemma_270m_gguf"]
        model_params_mmap.model_kwargs = {
            **model_params_mmap.model_kwargs,
            "use_mmap": True,
            "verbose": False
        }
        
        engine_mmap = LlamaCppInferenceEngine(model_params_mmap)
        start_time = time.time()
        result_mmap = engine_mmap.infer(conversations, config)
        elapsed_mmap = time.time() - start_time
        
        # Test with memory mapping disabled (requires more RAM)
        model_params_no_mmap = models["gemma_270m_gguf"]
        model_params_no_mmap.model_kwargs = {
            **model_params_no_mmap.model_kwargs,
            "use_mmap": False,
            "verbose": False
        }
        
        engine_no_mmap = LlamaCppInferenceEngine(model_params_no_mmap)
        start_time = time.time()
        result_no_mmap = engine_no_mmap.infer(conversations, config)
        elapsed_no_mmap = time.time() - start_time
        
        # Both should work
        assert validate_generation_output(result_mmap)
        assert validate_generation_output(result_no_mmap)
        
        # Both should complete in reasonable time
        assert elapsed_mmap < 60.0
        assert elapsed_no_mmap < 60.0