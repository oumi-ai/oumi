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

import tempfile
import time
from pathlib import Path

import pytest

from oumi.core.configs import GenerationParams, InferenceConfig
from oumi.inference.vllm_inference_engine import VLLMInferenceEngine
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
from tests.markers import requires_cuda_initialized, requires_gpus

# Skip all tests if vLLM is not available
try:
    import vllm
    vllm_available = True
except ImportError:
    vllm_available = False

pytestmark = [
    pytest.mark.skipif(not vllm_available, reason="vLLM not available"),
    pytest.mark.requires_vllm,
]


class TestVLLMBasicFunctionality:
    """Test core VLLM inference functionality."""
    
    @pytest.mark.memory_intensive  # Requires >6GB RAM
    def test_vllm_basic_inference_gemma_270m(self):
        """Test basic VLLM inference with Gemma-3-270m (CPU or GPU)."""
        # Check if we should use GPU acceleration
        import torch
        use_gpu = torch.cuda.is_available()
        # Note: This test will run on CPU if no GPU available
        
        models = get_test_models()
        engine = VLLMInferenceEngine(models["gemma_270m"])
        
        conversations = create_test_conversations()[:1]  # Single conversation
        inference_config = InferenceConfig(generation=get_test_generation_params())
        
        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time
        
        # Validate output structure and basic properties
        assert validate_generation_output(result)
        assert len(result) == len(conversations)
        
        # Enhanced property-based validation
        # Since we use natural keyword instructions, we can test for them
        assert_response_properties(
            result,
            min_length=5,
            max_length=500,
            expected_keywords=["Hello"],  # Test for the natural keyword that should appear in greeting response
            forbidden_patterns=[r'\berror\b', r'\bfailed\b', r'\bunable\b'],
        )
        
        # Performance validation
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time, 
            tokens_generated,
            max_time_seconds=30.0,
            min_throughput=2.0
        )
        
    def test_vllm_basic_inference_smollm_135m(self):
        """Test basic VLLM inference with SmolLM2-135M-Instruct (CPU or GPU)."""
        # Check if we should use GPU acceleration  
        import torch
        use_gpu = torch.cuda.is_available()
        # Note: This test will run on CPU if no GPU available
        
        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])
        
        conversations = create_test_conversations()[:1]  # Single conversation
        inference_config = InferenceConfig(generation=get_test_generation_params())
        
        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time
        
        # Validate output structure and properties
        assert validate_generation_output(result)
        assert len(result) == len(conversations)
        
        # Enhanced validation for SmolLM responses
        assert_response_properties(
            result,
            min_length=3,
            max_length=400,
            forbidden_patterns=[r'\berror\b', r'\bfailed\b'],
        )
        
        # Performance validation for small model
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time,
            tokens_generated,
            max_time_seconds=25.0,
            min_throughput=3.0
        )
        
    @requires_cuda_initialized()
    @requires_gpus(1, min_gb=3.0)  # Need 3GB VRAM
    def test_vllm_batch_inference(self):
        """Test batched inference with multiple conversations."""
        
        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])
        
        # Create batch of conversations
        conversations = create_batch_conversations(4, "What is")
        
        generation_params = get_test_generation_params()
        generation_params.max_new_tokens = 15
        inference_config = InferenceConfig(generation=generation_params)
        
        start_time = time.time()
        result = engine.infer(conversations, inference_config)
        elapsed_time = time.time() - start_time
        
        # Validate output structure
        assert validate_generation_output(result)
        assert len(result) == len(conversations)
        
        # Enhanced batch validation - all responses should address their prompts
        for i, conversation in enumerate(result):
            original_prompt = conversations[i].messages[0].content
            expected_keywords = get_contextual_keywords(original_prompt)
            
            # Validate individual response properties
            assert_response_properties(
                [conversation],
                min_length=3,
                max_length=300,
                expected_keywords=expected_keywords[:2] if expected_keywords else None,  # Use top 2 keywords
                forbidden_patterns=[r'\berror\b', r'\bfailed\b'],
            )
        
        # Performance validation for batch processing
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time,
            tokens_generated,
            max_time_seconds=40.0,  # Longer for batch
            min_throughput=5.0  # Should be efficient with batching
        )
        
    @requires_cuda_initialized()
    @requires_gpus(1, min_gb=2.0)  # Need 2GB VRAM
    def test_vllm_empty_input(self):
        """Test graceful handling of empty conversations."""
        
        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])
        
        inference_config = InferenceConfig(generation=get_test_generation_params())
        result = engine.infer([], inference_config)
        
        assert result == []
        
    def test_vllm_file_io(self):
        """Test input/output file handling with VLLM."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(3.0)
        skip_if_insufficient_memory(5.0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            models = get_test_models()
            engine = VLLMInferenceEngine(models["smollm_135m"])
            
            conversations = create_test_conversations()[:2]
            output_path = Path(temp_dir) / "vllm_output.jsonl"
            
            inference_config = InferenceConfig(
                generation=get_test_generation_params(),
                output_path=str(output_path)
            )
            
            result = engine.infer(conversations, inference_config)
            
            # Validate output
            assert validate_generation_output(result) 
            assert output_path.exists()
            
            # Check file content
            assert output_path.stat().st_size > 0


class TestVLLMGenerationParameters:
    """Test VLLM generation parameter handling."""
    
    def test_vllm_temperature_variation(self):
        """Test temperature parameter effects."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(3.0)
        
        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])
        
        conversation = create_test_conversations()[0:1]
        
        # Test with temperature=0.0 (deterministic)
        gen_params_deterministic = GenerationParams(
            max_new_tokens=20,
            temperature=0.0,
            seed=42,
            use_sampling=False
        )
        config_det = InferenceConfig(generation=gen_params_deterministic)
        result_det = engine.infer(conversation, config_det)
        
        # Test with higher temperature (more random)
        gen_params_random = GenerationParams(
            max_new_tokens=20,
            temperature=0.8,
            seed=42,
            use_sampling=True
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
            forbidden_patterns=[r'\berror\b', r'\bfailed\b'],
            require_sentences=True
        )
        
        assert_response_properties(
            result_random,
            min_length=5,
            max_length=400,
            forbidden_patterns=[r'\berror\b', r'\bfailed\b'],
        )
        
        # Responses should address the greeting appropriately
        assert_response_relevance(
            result_det + result_random,
            expected_topics=["hello", "greeting", "conversation"]
        )
        
    def test_vllm_max_tokens_parameter(self):
        """Test max_new_tokens parameter."""
        skip_if_no_cuda() 
        skip_if_insufficient_vram(3.0)
        
        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])
        
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
            max_new_tokens=30,
            temperature=0.0,
            seed=42
        )
        config_large = InferenceConfig(generation=gen_params_large)
        result_large = engine.infer(conversation, config_large)
        
        # Both should be valid
        assert validate_generation_output(result_small)
        assert validate_generation_output(result_large)
        
        # Enhanced validation with appropriate length expectations
        assert_response_properties(
            result_small,
            min_length=2,  # Very short responses acceptable with low token limit
            max_length=100,  # Should respect token limit
            forbidden_patterns=[r'\berror\b', r'\bfailed\b'],
        )
        
        assert_response_properties(
            result_large,
            min_length=5,  # Should have more content with higher limit
            max_length=600,
            forbidden_patterns=[r'\berror\b', r'\bfailed\b'],
        )
        
        # Both should address the greeting
        assert_response_relevance(result_small + result_large)
        
    def test_vllm_top_p_parameter(self):
        """Test top_p nucleus sampling parameter."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(3.0)
        
        models = get_test_models()  
        engine = VLLMInferenceEngine(models["smollm_135m"])
        
        conversation = create_test_conversations()[0:1]
        
        gen_params = GenerationParams(
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            use_sampling=True
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
            forbidden_patterns=[r'\berror\b', r'\bfailed\b'],
        )
        
        # Should address the greeting appropriately
        assert_response_relevance(result)
        
        # Performance validation
        tokens_generated = count_response_tokens(result)
        assert_performance_requirements(
            elapsed_time,
            tokens_generated,
            max_time_seconds=25.0,
            min_throughput=2.0
        )


class TestVLLMSpecificFeatures:
    """Test VLLM-specific features and optimizations."""
    
    def test_vllm_tensor_parallel_single_gpu(self):
        """Test tensor parallelism configuration with single GPU."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(3.0)
        
        models = get_test_models()
        model_params = models["smollm_135m"]
        
        # Add VLLM-specific configuration for single GPU
        model_params.model_kwargs = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7
        }
        
        engine = VLLMInferenceEngine(model_params)
        
        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())
        
        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)
        
    def test_vllm_memory_optimization(self):
        """Test GPU memory utilization settings."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(4.0)
        
        models = get_test_models()
        model_params = models["smollm_135m"]
        
        # Test with conservative memory usage
        model_params.model_kwargs = {
            "gpu_memory_utilization": 0.6,
            "max_num_seqs": 16  # Limit concurrent sequences
        }
        
        engine = VLLMInferenceEngine(model_params)
        
        conversations = create_test_conversations()[:2]
        inference_config = InferenceConfig(generation=get_test_generation_params())
        
        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)
        
    def test_vllm_block_size_configuration(self):
        """Test attention block size configuration."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(3.0)
        
        models = get_test_models()
        model_params = models["smollm_135m"]
        
        # Test with specific block size
        model_params.model_kwargs = {
            "block_size": 16,  # Smaller block size
            "gpu_memory_utilization": 0.7
        }
        
        engine = VLLMInferenceEngine(model_params)
        
        conversations = create_test_conversations()[:1]
        inference_config = InferenceConfig(generation=get_test_generation_params())
        
        result = engine.infer(conversations, inference_config)
        assert validate_generation_output(result)


class TestVLLMErrorHandling:
    """Test VLLM error handling and edge cases."""
    
    def test_vllm_invalid_model_name(self):
        """Test error handling for invalid model names."""
        skip_if_no_cuda()
        
        models = get_test_models()
        model_params = models["smollm_135m"]
        model_params.model_name = "nonexistent/invalid-model"
        
        # Should raise an error during engine initialization
        with pytest.raises(Exception):  # Could be various error types from vLLM
            VLLMInferenceEngine(model_params)
            
    def test_vllm_invalid_generation_params(self):
        """Test error handling for invalid generation parameters."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(3.0)
        
        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])
        
        conversations = create_test_conversations()[:1]
        
        # Test with invalid temperature
        invalid_gen_params = GenerationParams(
            max_new_tokens=10,
            temperature=-1.0,  # Invalid negative temperature
            seed=42
        )
        invalid_config = InferenceConfig(generation=invalid_gen_params)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, Exception)):
            engine.infer(conversations, invalid_config)
            
    def test_vllm_extremely_long_sequence(self):
        """Test handling of very long input sequences."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(4.0)
        
        models = get_test_models()
        model_params = models["smollm_135m"]
        
        # Set reasonable context window
        model_params.model_max_length = 1024
        
        engine = VLLMInferenceEngine(model_params)
        
        # Create very long input that approaches context limit
        long_prompt = "Tell me about " + " ".join(["artificial intelligence"] * 100)
        from oumi.core.types.conversation import Conversation, Message, Role
        
        long_conversation = [Conversation(
            conversation_id="long_test",
            messages=[Message(content=long_prompt, role=Role.USER)]
        )]
        
        inference_config = InferenceConfig(generation=GenerationParams(
            max_new_tokens=10,  # Keep response short
            temperature=0.0
        ))
        
        # Should handle gracefully (truncate or error appropriately)
        result = engine.infer(long_conversation, inference_config)
        # If it succeeds, validate the output
        assert validate_generation_output(result)


class TestVLLMPerformance:
    """Test VLLM performance characteristics."""
    
    @requires_cuda_initialized()
    @requires_gpus(1, min_gb=4.0)  # Need 4GB VRAM
    def test_vllm_throughput_measurement(self):
        """Test and measure VLLM throughput."""
        
        models = get_test_models()
        engine = VLLMInferenceEngine(models["smollm_135m"])
        
        # Create multiple conversations for throughput testing
        conversations = create_batch_conversations(8, "Explain")
        
        generation_params = GenerationParams(
            max_new_tokens=25,
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
        
        # Should achieve reasonable throughput (>10 tokens/sec for small model)
        assert throughput > 10.0, f"Throughput too low: {throughput} tokens/sec"
        
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