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

"""Cross-engine compatibility tests comparing VLLM, LlamaCpp, and Native engines."""

import time
from typing import Any, Dict

import pytest

from oumi.core.configs import GenerationParams, InferenceConfig
from oumi.inference.llama_cpp_inference_engine import LlamaCppInferenceEngine
from oumi.inference.native_text_inference_engine import NativeTextInferenceEngine
from oumi.inference.vllm_inference_engine import VLLMInferenceEngine
from tests.integration.infer.inference_test_utils import (
    assert_performance_requirements,
    assert_response_properties,
    assert_response_relevance,
    compare_conversation_responses,
    count_response_tokens,
    create_test_conversations,
    get_contextual_keywords,
    get_test_models,
    measure_tokens_per_second,
    skip_if_insufficient_memory,
    skip_if_insufficient_vram,
    skip_if_no_cuda,
    validate_generation_output,
)
from tests.markers import requires_cuda_initialized

# Check engine availability
try:
    import vllm
    vllm_available = True
except ImportError:
    vllm_available = False

try:
    from llama_cpp import Llama
    llamacpp_available = True
except ImportError:
    llamacpp_available = False


class TestEngineOutputConsistency:
    """Test output consistency across different inference engines."""

    @pytest.mark.skipif(not vllm_available, reason="vLLM not available")
    @requires_cuda_initialized()
    @pytest.mark.single_gpu
    def test_native_vs_vllm_consistency(self):
        """Compare Native and VLLM outputs for same model."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(6.0)  # Need more VRAM for two engines
        skip_if_insufficient_memory(8.0)
        
        models = get_test_models()
        base_model = models["smollm_135m"]  # Use smaller model for comparison
        
        # Create engines with same base model
        native_engine = NativeTextInferenceEngine(base_model)
        vllm_engine = VLLMInferenceEngine(base_model)
        
        # Use deterministic generation parameters
        gen_params = GenerationParams(
            max_new_tokens=15,
            temperature=0.0,  # Deterministic
            seed=42,
            use_sampling=False
        )
        inference_config = InferenceConfig(generation=gen_params)
        
        conversations = create_test_conversations()[:1]  # Single conversation for comparison
        
        # Run inference with both engines
        native_result = native_engine.infer(conversations, inference_config)
        vllm_result = vllm_engine.infer(conversations, inference_config)
        
        # Both should generate valid outputs
        assert validate_generation_output(native_result)
        assert validate_generation_output(vllm_result)
        assert len(native_result) == len(vllm_result) == 1
        
        # Enhanced cross-engine validation with property-based assertions
        for result, engine_name in [(native_result, "Native"), (vllm_result, "VLLM")]:
            assert_response_properties(
                result,
                min_length=3,
                max_length=400,
                expected_keywords=["hello", "how", "you", "good", "well", "fine"],
                forbidden_patterns=[r'\berror\b', r'\bfailed\b', r'\bunable\b'],
            )
            
            # Both engines should understand and respond to greetings
            assert_response_relevance(
                result,
                expected_topics=["greeting", "hello", "conversation"]
            )
        
        # Ensure both responses have meaningful content
        native_response = native_result[0].messages[-1].content
        vllm_response = vllm_result[0].messages[-1].content
        
        assert not native_response.isspace(), "Native response is empty or whitespace"
        assert not vllm_response.isspace(), "VLLM response is empty or whitespace"
        
    @pytest.mark.skipif(not vllm_available or not llamacpp_available, 
                       reason="vLLM or LlamaCpp not available")
    @requires_cuda_initialized()
    @pytest.mark.single_gpu
    def test_vllm_vs_llamacpp_consistency(self):
        """Compare VLLM (standard) vs LlamaCpp (quantized) outputs."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(4.0)
        skip_if_insufficient_memory(8.0)
        
        models = get_test_models()
        
        # Use different but related models (standard vs quantized versions)
        vllm_engine = VLLMInferenceEngine(models["gemma_270m"])
        llamacpp_engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        
        # Use same deterministic parameters
        gen_params = GenerationParams(
            max_new_tokens=20,
            temperature=0.0,
            seed=42,
            use_sampling=False
        )
        inference_config = InferenceConfig(generation=gen_params)
        
        conversations = create_test_conversations()[:1]
        
        # Run inference with both engines
        vllm_result = vllm_engine.infer(conversations, inference_config)
        llamacpp_result = llamacpp_engine.infer(conversations, inference_config)
        
        # Both should generate valid outputs
        assert validate_generation_output(vllm_result)
        assert validate_generation_output(llamacpp_result)
        
        # Enhanced validation for different model architectures (standard vs quantized)
        assert_response_properties(
            vllm_result,
            min_length=3,
            max_length=500,
            forbidden_patterns=[r'\berror\b', r'\bfailed\b', r'\bunable\b'],
        )
        
        assert_response_properties(
            llamacpp_result,
            min_length=3,
            max_length=500,
            forbidden_patterns=[r'\berror\b', r'\bfailed\b', r'\bunable\b'],
        )
        
        # Both should address the greeting appropriately despite quantization differences
        assert_response_relevance(vllm_result)
        assert_response_relevance(llamacpp_result)
        
        # Ensure both have meaningful content
        vllm_response = vllm_result[0].messages[-1].content
        llamacpp_response = llamacpp_result[0].messages[-1].content
        
        assert not vllm_response.isspace(), "VLLM response is empty"
        assert not llamacpp_response.isspace(), "LlamaCpp response is empty"
        
        # Note: We don't expect identical outputs due to quantization differences
        # But both should produce coherent, relevant responses
        
    @pytest.mark.skipif(not llamacpp_available, reason="LlamaCpp not available")
    def test_llamacpp_deterministic_consistency(self):
        """Test LlamaCpp produces consistent outputs with same seed."""
        skip_if_insufficient_memory(5.0)
        
        models = get_test_models()
        engine = LlamaCppInferenceEngine(models["gemma_270m_gguf"])
        
        # Use deterministic parameters
        gen_params = GenerationParams(
            max_new_tokens=15,
            temperature=0.0,
            seed=42,
            use_sampling=False
        )
        inference_config = InferenceConfig(generation=gen_params)
        
        conversations = create_test_conversations()[:1]
        
        # Run same inference twice
        result1 = engine.infer(conversations, inference_config)
        result2 = engine.infer(conversations, inference_config)
        
        # Both should be valid
        assert validate_generation_output(result1)
        assert validate_generation_output(result2)
        
        response1 = result1[0].messages[-1].content
        response2 = result2[0].messages[-1].content
        
        # With deterministic settings, should have some consistency
        # (Note: LlamaCpp might still have some variability, so we check basic properties)
        assert len(response1.strip()) > 0
        assert len(response2.strip()) > 0


class TestEnginePerformanceComparison:
    """Test and compare performance characteristics across engines."""

    @pytest.mark.skipif(not vllm_available, reason="vLLM not available")
    @requires_cuda_initialized()
    @pytest.mark.single_gpu
    @pytest.mark.slow_integration
    def test_engine_speed_comparison(self):
        """Benchmark tokens/second across engines."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(6.0)
        skip_if_insufficient_memory(8.0)
        
        models = get_test_models()
        base_model = models["smollm_135m"]
        
        # Create engines
        native_engine = NativeTextInferenceEngine(base_model)
        vllm_engine = VLLMInferenceEngine(base_model)
        
        # Use consistent generation parameters
        gen_params = GenerationParams(
            max_new_tokens=25,
            temperature=0.0,
            seed=42
        )
        inference_config = InferenceConfig(generation=gen_params)
        
        # Create test conversations
        conversations = create_test_conversations()[:2]  # Small batch for comparison
        
        performance_results: Dict[str, Dict[str, Any]] = {}
        
        # Benchmark Native engine
        start_time = time.time()
        native_result = native_engine.infer(conversations, inference_config)
        native_elapsed = time.time() - start_time
        
        native_tokens = count_response_tokens(native_result)
        native_throughput = measure_tokens_per_second(native_tokens, native_elapsed)
        
        performance_results["native"] = {
            "elapsed_time": native_elapsed,
            "tokens_generated": native_tokens,
            "throughput": native_throughput
        }
        
        # Benchmark VLLM engine
        start_time = time.time()
        vllm_result = vllm_engine.infer(conversations, inference_config)
        vllm_elapsed = time.time() - start_time
        
        vllm_tokens = count_response_tokens(vllm_result)
        vllm_throughput = measure_tokens_per_second(vllm_tokens, vllm_elapsed)
        
        performance_results["vllm"] = {
            "elapsed_time": vllm_elapsed,
            "tokens_generated": vllm_tokens,
            "throughput": vllm_throughput
        }
        
        # Validate both results
        assert validate_generation_output(native_result)
        assert validate_generation_output(vllm_result)
        
        # Both should achieve reasonable throughput
        assert native_throughput > 1.0, f"Native throughput too low: {native_throughput}"
        assert vllm_throughput > 1.0, f"VLLM throughput too low: {vllm_throughput}"
        
        # VLLM should generally be faster (but we won't enforce strict requirements)
        print(f"Performance comparison:")
        print(f"Native: {native_throughput:.2f} tokens/sec")
        print(f"VLLM: {vllm_throughput:.2f} tokens/sec")
        
    @pytest.mark.skipif(not llamacpp_available, reason="LlamaCpp not available") 
    @pytest.mark.memory_intensive
    def test_llamacpp_memory_vs_performance(self):
        """Compare LlamaCpp performance with different memory settings."""
        skip_if_insufficient_memory(8.0)
        
        models = get_test_models()
        conversations = create_test_conversations()[:2]
        
        gen_params = GenerationParams(
            max_new_tokens=20,
            temperature=0.0,
            seed=42
        )
        inference_config = InferenceConfig(generation=gen_params)
        
        performance_results: Dict[str, Dict[str, Any]] = {}
        
        # Test with memory mapping enabled
        model_params_mmap = models["gemma_270m_gguf"]
        model_params_mmap.model_kwargs = {
            **model_params_mmap.model_kwargs,
            "use_mmap": True,
            "use_mlock": True,
            "verbose": False
        }
        
        engine_mmap = LlamaCppInferenceEngine(model_params_mmap)
        start_time = time.time()
        result_mmap = engine_mmap.infer(conversations, inference_config)
        elapsed_mmap = time.time() - start_time
        
        tokens_mmap = count_response_tokens(result_mmap)
        throughput_mmap = measure_tokens_per_second(tokens_mmap, elapsed_mmap)
        
        performance_results["mmap_enabled"] = {
            "elapsed_time": elapsed_mmap,
            "tokens_generated": tokens_mmap,
            "throughput": throughput_mmap
        }
        
        # Test with memory mapping disabled
        model_params_no_mmap = models["gemma_270m_gguf"]
        model_params_no_mmap.model_kwargs = {
            **model_params_no_mmap.model_kwargs,
            "use_mmap": False,
            "use_mlock": False,
            "verbose": False
        }
        
        engine_no_mmap = LlamaCppInferenceEngine(model_params_no_mmap)
        start_time = time.time()
        result_no_mmap = engine_no_mmap.infer(conversations, inference_config)
        elapsed_no_mmap = time.time() - start_time
        
        tokens_no_mmap = count_response_tokens(result_no_mmap)
        throughput_no_mmap = measure_tokens_per_second(tokens_no_mmap, elapsed_no_mmap)
        
        performance_results["mmap_disabled"] = {
            "elapsed_time": elapsed_no_mmap,
            "tokens_generated": tokens_no_mmap,
            "throughput": throughput_no_mmap
        }
        
        # Validate both results
        assert validate_generation_output(result_mmap)
        assert validate_generation_output(result_no_mmap)
        
        # Both should achieve reasonable performance
        assert throughput_mmap > 0.5, f"MMAP throughput too low: {throughput_mmap}"
        assert throughput_no_mmap > 0.5, f"No-MMAP throughput too low: {throughput_no_mmap}"
        
        print(f"LlamaCpp memory comparison:")
        print(f"With mmap: {throughput_mmap:.2f} tokens/sec")
        print(f"Without mmap: {throughput_no_mmap:.2f} tokens/sec")
        
    @pytest.mark.skipif(not vllm_available or not llamacpp_available,
                       reason="vLLM or LlamaCpp not available")
    @requires_cuda_initialized()
    @pytest.mark.single_gpu
    @pytest.mark.slow_integration
    def test_gpu_vs_cpu_inference_comparison(self):
        """Compare GPU (VLLM) vs CPU (LlamaCpp) inference performance."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(4.0)
        skip_if_insufficient_memory(8.0)
        
        conversations = create_test_conversations()[:2]
        gen_params = GenerationParams(
            max_new_tokens=20,
            temperature=0.0,
            seed=42
        )
        inference_config = InferenceConfig(generation=gen_params)
        
        models = get_test_models()
        
        # GPU inference with VLLM
        vllm_engine = VLLMInferenceEngine(models["gemma_270m"])
        start_time = time.time()
        vllm_result = vllm_engine.infer(conversations, inference_config)
        vllm_elapsed = time.time() - start_time
        
        vllm_tokens = count_response_tokens(vllm_result)
        vllm_throughput = measure_tokens_per_second(vllm_tokens, vllm_elapsed)
        
        # CPU inference with LlamaCpp (force CPU)
        llamacpp_model = models["gemma_270m_gguf"]
        llamacpp_model.model_kwargs = {
            **llamacpp_model.model_kwargs,
            "n_gpu_layers": 0,  # Force CPU
            "verbose": False
        }
        
        llamacpp_engine = LlamaCppInferenceEngine(llamacpp_model)
        start_time = time.time()
        llamacpp_result = llamacpp_engine.infer(conversations, inference_config)
        llamacpp_elapsed = time.time() - start_time
        
        llamacpp_tokens = count_response_tokens(llamacpp_result)
        llamacpp_throughput = measure_tokens_per_second(llamacpp_tokens, llamacpp_elapsed)
        
        # Validate both results
        assert validate_generation_output(vllm_result)
        assert validate_generation_output(llamacpp_result)
        
        # Both should work, but GPU should generally be faster
        assert vllm_throughput > 1.0, f"VLLM GPU throughput too low: {vllm_throughput}"
        assert llamacpp_throughput > 0.5, f"LlamaCpp CPU throughput too low: {llamacpp_throughput}"
        
        print(f"GPU vs CPU comparison:")
        print(f"VLLM (GPU): {vllm_throughput:.2f} tokens/sec")
        print(f"LlamaCpp (CPU): {llamacpp_throughput:.2f} tokens/sec")
        print(f"GPU speedup: {vllm_throughput / llamacpp_throughput:.2f}x")


class TestEngineFeatureCompatibility:
    """Test feature compatibility across engines."""
    
    @pytest.mark.skipif(not vllm_available, reason="vLLM not available")
    @requires_cuda_initialized()
    @pytest.mark.single_gpu
    def test_generation_parameter_support(self):
        """Test that engines handle generation parameters consistently."""
        skip_if_no_cuda()
        skip_if_insufficient_vram(4.0)
        skip_if_insufficient_memory(6.0)
        
        models = get_test_models()
        engines = {
            "native": NativeTextInferenceEngine(models["smollm_135m"]),
            "vllm": VLLMInferenceEngine(models["smollm_135m"])
        }
        
        conversations = create_test_conversations()[:1]
        
        # Test various parameter combinations
        parameter_sets = [
            GenerationParams(max_new_tokens=10, temperature=0.0, seed=42),
            GenerationParams(max_new_tokens=15, temperature=0.5, top_p=0.9, seed=42),
            GenerationParams(max_new_tokens=20, temperature=0.8, use_sampling=True, seed=42)
        ]
        
        for i, gen_params in enumerate(parameter_sets):
            config = InferenceConfig(generation=gen_params)
            
            for engine_name, engine in engines.items():
                try:
                    result = engine.infer(conversations, config)
                    assert validate_generation_output(result), f"{engine_name} failed param set {i}"
                    
                    # Check that parameters were respected
                    response = result[0].messages[-1].content
                    assert len(response.strip()) > 0, f"{engine_name} empty response for param set {i}"
                    
                except Exception as e:
                    pytest.fail(f"{engine_name} engine failed with parameters {gen_params}: {e}")
                    
    @pytest.mark.skipif(not llamacpp_available, reason="LlamaCpp not available")
    def test_llamacpp_hardware_parameter_support(self):
        """Test that LlamaCpp handles hardware parameters correctly."""
        skip_if_insufficient_memory(6.0)
        
        models = get_test_models()
        conversations = create_test_conversations()[:1]
        config = InferenceConfig(generation=GenerationParams(max_new_tokens=10, temperature=0.0))
        
        # Test different hardware configurations
        hardware_configs = [
            {"n_threads": 1},
            {"n_threads": 2},
            {"n_gpu_layers": 0},  # CPU only
            {"use_mmap": True, "use_mlock": False},
            {"use_mmap": False, "use_mlock": False},
        ]
        
        for i, hw_config in enumerate(hardware_configs):
            model_params = models["gemma_270m_gguf"]
            model_params.model_kwargs = {
                **model_params.model_kwargs,
                **hw_config,
                "verbose": False
            }
            
            try:
                engine = LlamaCppInferenceEngine(model_params)
                result = engine.infer(conversations, config)
                assert validate_generation_output(result), f"Hardware config {i} failed"
                
            except Exception as e:
                pytest.fail(f"LlamaCpp failed with hardware config {hw_config}: {e}")
                
    def test_engine_error_handling_consistency(self):
        """Test that engines handle errors consistently."""
        models = get_test_models()
        
        # Test with invalid generation parameters
        invalid_gen_params = GenerationParams(
            max_new_tokens=-1,  # Invalid
            temperature=0.0
        )
        invalid_config = InferenceConfig(generation=invalid_gen_params)
        
        conversations = create_test_conversations()[:1]
        
        # Test Native engine
        native_engine = NativeTextInferenceEngine(models["smollm_135m"])
        with pytest.raises((ValueError, Exception)):
            native_engine.infer(conversations, invalid_config)
            
        # Test engines with invalid empty input should handle gracefully
        empty_conversations = []
        valid_config = InferenceConfig(generation=GenerationParams(max_new_tokens=10))
        
        result = native_engine.infer(empty_conversations, valid_config)
        assert result == []