#!/usr/bin/env python3
"""
AWQ Quantization Python Testing Examples

This script demonstrates how to test AWQ quantization functionality
using Python API calls rather than CLI commands.
"""

import tempfile
from pathlib import Path

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize import quantize


def test_basic_awq_quantization():
    """Test basic AWQ quantization with minimal configuration."""
    print("🧪 Testing basic AWQ quantization...")
    
    config = QuantizationConfig(
        model=ModelParams(model_name="microsoft/DialoGPT-small"),
        method="awq_q4_0",
        output_path="test_outputs/python_basic_awq.gguf",
        output_format="gguf",
        calibration_samples=128,
        verbose=True
    )
    
    try:
        result = quantize(config)
        print(f"✅ Basic AWQ test completed: {result}")
        return True
    except Exception as e:
        print(f"❌ Basic AWQ test failed: {e}")
        return False


def test_awq_methods_comparison():
    """Test different AWQ quantization methods."""
    print("🧪 Testing AWQ methods comparison...")
    
    methods = ["awq_q4_0", "awq_q4_1", "awq_q8_0", "awq_f16"]
    results = {}
    
    for method in methods:
        print(f"  Testing {method}...")
        
        config = QuantizationConfig(
            model=ModelParams(model_name="microsoft/DialoGPT-small"),
            method=method,
            output_path=f"test_outputs/python_{method}_test.gguf",
            output_format="gguf",
            calibration_samples=64,  # Fast testing
            cleanup_temp=True,
            verbose=False
        )
        
        try:
            result = quantize(config)
            results[method] = {"status": "success", "result": result}
            print(f"    ✅ {method} completed")
        except Exception as e:
            results[method] = {"status": "failed", "error": str(e)}
            print(f"    ❌ {method} failed: {e}")
    
    print(f"📊 AWQ methods comparison results: {results}")
    return results


def test_awq_configuration_options():
    """Test various AWQ configuration options."""
    print("🧪 Testing AWQ configuration options...")
    
    # Test with different AWQ settings
    test_configs = [
        {
            "name": "small_group_size",
            "awq_group_size": 64,
            "calibration_samples": 256
        },
        {
            "name": "large_group_size", 
            "awq_group_size": 256,
            "calibration_samples": 128
        },
        {
            "name": "no_zero_point",
            "awq_zero_point": False,
            "calibration_samples": 128
        },
        {
            "name": "gemv_kernel",
            "awq_version": "GEMV",
            "calibration_samples": 128
        }
    ]
    
    results = {}
    
    for test_config in test_configs:
        name = test_config.pop("name")
        print(f"  Testing {name}...")
        
        config = QuantizationConfig(
            model=ModelParams(model_name="microsoft/DialoGPT-small"),
            method="awq_q4_0",
            output_path=f"test_outputs/python_config_{name}.gguf",
            output_format="gguf",
            verbose=False,
            **test_config
        )
        
        try:
            result = quantize(config)
            results[name] = {"status": "success", "result": result}
            print(f"    ✅ {name} completed")
        except Exception as e:
            results[name] = {"status": "failed", "error": str(e)}
            print(f"    ❌ {name} failed: {e}")
    
    print(f"📊 Configuration options results: {results}")
    return results


def test_awq_error_handling():
    """Test AWQ error handling and validation."""
    print("🧪 Testing AWQ error handling...")
    
    error_tests = [
        {
            "name": "invalid_model",
            "config": QuantizationConfig(
                model=ModelParams(model_name="nonexistent/model"),
                method="awq_q4_0",
                output_path="test_outputs/error_test.gguf"
            ),
            "expected_error": "Model not found"
        },
        {
            "name": "invalid_method",
            "config": QuantizationConfig(
                model=ModelParams(model_name="microsoft/DialoGPT-small"),
                method="awq_invalid",
                output_path="test_outputs/error_test.gguf"
            ),
            "expected_error": "Unsupported quantization method"
        },
        {
            "name": "invalid_output_format",
            "config": QuantizationConfig(
                model=ModelParams(model_name="microsoft/DialoGPT-small"),
                method="awq_q4_0",
                output_path="test_outputs/error_test.gguf",
                output_format="invalid_format"
            ),
            "expected_error": "Unsupported output format"
        }
    ]
    
    results = {}
    
    for test in error_tests:
        name = test["name"]
        config = test["config"]
        expected_error = test["expected_error"]
        
        print(f"  Testing {name}...")
        
        try:
            result = quantize(config)
            results[name] = {"status": "unexpected_success", "result": result}
            print(f"    ⚠️ {name} unexpectedly succeeded")
        except Exception as e:
            if expected_error.lower() in str(e).lower():
                results[name] = {"status": "expected_error", "error": str(e)}
                print(f"    ✅ {name} failed as expected")
            else:
                results[name] = {"status": "unexpected_error", "error": str(e)}
                print(f"    ❌ {name} failed with unexpected error: {e}")
    
    print(f"📊 Error handling results: {results}")
    return results


def test_awq_with_local_model():
    """Test AWQ quantization with a local model directory."""
    print("🧪 Testing AWQ with local model...")
    
    # This test assumes you have a local model directory
    # Modify the path to match your setup
    local_model_path = "./test_model"  # Replace with actual local model path
    
    if not Path(local_model_path).exists():
        print(f"  ⏭️ Skipping local model test - {local_model_path} not found")
        return {"status": "skipped", "reason": "No local model found"}
    
    config = QuantizationConfig(
        model=ModelParams(model_name=local_model_path),
        method="awq_q4_0",
        output_path="test_outputs/python_local_model.gguf",
        output_format="gguf",
        calibration_samples=128,
        verbose=True
    )
    
    try:
        result = quantize(config)
        print(f"✅ Local model test completed: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"❌ Local model test failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all AWQ testing examples."""
    print("🚀 AWQ Quantization Python Testing Suite")
    print("=========================================")
    
    # Create output directory
    Path("test_outputs").mkdir(exist_ok=True)
    
    # Run all tests
    tests = [
        test_basic_awq_quantization,
        test_awq_methods_comparison,
        test_awq_configuration_options,
        test_awq_error_handling,
        test_awq_with_local_model
    ]
    
    all_results = {}
    
    for test_func in tests:
        print(f"\n" + "="*50)
        try:
            result = test_func()
            all_results[test_func.__name__] = result
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            all_results[test_func.__name__] = {"status": "crashed", "error": str(e)}
    
    print(f"\n" + "="*50)
    print("📋 Final Test Summary:")
    print("=====================")
    
    for test_name, result in all_results.items():
        status = result.get("status", "unknown") if isinstance(result, dict) else "completed"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 All tests completed. Check 'test_outputs/' directory for generated files.")


if __name__ == "__main__":
    main()