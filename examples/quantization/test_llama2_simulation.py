#!/usr/bin/env python3
"""
Direct test of Llama 2 7B AWQ quantization in simulation mode
This bypasses CLI path issues and demonstrates the quantization pipeline directly
"""

import sys
import os
from pathlib import Path

# Add src to path to import oumi modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.quantize import quantize

def test_llama2_awq_simulation():
    """Test Llama 2 7B AWQ quantization in simulation mode."""
    
    print("🦙 Llama 2 7B AWQ Quantization Test")
    print("==================================")
    print()
    
    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    print("📋 Configuration:")
    print("   Model: meta-llama/Llama-2-7b-hf")
    print("   Method: awq_q4_0 (AWQ 4-bit)")
    print("   Output: models/llama2-7b-awq-q4.gguf")
    print("   Group Size: 128")
    print("   Calibration Samples: 512")
    print()
    
    # Create quantization configuration
    config = QuantizationConfig(
        model=ModelParams(
            model_name="meta-llama/Llama-2-7b-hf",
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            model_kwargs={
                "torch_dtype": "auto",
                "device_map": "auto"
            }
        ),
        method="awq_q4_0",
        output_path="models/llama2-7b-awq-q4.gguf",
        output_format="gguf",
        awq_group_size=128,
        awq_zero_point=True,
        awq_version="GEMM",
        calibration_samples=512,
        cleanup_temp=True,
        batch_size=8,
        verbose=True
    )
    
    print("🚀 Starting quantization...")
    print()
    
    try:
        # Run quantization (will automatically use simulation mode if autoawq not available)
        result = quantize(config)
        
        print()
        print("✅ Quantization completed!")
        print()
        print("📊 Results:")
        for key, value in result.items():
            print(f"   {key}: {value}")
        
        # Check output file
        output_file = Path(config.output_path)
        if output_file.exists():
            file_size = output_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            print()
            print("📁 Output File:")
            print(f"   Path: {output_file}")
            print(f"   Size: {file_size_mb:.1f} MB ({file_size:,} bytes)")
            
            # Check if it's a real GGUF file
            with open(output_file, 'rb') as f:
                header = f.read(4)
                if header == b'GGUF':
                    print("   Format: ✅ Valid GGUF header")
                    
                    # Read version
                    version = int.from_bytes(f.read(4), byteorder='little')
                    print(f"   GGUF Version: {version}")
                    
                    if result.get("simulation_mode"):
                        print("   Type: 🎭 Mock file (simulation mode)")
                        print("   Note: Install 'autoawq' for real quantization")
                    else:
                        print("   Type: 🎯 Real quantized model")
                else:
                    print("   Format: ❌ Invalid GGUF header")
            
            print()
            print("🔧 Usage Examples:")
            print()
            print("1. Test with llama-cpp-python:")
            print("   pip install llama-cpp-python")
            print("   python3 -c \"")
            print("   from llama_cpp import Llama")
            print(f"   llm = Llama(model_path='{output_file}')")
            print("   response = llm('Hello, I am Llama 2', max_tokens=50)")
            print("   print(response['choices'][0]['text'])")
            print("   \"")
            print()
            print("2. Test with llama.cpp (if installed):")
            print(f"   ./llama.cpp/main -m {output_file} -p \"Hello, I am Llama 2\"")
            print()
            
            if result.get("simulation_mode"):
                print("⚠️  Note: This is a mock file from simulation mode.")
                print("   For real quantization, install dependencies:")
                print("   pip install autoawq torch transformers accelerate")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        print()
        print("💡 Troubleshooting:")
        print("1. Install dependencies: pip install autoawq torch transformers")
        print("2. For Llama 2 access: Accept license on HuggingFace and login")
        print("3. For memory issues: Reduce batch_size or calibration_samples")
        print()
        return False

def test_multiple_variants():
    """Test multiple AWQ variants for comparison."""
    
    print("🔬 Testing Multiple AWQ Variants")
    print("===============================")
    print()
    
    variants = [
        {
            "name": "AWQ Q4_0 (Compact)",
            "method": "awq_q4_0",
            "output": "models/llama2-awq-q4_0.gguf",
            "description": "Best balance of quality and size"
        },
        {
            "name": "AWQ Q4_1 (Enhanced)",
            "method": "awq_q4_1", 
            "output": "models/llama2-awq-q4_1.gguf",
            "description": "Better 4-bit quality with bias terms"
        },
        {
            "name": "AWQ Q8_0 (Quality)",
            "method": "awq_q8_0",
            "output": "models/llama2-awq-q8_0.gguf",
            "description": "Minimal quality loss, larger size"
        },
        {
            "name": "AWQ F16 (Fast)",
            "method": "awq_f16",
            "output": "models/llama2-awq-f16.gguf", 
            "description": "Format conversion with AWQ optimizations"
        }
    ]
    
    results = {}
    
    for variant in variants:
        print(f"🧪 Testing {variant['name']}")
        print(f"   Method: {variant['method']}")
        print(f"   Description: {variant['description']}")
        
        config = QuantizationConfig(
            model=ModelParams(model_name="meta-llama/Llama-2-7b-hf"),
            method=variant["method"],
            output_path=variant["output"],
            output_format="gguf",
            calibration_samples=256,  # Faster for testing
            batch_size=4,
            verbose=False  # Reduce output for multiple tests
        )
        
        try:
            result = quantize(config)
            results[variant["name"]] = {
                "status": "success",
                "size": result.get("quantized_size", "Unknown"),
                "method": result.get("quantization_method", variant["method"])
            }
            print(f"   ✅ Success - Size: {result.get('quantized_size', 'Unknown')}")
            
        except Exception as e:
            results[variant["name"]] = {
                "status": "failed", 
                "error": str(e)
            }
            print(f"   ❌ Failed: {e}")
        
        print()
    
    print("📊 Summary of Variants:")
    print("======================")
    for name, result in results.items():
        status = result["status"]
        if status == "success":
            size = result["size"]
            method = result["method"]
            print(f"✅ {name}: {size} ({method})")
        else:
            print(f"❌ {name}: {result['error']}")
    
    return results

if __name__ == "__main__":
    print("🧪 Llama 2 7B AWQ Quantization Examples")
    print("======================================")
    print()
    
    # Test single configuration
    success = test_llama2_awq_simulation()
    
    if success:
        print()
        print("━" * 50)
        print()
        
        # Test multiple variants
        test_multiple_variants()
    
    print()
    print("🎯 Example completed!")
    print("Check the 'models/' directory for output files.")