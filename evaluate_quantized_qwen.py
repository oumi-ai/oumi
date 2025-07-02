#!/usr/bin/env python3
"""
Comprehensive evaluation script for quantized Qwen models.
Tests functionality, quality, and performance of AWQ-quantized models.
"""

import time
import torch
import psutil
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def load_model(model_path, device="auto"):
    """Load the quantized model and tokenizer."""
    print(f"📥 Loading model from: {model_path}")
    start_time = time.time()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        return model, tokenizer, load_time
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

def get_model_size(model_path):
    """Calculate model size on disk."""
    if Path(model_path).is_dir():
        total_size = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file())
    else:
        total_size = Path(model_path).stat().st_size
    
    size_gb = total_size / (1024**3)
    size_mb = total_size / (1024**2)
    
    return {
        "bytes": total_size,
        "mb": size_mb,
        "gb": size_gb,
        "formatted": f"{size_gb:.2f} GB" if size_gb >= 1 else f"{size_mb:.1f} MB"
    }

def measure_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss_gb": memory_info.rss / (1024**3),
        "vms_gb": memory_info.vms / (1024**3),
        "percent": process.memory_percent()
    }

def test_generation(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    """Test text generation and measure performance."""
    print(f"\n🧪 Testing generation with prompt: '{prompt[:50]}...'")
    
    # Measure memory before generation
    memory_before = measure_memory_usage()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate tokens per second
    output_length = outputs.shape[1]
    new_tokens = output_length - input_length
    tokens_per_second = new_tokens / generation_time if generation_time > 0 else 0
    
    # Measure memory after generation
    memory_after = measure_memory_usage()
    
    return {
        "prompt": prompt,
        "output": output_text,
        "input_tokens": input_length,
        "output_tokens": new_tokens,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
        "memory_before_gb": memory_before["rss_gb"],
        "memory_after_gb": memory_after["rss_gb"],
        "memory_peak_gb": max(memory_before["rss_gb"], memory_after["rss_gb"])
    }

def run_evaluation_suite():
    """Run comprehensive evaluation tests."""
    print("🚀 Starting Quantized Qwen Model Evaluation")
    print("=" * 60)
    
    model_path = "qwen-test-awq.pytorch"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please make sure the quantized model is in the current directory.")
        return
    
    # Get model size information
    model_size = get_model_size(model_path)
    print(f"📦 Model size: {model_size['formatted']} ({model_size['bytes']:,} bytes)")
    
    # Load model
    model, tokenizer, load_time = load_model(model_path)
    if model is None:
        return
    
    # Test prompts covering different capabilities
    test_prompts = [
        {
            "name": "Chat",
            "prompt": "Hello! How are you today? Can you tell me about yourself?",
            "max_tokens": 80
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to calculate the factorial of a number:",
            "max_tokens": 120
        },
        {
            "name": "Reasoning",
            "prompt": "If I have 3 apples and buy 2 more, then give 1 to my friend, how many apples do I have? Explain your reasoning.",
            "max_tokens": 100
        },
        {
            "name": "Creative Writing",
            "prompt": "Write a short story about a robot learning to paint:",
            "max_tokens": 150
        },
        {
            "name": "Math Problem",
            "prompt": "Solve this equation step by step: 2x + 5 = 13",
            "max_tokens": 100
        }
    ]
    
    results = []
    total_tokens = 0
    total_time = 0
    
    print(f"\n📋 Running {len(test_prompts)} evaluation tests...")
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        
        result = test_generation(
            model, tokenizer, 
            test['prompt'], 
            max_new_tokens=test['max_tokens']
        )
        
        results.append({
            "test_name": test['name'],
            **result
        })
        
        total_tokens += result['output_tokens']
        total_time += result['generation_time']
        
        print(f"⏱️  Generation time: {result['generation_time']:.2f}s")
        print(f"🚄 Tokens/second: {result['tokens_per_second']:.2f}")
        print(f"🧠 Memory usage: {result['memory_peak_gb']:.2f} GB")
        print(f"💬 Output preview: {result['output'][:100]}...")
    
    # Calculate overall statistics
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    peak_memory = max(r['memory_peak_gb'] for r in results)
    
    # Summary report
    print(f"\n📊 EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Model: Qwen AWQ Quantized")
    print(f"Size: {model_size['formatted']}")
    print(f"Load time: {load_time:.2f} seconds")
    print(f"Total tests: {len(test_prompts)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total generation time: {total_time:.2f} seconds")
    print(f"Average tokens/second: {avg_tokens_per_second:.2f}")
    print(f"Peak memory usage: {peak_memory:.2f} GB")
    
    # Performance ratings
    print(f"\n⭐ PERFORMANCE RATINGS")
    print(f"Speed: {'🚀 Excellent' if avg_tokens_per_second > 50 else '✅ Good' if avg_tokens_per_second > 20 else '⚠️ Slow'}")
    print(f"Memory: {'🟢 Efficient' if peak_memory < 8 else '🟡 Moderate' if peak_memory < 16 else '🔴 High'}")
    print(f"Load time: {'⚡ Fast' if load_time < 10 else '✅ Good' if load_time < 30 else '⏳ Slow'}")
    
    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT")
    for result in results:
        output_length = len(result['output'])
        coherent = len(result['output'].split()) > 5  # Basic coherence check
        
        quality = "✅ Good" if coherent and output_length > 50 else "⚠️ Check needed"
        print(f"{result['test_name']}: {quality}")
    
    # Save detailed results
    report = {
        "model_info": {
            "path": model_path,
            "size": model_size,
            "load_time": load_time
        },
        "performance": {
            "avg_tokens_per_second": avg_tokens_per_second,
            "peak_memory_gb": peak_memory,
            "total_generation_time": total_time
        },
        "test_results": results
    }
    
    report_file = "qwen_awq_evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    print(f"\n🎉 Evaluation completed successfully!")
    
    return results

if __name__ == "__main__":
    run_evaluation_suite()