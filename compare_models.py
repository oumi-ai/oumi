#!/usr/bin/env python3
"""
Compare quantized model with original model.
Tests quality degradation and performance differences.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

def load_model_safe(model_path, model_name):
    """Safely load a model with error handling."""
    print(f"📥 Loading {model_name} from: {model_path}")
    try:
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16  # Use fp16 for fair comparison
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        load_time = time.time() - start_time
        print(f"✅ {model_name} loaded in {load_time:.2f} seconds")
        return model, tokenizer, load_time
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None, None, None

def generate_response(model, tokenizer, prompt, max_tokens=100):
    """Generate response and measure performance."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_text = response[len(prompt):].strip()
    
    # Calculate tokens per second
    output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
    
    return {
        "response": new_text,
        "generation_time": generation_time,
        "output_tokens": output_tokens,
        "tokens_per_second": tokens_per_second
    }

def compare_responses(original_resp, quantized_resp, prompt):
    """Basic quality comparison between responses."""
    orig_len = len(original_resp.split())
    quant_len = len(quantized_resp.split())
    
    # Simple similarity check (word overlap)
    orig_words = set(original_resp.lower().split())
    quant_words = set(quantized_resp.lower().split())
    
    if orig_words and quant_words:
        overlap = len(orig_words.intersection(quant_words))
        similarity = overlap / max(len(orig_words), len(quant_words))
    else:
        similarity = 0.0
    
    return {
        "original_length": orig_len,
        "quantized_length": quant_len,
        "length_ratio": quant_len / orig_len if orig_len > 0 else 0,
        "word_similarity": similarity,
        "quality_score": (similarity + min(1.0, quant_len / max(1, orig_len))) / 2
    }

def run_comparison():
    """Run comparison between original and quantized models."""
    print("🔍 Model Comparison: Original vs AWQ Quantized")
    print("=" * 55)
    
    # Model paths
    quantized_path = "qwen-test-awq.pytorch"
    original_path = "Qwen/Qwen2.5-7B-Instruct"  # Original model
    
    # Check if quantized model exists
    if not Path(quantized_path).exists():
        print(f"❌ Quantized model not found: {quantized_path}")
        print("Please run quantization first.")
        return
    
    # Load models
    print("Loading models...")
    quantized_model, quantized_tokenizer, quant_load_time = load_model_safe(quantized_path, "Quantized Model")
    
    if quantized_model is None:
        return
    
    # Try to load original model (may fail due to memory)
    print(f"\nAttempting to load original model...")
    print(f"⚠️  Note: This requires significant memory (~14GB)")
    
    original_model, original_tokenizer, orig_load_time = load_model_safe(original_path, "Original Model")
    
    if original_model is None:
        print(f"⚠️  Cannot load original model (likely memory constraint)")
        print(f"📊 Testing quantized model alone...")
        
        # Test only quantized model
        test_prompts = [
            "Explain quantum physics in simple terms:",
            "Write a Python function to sort a list:",
            "What are the benefits of renewable energy?",
            "Describe the process of machine learning:"
        ]
        
        print(f"\n🧪 Testing quantized model performance...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            result = generate_response(quantized_model, quantized_tokenizer, prompt)
            print(f"Response: {result['response'][:100]}...")
            print(f"Performance: {result['tokens_per_second']:.2f} tokens/sec")
        
        return
    
    # Compare both models
    test_prompts = [
        "Explain the concept of artificial intelligence:",
        "Write a simple sorting algorithm in Python:",
        "What is climate change and its impacts?",
        "Describe how neural networks work:"
    ]
    
    print(f"\n🧪 Running comparison tests...")
    
    comparison_results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: {prompt[:40]}... ---")
        
        # Generate with original model
        print("  🔵 Original model generating...")
        orig_result = generate_response(original_model, original_tokenizer, prompt)
        
        # Generate with quantized model
        print("  🟠 Quantized model generating...")
        quant_result = generate_response(quantized_model, quantized_tokenizer, prompt)
        
        # Compare responses
        comparison = compare_responses(orig_result['response'], quant_result['response'], prompt)
        
        result = {
            "prompt": prompt,
            "original": orig_result,
            "quantized": quant_result,
            "comparison": comparison
        }
        comparison_results.append(result)
        
        # Display comparison
        print(f"  📊 Performance comparison:")
        print(f"    Original: {orig_result['tokens_per_second']:.2f} tok/s")
        print(f"    Quantized: {quant_result['tokens_per_second']:.2f} tok/s")
        print(f"    Speedup: {quant_result['tokens_per_second'] / orig_result['tokens_per_second']:.2f}x")
        
        print(f"  🎯 Quality comparison:")
        print(f"    Length ratio: {comparison['length_ratio']:.2f}")
        print(f"    Similarity: {comparison['word_similarity']:.2f}")
        print(f"    Quality score: {comparison['quality_score']:.2f}")
    
    # Overall summary
    avg_orig_speed = sum(r['original']['tokens_per_second'] for r in comparison_results) / len(comparison_results)
    avg_quant_speed = sum(r['quantized']['tokens_per_second'] for r in comparison_results) / len(comparison_results)
    avg_quality = sum(r['comparison']['quality_score'] for r in comparison_results) / len(comparison_results)
    
    print(f"\n📋 COMPARISON SUMMARY")
    print("=" * 30)
    print(f"Load time - Original: {orig_load_time:.2f}s | Quantized: {quant_load_time:.2f}s")
    print(f"Speed - Original: {avg_orig_speed:.2f} tok/s | Quantized: {avg_quant_speed:.2f} tok/s")
    print(f"Speedup: {avg_quant_speed / avg_orig_speed:.2f}x")
    print(f"Average quality retention: {avg_quality:.2f} ({avg_quality*100:.1f}%)")
    
    # Rating
    quality_rating = "🟢 Excellent" if avg_quality > 0.8 else "🟡 Good" if avg_quality > 0.6 else "🔴 Poor"
    speed_rating = "🚀 Faster" if avg_quant_speed > avg_orig_speed else "⚡ Similar" if abs(avg_quant_speed - avg_orig_speed) < 5 else "🐌 Slower"
    
    print(f"\n⭐ OVERALL RATING")
    print(f"Quality: {quality_rating} ({avg_quality:.2f})")
    print(f"Speed: {speed_rating}")
    
    # Save results
    report = {
        "load_times": {"original": orig_load_time, "quantized": quant_load_time},
        "performance": {"original_avg_speed": avg_orig_speed, "quantized_avg_speed": avg_quant_speed},
        "quality": {"average_score": avg_quality},
        "detailed_results": comparison_results
    }
    
    with open("model_comparison_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed comparison saved to: model_comparison_report.json")

if __name__ == "__main__":
    run_comparison()