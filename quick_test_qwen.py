#!/usr/bin/env python3
"""
Quick test script for quantized Qwen model.
Simple functionality check with basic examples.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def quick_test():
    """Quick functionality test of the quantized model."""
    print("🚀 Quick Test: Quantized Qwen Model")
    print("=" * 40)
    
    model_path = "qwen-test-awq.pytorch"
    
    print("📥 Loading model...")
    start_time = time.time()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Quick tests
    test_cases = [
        "Hello! Please introduce yourself.",
        "What is 15 + 27?",
        "Write a hello world program in Python:",
        "Explain what machine learning is in one sentence."
    ]
    
    print(f"\n🧪 Running {len(test_cases)} quick tests...")
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"📝 Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the new part (response after prompt)
        new_text = response[len(prompt):].strip()
        
        print(f"💬 Response: {new_text}")
        print(f"⏱️  Time: {generation_time:.2f}s")
        
        # Basic quality check
        if len(new_text) > 10 and any(word in new_text.lower() for word in ['the', 'is', 'and', 'to', 'a']):
            print("✅ Quality: Looks good")
        else:
            print("⚠️  Quality: May need review")
    
    print(f"\n🎉 Quick test completed!")
    print(f"💡 For detailed evaluation, run: python evaluate_quantized_qwen.py")

if __name__ == "__main__":
    quick_test()