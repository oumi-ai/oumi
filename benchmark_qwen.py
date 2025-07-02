#!/usr/bin/env python3
"""
Performance benchmark script for quantized Qwen model.
Measures throughput, latency, and resource usage.
"""

import time
import torch
import psutil
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime

class ModelBenchmark:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"📥 Loading model from: {self.model_path}")
        start_time = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        return load_time
    
    def measure_throughput(self, prompt, num_runs=5, max_tokens=100):
        """Measure token generation throughput."""
        print(f"\n📊 Measuring throughput ({num_runs} runs)...")
        
        times = []
        token_counts = []
        
        for run in range(num_runs):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            output_length = outputs.shape[1]
            new_tokens = output_length - input_length
            
            times.append(generation_time)
            token_counts.append(new_tokens)
            
            tokens_per_second = new_tokens / generation_time
            print(f"  Run {run+1}: {new_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tok/s)")
        
        avg_time = statistics.mean(times)
        avg_tokens = statistics.mean(token_counts)
        avg_throughput = avg_tokens / avg_time
        
        return {
            "runs": num_runs,
            "avg_generation_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "avg_tokens_per_second": avg_throughput,
            "min_tokens_per_second": min(t/time for t, time in zip(token_counts, times)),
            "max_tokens_per_second": max(t/time for t, time in zip(token_counts, times))
        }
    
    def measure_latency(self, prompts, max_tokens=50):
        """Measure first token latency."""
        print(f"\n⚡ Measuring latency...")
        
        latencies = []
        
        for i, prompt in enumerate(prompts):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Measure time to first token
            start_time = time.time()
            with torch.no_grad():
                # Generate just one token to measure first token latency
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            first_token_time = time.time() - start_time
            latencies.append(first_token_time)
            
            print(f"  Prompt {i+1}: {first_token_time:.3f}s to first token")
        
        return {
            "num_prompts": len(prompts),
            "avg_first_token_latency": statistics.mean(latencies),
            "min_first_token_latency": min(latencies),
            "max_first_token_latency": max(latencies)
        }
    
    def measure_memory_usage(self):
        """Measure memory usage during inference."""
        print(f"\n🧠 Measuring memory usage...")
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**3)  # GB
        
        # Run inference
        prompt = "Explain quantum computing in detail with examples and applications."
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        memory_after = process.memory_info().rss / (1024**3)  # GB
        memory_percent = process.memory_percent()
        
        return {
            "memory_before_gb": memory_before,
            "memory_after_gb": memory_after,
            "memory_used_gb": memory_after - memory_before,
            "memory_percent": memory_percent,
            "peak_memory_gb": memory_after
        }
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("🚀 Starting Full Performance Benchmark")
        print("=" * 50)
        
        # Load model
        load_time = self.load_model()
        
        # Benchmark prompts
        throughput_prompt = "Write a comprehensive guide about artificial intelligence, including its history, current applications, and future prospects."
        
        latency_prompts = [
            "Hello!",
            "What is 2+2?",
            "Explain machine learning.",
            "Write code for sorting.",
            "Describe the weather."
        ]
        
        # Run benchmarks
        throughput_results = self.measure_throughput(throughput_prompt)
        latency_results = self.measure_latency(latency_prompts)
        memory_results = self.measure_memory_usage()
        
        # Compile results
        benchmark_results = {
            "model_path": self.model_path,
            "timestamp": datetime.now().isoformat(),
            "load_time": load_time,
            "throughput": throughput_results,
            "latency": latency_results,
            "memory": memory_results
        }
        
        # Display summary
        print(f"\n📋 BENCHMARK SUMMARY")
        print("=" * 30)
        print(f"Model: {self.model_path}")
        print(f"Load time: {load_time:.2f} seconds")
        print(f"Average throughput: {throughput_results['avg_tokens_per_second']:.2f} tokens/sec")
        print(f"Average first token latency: {latency_results['avg_first_token_latency']:.3f} seconds")
        print(f"Memory usage: {memory_results['peak_memory_gb']:.2f} GB")
        
        # Performance rating
        throughput = throughput_results['avg_tokens_per_second']
        latency = latency_results['avg_first_token_latency']
        memory = memory_results['peak_memory_gb']
        
        print(f"\n⭐ PERFORMANCE RATING")
        print(f"Throughput: {'🚀 Excellent' if throughput > 50 else '✅ Good' if throughput > 20 else '⚠️ Slow'} ({throughput:.1f} tok/s)")
        print(f"Latency: {'⚡ Fast' if latency < 0.1 else '✅ Good' if latency < 0.5 else '⏳ Slow'} ({latency:.3f}s)")
        print(f"Memory: {'🟢 Efficient' if memory < 8 else '🟡 Moderate' if memory < 16 else '🔴 High'} ({memory:.1f}GB)")
        
        # Save results
        report_file = f"qwen_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\n📄 Detailed benchmark report saved to: {report_file}")
        return benchmark_results

def main():
    """Run the benchmark."""
    model_path = "qwen-test-awq.pytorch"
    
    benchmark = ModelBenchmark(model_path)
    results = benchmark.run_full_benchmark()
    
    print(f"\n🎉 Benchmark completed successfully!")

if __name__ == "__main__":
    main()