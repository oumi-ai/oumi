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

"""Micro-benchmarks for specific dataset pipeline bottlenecks.

This script focuses on profiling specific operations that were identified
as potential performance bottlenecks:

1. Oversampling with deep copy vs lazy index mapping
2. Feature detection sampling overhead
3. DataFrame to HuggingFace conversion overhead
4. Converter auto-detection overhead

Usage:
    python benchmarks/dataset_microbenchmarks.py [--benchmark NAME]
"""

import argparse
import copy
import gc
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import datasets

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class BenchmarkTimer:
    """Simple timer for micro-benchmarks."""

    def __init__(self, name: str, iterations: int = 1):
        self.name = name
        self.iterations = iterations
        self.times: list[float] = []
        self.peak_memories: list[float] = []

    def run(self, func: Callable, *args, **kwargs) -> Any:
        """Run the function and record timing."""
        result = None
        for i in range(self.iterations):
            gc.collect()
            tracemalloc.start()

            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.times.append(elapsed)
            self.peak_memories.append(peak / (1024 * 1024))

            gc.collect()

        return result

    def report(self) -> dict:
        """Generate report of timing results."""
        if not self.times:
            return {}

        avg_time = sum(self.times) / len(self.times)
        min_time = min(self.times)
        max_time = max(self.times)
        avg_mem = sum(self.peak_memories) / len(self.peak_memories)
        max_mem = max(self.peak_memories)

        return {
            "name": self.name,
            "iterations": self.iterations,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "avg_peak_memory_mb": avg_mem,
            "max_peak_memory_mb": max_mem,
        }

    def print_report(self):
        """Print formatted report."""
        r = self.report()
        if not r:
            print(f"{self.name}: No data")
            return

        print(f"\n{self.name}:")
        print(f"  Iterations: {r['iterations']}")
        print(f"  Avg time: {r['avg_time_ms']:.2f} ms")
        print(f"  Min/Max: {r['min_time_ms']:.2f} / {r['max_time_ms']:.2f} ms")
        print(f"  Avg peak memory: {r['avg_peak_memory_mb']:.1f} MB")
        print(f"  Max peak memory: {r['max_peak_memory_mb']:.1f} MB")


# =============================================================================
# Benchmark 1: Oversampling Strategies
# =============================================================================


def benchmark_oversampling():
    """Compare deep copy oversampling vs lazy index mapping."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Oversampling Strategies")
    print("=" * 70)

    # Create a sample dataset
    data = {
        "text": [f"This is example text number {i}" for i in range(1000)],
        "label": [i % 10 for i in range(1000)],
        "metadata": [{"id": i, "category": f"cat_{i % 5}"} for i in range(1000)],
    }
    base_dataset = datasets.Dataset.from_dict(data)

    oversample_factor = 5  # 5x oversampling

    # Method 1: Current implementation (deep copy)
    def oversample_deepcopy(ds: datasets.Dataset, factor: int) -> datasets.Dataset:
        """Current Oumi implementation using deep copy."""
        copies = [copy.deepcopy(ds) for _ in range(factor)]
        return datasets.concatenate_datasets(copies)

    # Method 2: Lazy index mapping (proposed optimization)
    class LazyOversampledDataset:
        """Lazy oversampling using index mapping."""

        def __init__(self, dataset: datasets.Dataset, factor: int):
            self.dataset = dataset
            self.factor = factor
            self._len = len(dataset) * factor

        def __len__(self):
            return self._len

        def __getitem__(self, idx):
            return self.dataset[idx % len(self.dataset)]

        def __iter__(self):
            for i in range(self._len):
                yield self[i]

    def oversample_lazy(ds: datasets.Dataset, factor: int):
        """Proposed lazy oversampling."""
        return LazyOversampledDataset(ds, factor)

    # Method 3: HuggingFace select with repeated indices
    def oversample_hf_select(ds: datasets.Dataset, factor: int) -> datasets.Dataset:
        """HuggingFace select with repeated indices."""
        indices = list(range(len(ds))) * factor
        return ds.select(indices)

    # Run benchmarks
    iterations = 5

    print(f"\nBase dataset size: {len(base_dataset)}")
    print(f"Target size: {len(base_dataset) * oversample_factor}")
    print(f"Oversample factor: {oversample_factor}x")

    # Deep copy benchmark
    timer1 = BenchmarkTimer("Deep Copy (current)", iterations=iterations)
    result1 = timer1.run(oversample_deepcopy, base_dataset, oversample_factor)
    timer1.print_report()
    del result1

    # Lazy benchmark
    timer2 = BenchmarkTimer("Lazy Index Mapping (proposed)", iterations=iterations)
    result2 = timer2.run(oversample_lazy, base_dataset, oversample_factor)
    timer2.print_report()

    # Verify lazy works correctly
    lazy_ds = result2
    assert len(lazy_ds) == len(base_dataset) * oversample_factor
    # Check a few samples
    for i in [0, 100, 1000, 2500, 4999]:
        expected_idx = i % len(base_dataset)
        assert lazy_ds[i]["text"] == base_dataset[expected_idx]["text"]
    print("  Correctness: VERIFIED")
    del result2

    # HF select benchmark
    timer3 = BenchmarkTimer("HF Select (alternative)", iterations=iterations)
    result3 = timer3.run(oversample_hf_select, base_dataset, oversample_factor)
    timer3.print_report()
    del result3

    # Summary
    r1, r2, r3 = timer1.report(), timer2.report(), timer3.report()
    speedup_lazy = r1["avg_time_ms"] / r2["avg_time_ms"] if r2["avg_time_ms"] > 0 else 0
    mem_reduction = (
        (r1["avg_peak_memory_mb"] - r2["avg_peak_memory_mb"])
        / r1["avg_peak_memory_mb"]
        * 100
        if r1["avg_peak_memory_mb"] > 0
        else 0
    )

    print(f"\n  Speedup (lazy vs deep copy): {speedup_lazy:.1f}x")
    print(f"  Memory reduction (lazy vs deep copy): {mem_reduction:.1f}%")


# =============================================================================
# Benchmark 2: Feature Detection Sampling
# =============================================================================


def benchmark_feature_detection():
    """Compare different feature detection sampling strategies."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Feature Detection Sampling Strategies")
    print("=" * 70)

    # Create a dataset simulating tokenized data
    num_examples = 10000
    data = {
        "input_ids": [[i % 100 for _ in range(512)] for i in range(num_examples)],
        "attention_mask": [[1] * 512 for _ in range(num_examples)],
        "labels": [[i % 100 for _ in range(512)] for i in range(num_examples)],
    }
    dataset = datasets.Dataset.from_dict(data)

    def sample_one_eighth(ds):
        """Current implementation: sample 1/8 of dataset."""
        total = len(ds)
        indices = list(range(0, total, max(1, total // 8)))
        samples = [ds[i] for i in indices]
        return samples

    def sample_fixed_100(ds):
        """Proposed: sample fixed 100 examples."""
        total = len(ds)
        step = max(1, total // 100)
        indices = list(range(0, total, step))[:100]
        samples = [ds[i] for i in indices]
        return samples

    def sample_fixed_10(ds):
        """Minimal: sample just 10 examples."""
        total = len(ds)
        step = max(1, total // 10)
        indices = list(range(0, total, step))[:10]
        samples = [ds[i] for i in indices]
        return samples

    iterations = 5

    print(f"\nDataset size: {len(dataset)}")

    timer1 = BenchmarkTimer("1/8 sampling (current)", iterations=iterations)
    result1 = timer1.run(sample_one_eighth, dataset)
    timer1.print_report()
    print(f"  Samples collected: {len(result1)}")

    timer2 = BenchmarkTimer("Fixed 100 (proposed)", iterations=iterations)
    result2 = timer2.run(sample_fixed_100, dataset)
    timer2.print_report()
    print(f"  Samples collected: {len(result2)}")

    timer3 = BenchmarkTimer("Fixed 10 (minimal)", iterations=iterations)
    result3 = timer3.run(sample_fixed_10, dataset)
    timer3.print_report()
    print(f"  Samples collected: {len(result3)}")

    # Summary
    r1, r2, r3 = timer1.report(), timer2.report(), timer3.report()
    speedup = r1["avg_time_ms"] / r2["avg_time_ms"] if r2["avg_time_ms"] > 0 else 0
    print(f"\n  Speedup (fixed 100 vs 1/8): {speedup:.1f}x")


# =============================================================================
# Benchmark 3: DataFrame vs Direct HuggingFace Processing
# =============================================================================


def benchmark_dataframe_conversion():
    """Compare DataFrame intermediate vs direct HuggingFace processing."""
    print("\n" + "=" * 70)
    print("BENCHMARK: DataFrame vs Direct HuggingFace Processing")
    print("=" * 70)

    # Simulate loading from HuggingFace
    num_examples = 5000
    raw_data = {
        "instruction": [f"Do task {i}" for i in range(num_examples)],
        "input": [f"Input data {i}" for i in range(num_examples)],
        "output": [f"Output result {i}" for i in range(num_examples)],
    }
    hf_dataset = datasets.Dataset.from_dict(raw_data)

    def transform_fn(example):
        """Simple transform function."""
        return {
            "text": f"{example['instruction']}\n{example['input']}\n{example['output']}",
            "length": len(example["instruction"])
            + len(example["input"])
            + len(example["output"]),
        }

    # Method 1: Current flow (HF -> pandas -> process -> HF)
    def via_pandas(ds: datasets.Dataset):
        # Convert to pandas
        df = ds.to_pandas()
        # Process
        results = []
        for _, row in df.iterrows():
            results.append(transform_fn(row.to_dict()))
        # Convert back to HF
        return datasets.Dataset.from_list(results)

    # Method 2: Direct HuggingFace map
    def via_hf_map(ds: datasets.Dataset):
        return ds.map(transform_fn, remove_columns=ds.column_names)

    # Method 3: Generator approach
    def via_generator(ds: datasets.Dataset):
        def gen():
            for example in ds:
                yield transform_fn(example)

        return datasets.Dataset.from_generator(gen)

    iterations = 3

    print(f"\nDataset size: {len(hf_dataset)}")

    timer1 = BenchmarkTimer("Via Pandas (current)", iterations=iterations)
    result1 = timer1.run(via_pandas, hf_dataset)
    timer1.print_report()
    del result1

    timer2 = BenchmarkTimer("Via HF map (direct)", iterations=iterations)
    result2 = timer2.run(via_hf_map, hf_dataset)
    timer2.print_report()
    del result2

    timer3 = BenchmarkTimer("Via Generator", iterations=iterations)
    result3 = timer3.run(via_generator, hf_dataset)
    timer3.print_report()
    del result3

    # Summary
    r1, r2 = timer1.report(), timer2.report()
    speedup = r1["avg_time_ms"] / r2["avg_time_ms"] if r2["avg_time_ms"] > 0 else 0
    mem_reduction = (
        (r1["avg_peak_memory_mb"] - r2["avg_peak_memory_mb"])
        / r1["avg_peak_memory_mb"]
        * 100
        if r1["avg_peak_memory_mb"] > 0
        else 0
    )
    print(f"\n  Speedup (HF map vs pandas): {speedup:.1f}x")
    print(f"  Memory reduction: {mem_reduction:.1f}%")


# =============================================================================
# Benchmark 4: Converter Auto-Detection
# =============================================================================


def benchmark_converter_detection():
    """Benchmark converter auto-detection overhead."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Converter Auto-Detection")
    print("=" * 70)

    # Different format examples
    examples = {
        "oumi": {"messages": [{"role": "user", "content": "Hello"}]},
        "alpaca": {"instruction": "Do this", "input": "data", "output": "result"},
        "sharegpt": {"conversations": [{"from": "human", "value": "Hi"}]},
        "langfuse": {"input": "query", "output": "response"},
    }

    # Implement auto_detect logic directly to avoid circular import
    def auto_detect_converter(example: dict) -> str:
        """Auto-detect format based on example structure."""
        # Check for Oumi/OpenAI format
        if "messages" in example:
            messages = example["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                if all(
                    isinstance(m, dict) and "role" in m and "content" in m
                    for m in messages
                ):
                    return "oumi"

        # Check for nested conversations format
        if "conversation" in example:
            conv = example["conversation"]
            if isinstance(conv, dict) and "messages" in conv:
                return "conversations"

        # Check for Alpaca format
        if all(key in example for key in ["instruction", "input", "output"]):
            return "alpaca"

        # Check for ShareGPT format
        if "conversations" in example:
            convs = example["conversations"]
            if isinstance(convs, list) and len(convs) > 0:
                if all("from" in c and "value" in c for c in convs):
                    return "sharegpt"

        # Check for Langfuse format
        if ("input" in example and "output" in example) or (
            "prompt" in example and "completion" in example
        ):
            return "langfuse"

        # Check for OpenTelemetry format
        attrs = example.get("attributes", example)
        if "gen_ai.prompt" in attrs:
            return "opentelemetry"

        # Check for LangChain format
        if "inputs" in example and "outputs" in example:
            return "langchain"

        raise ValueError(
            f"Unable to auto-detect format for keys: {list(example.keys())}"
        )

    iterations = 1000

    print(f"\nIterations per format: {iterations}")

    for format_name, example in examples.items():
        timer = BenchmarkTimer(f"Auto-detect '{format_name}'", iterations=iterations)
        timer.run(auto_detect_converter, example)
        r = timer.report()
        print(f"  {format_name}: {r['avg_time_ms']:.4f} ms avg")

    # Test cached vs uncached lookup
    print("\n  Caching comparison:")

    # Simulate what caching would look like
    detection_cache: dict[str, str] = {}

    def detect_with_cache(example: dict, cache_key: str) -> str:
        if cache_key in detection_cache:
            return detection_cache[cache_key]
        result = auto_detect_converter(example)
        detection_cache[cache_key] = result
        return result

    # First call (cache miss)
    timer_miss = BenchmarkTimer("Cache miss", iterations=1)
    timer_miss.run(detect_with_cache, examples["alpaca"], "alpaca_dataset")
    r_miss = timer_miss.report()

    # Subsequent calls (cache hit)
    timer_hit = BenchmarkTimer("Cache hit", iterations=iterations)
    timer_hit.run(detect_with_cache, examples["alpaca"], "alpaca_dataset")
    r_hit = timer_hit.report()

    print(f"    Cache miss: {r_miss['avg_time_ms']:.4f} ms")
    print(f"    Cache hit: {r_hit['avg_time_ms']:.6f} ms")
    speedup = (
        r_miss["avg_time_ms"] / r_hit["avg_time_ms"] if r_hit["avg_time_ms"] > 0 else 0
    )
    print(f"    Speedup with caching: {speedup:.0f}x")


# =============================================================================
# Benchmark 5: Dataset Mixture Operations
# =============================================================================


def benchmark_mixture_operations():
    """Benchmark dataset mixture and interleaving operations."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Dataset Mixture Operations")
    print("=" * 70)

    # Create multiple datasets of different sizes
    sizes = [1000, 2000, 500, 1500]
    dataset_list = []
    for i, size in enumerate(sizes):
        data = {
            "text": [f"Dataset {i} example {j}" for j in range(size)],
            "source": [i] * size,
        }
        dataset_list.append(datasets.Dataset.from_dict(data))

    iterations = 5

    print(f"\nDataset sizes: {sizes}")
    print(f"Total examples: {sum(sizes)}")

    # Method 1: Concatenate
    timer1 = BenchmarkTimer("Concatenate", iterations=iterations)
    result1 = timer1.run(datasets.concatenate_datasets, dataset_list)
    timer1.print_report()
    print(f"  Result size: {len(result1)}")
    del result1

    # Method 2: Interleave (equal probability)
    timer2 = BenchmarkTimer("Interleave (equal)", iterations=iterations)
    result2 = timer2.run(
        datasets.interleave_datasets,
        dataset_list,
        stopping_strategy="first_exhausted",
    )
    timer2.print_report()
    # Force materialization
    _ = list(result2)
    del result2

    # Method 3: Interleave with proportions
    proportions = [0.4, 0.3, 0.1, 0.2]
    timer3 = BenchmarkTimer("Interleave (weighted)", iterations=iterations)
    result3 = timer3.run(
        datasets.interleave_datasets,
        dataset_list,
        probabilities=proportions,
        stopping_strategy="first_exhausted",
    )
    timer3.print_report()
    del result3


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Dataset pipeline micro-benchmarks")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[
            "all",
            "oversampling",
            "feature_detection",
            "dataframe",
            "converter",
            "mixture",
        ],
        default="all",
        help="Which benchmark to run",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("OUMI DATASET PIPELINE MICRO-BENCHMARKS")
    print("=" * 70)

    benchmarks = {
        "oversampling": benchmark_oversampling,
        "feature_detection": benchmark_feature_detection,
        "dataframe": benchmark_dataframe_conversion,
        "converter": benchmark_converter_detection,
        "mixture": benchmark_mixture_operations,
    }

    if args.benchmark == "all":
        for name, func in benchmarks.items():
            try:
                func()
            except Exception as e:
                print(f"\n[ERROR] {name} benchmark failed: {e}")
                import traceback

                traceback.print_exc()
    else:
        benchmarks[args.benchmark]()

    print("\n" + "=" * 70)
    print("MICRO-BENCHMARKS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
