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

"""Deep profiling of dataset pipeline to find exact bottlenecks.

This script profiles:
1. CPU time per function (cProfile)
2. Memory allocations (tracemalloc)
3. Line-by-line timing for hot functions
4. Memory snapshots at key points

Usage:
    python benchmarks/profile_pipeline.py [--dataset NAME] [--samples N]
"""

import argparse
import cProfile
import gc
import io
import os
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_profile_cache"


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def get_top_memory_lines(snapshot, limit=20):
    """Get top memory-consuming lines from tracemalloc snapshot."""
    top_stats = snapshot.statistics("lineno")
    result = []
    for stat in top_stats[:limit]:
        result.append({
            "file": stat.traceback.format()[0] if stat.traceback else "unknown",
            "size": stat.size,
            "size_str": format_bytes(stat.size),
            "count": stat.count,
        })
    return result


def profile_with_cprofile(func, *args, **kwargs):
    """Profile a function with cProfile and return stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    return result, stream.getvalue()


def profile_memory_detailed(func, *args, **kwargs):
    """Profile memory with detailed line-by-line tracking."""
    gc.collect()
    tracemalloc.start()

    # Take snapshot before
    snapshot_before = tracemalloc.take_snapshot()

    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start_time

    # Take snapshot after
    snapshot_after = tracemalloc.take_snapshot()

    # Get memory diff
    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, {
        "elapsed_sec": elapsed,
        "current_memory": current,
        "peak_memory": peak,
        "top_allocations": top_stats[:20],
    }


def print_memory_stats(stats, title):
    """Print formatted memory statistics."""
    print(f"\n{'='*70}")
    print(f"MEMORY PROFILE: {title}")
    print(f"{'='*70}")
    print(f"Elapsed: {stats['elapsed_sec']:.2f}s")
    print(f"Current memory: {format_bytes(stats['current_memory'])}")
    print(f"Peak memory: {format_bytes(stats['peak_memory'])}")
    print(f"\nTop Memory Allocations (diff):")
    print("-" * 70)
    for i, stat in enumerate(stats["top_allocations"][:15], 1):
        # Format the traceback
        frame = stat.traceback[0] if stat.traceback else None
        if frame:
            location = f"{frame.filename}:{frame.lineno}"
            # Shorten path
            if "/oumi/" in location:
                location = "..." + location.split("/oumi/")[-1]
            elif "/site-packages/" in location:
                location = "..." + location.split("/site-packages/")[-1]
        else:
            location = "unknown"

        size_diff = stat.size_diff if hasattr(stat, 'size_diff') else stat.size
        print(f"{i:2}. {format_bytes(size_diff):>10} | {location[:55]}")


def profile_dataset_loading(dataset_name: str, split: str, num_samples: int):
    """Profile dataset loading phase."""
    from oumi.core.registry import REGISTRY

    print(f"\n{'#'*70}")
    print(f"PROFILING: Dataset Loading")
    print(f"Dataset: {dataset_name}, Split: {split}")
    print(f"{'#'*70}")

    dataset_class = REGISTRY.get_dataset(dataset_name)
    if dataset_class is None:
        print(f"Dataset {dataset_name} not found in registry")
        return None

    def load_dataset():
        return dataset_class(
            dataset_name=dataset_name,
            split=f"{split}[:{num_samples}]" if num_samples else split,
        )

    # Profile with cProfile
    print("\n--- CPU Profile (cProfile) ---")
    ds, cpu_stats = profile_with_cprofile(load_dataset)
    print(cpu_stats)

    # Profile memory
    gc.collect()
    ds2, mem_stats = profile_memory_detailed(load_dataset)
    print_memory_stats(mem_stats, "Dataset Loading")

    return ds


def profile_to_hf_conversion(dataset, use_native: bool = True):
    """Profile to_hf conversion phase."""
    print(f"\n{'#'*70}")
    print(f"PROFILING: to_hf Conversion (native={use_native})")
    print(f"Dataset size: {len(dataset)} examples")
    print(f"{'#'*70}")

    def convert_to_hf():
        return dataset.to_hf(use_native_map=use_native)

    # Profile with cProfile
    print("\n--- CPU Profile (cProfile) ---")
    hf_ds, cpu_stats = profile_with_cprofile(convert_to_hf)
    print(cpu_stats)

    # Profile memory
    gc.collect()
    hf_ds2, mem_stats = profile_memory_detailed(convert_to_hf)
    print_memory_stats(mem_stats, f"to_hf (native={use_native})")

    return hf_ds


def profile_transform_batch(dataset, batch_size: int = 1000):
    """Profile the _transform_batch method specifically."""
    print(f"\n{'#'*70}")
    print(f"PROFILING: _transform_batch")
    print(f"Batch size: {batch_size}")
    print(f"{'#'*70}")

    # Get a batch of raw data
    dataset._ensure_loaded()
    if dataset._raw_hf_data is not None:
        batch = dataset._raw_hf_data[:batch_size]
    else:
        print("No raw HF data available")
        return

    def transform_batch():
        return dataset._transform_batch(batch)

    # Profile with cProfile
    print("\n--- CPU Profile (cProfile) ---")
    result, cpu_stats = profile_with_cprofile(transform_batch)
    print(cpu_stats)

    # Profile memory
    gc.collect()
    result2, mem_stats = profile_memory_detailed(transform_batch)
    print_memory_stats(mem_stats, "_transform_batch")

    # Detailed timing breakdown
    print("\n--- Detailed Timing Breakdown ---")

    # Time individual steps
    gc.collect()

    # Step 1: Extract examples
    start = time.perf_counter()
    batch_size_actual = len(next(iter(batch.values())))
    examples = [{k: v[i] for k, v in batch.items()} for i in range(batch_size_actual)]
    step1_time = time.perf_counter() - start

    # Step 2: Transform to conversations
    start = time.perf_counter()
    conversations = [dataset.transform_conversation(ex) for ex in examples]
    step2_time = time.perf_counter() - start

    # Step 3: Apply chat template (if tokenizer available)
    if dataset._tokenizer:
        start = time.perf_counter()
        texts = []
        for conv in conversations:
            text = dataset._tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        step3_time = time.perf_counter() - start

        # Step 4: Tokenize
        start = time.perf_counter()
        tokenized = dataset._tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=dataset._tokenizer.model_max_length,
            return_tensors=None,
        )
        step4_time = time.perf_counter() - start

        print(f"  Extract examples:     {step1_time*1000:8.2f} ms ({step1_time/sum([step1_time,step2_time,step3_time,step4_time])*100:5.1f}%)")
        print(f"  Transform to conv:    {step2_time*1000:8.2f} ms ({step2_time/sum([step1_time,step2_time,step3_time,step4_time])*100:5.1f}%)")
        print(f"  Apply chat template:  {step3_time*1000:8.2f} ms ({step3_time/sum([step1_time,step2_time,step3_time,step4_time])*100:5.1f}%)")
        print(f"  Tokenize:             {step4_time*1000:8.2f} ms ({step4_time/sum([step1_time,step2_time,step3_time,step4_time])*100:5.1f}%)")
        print(f"  TOTAL:                {(step1_time+step2_time+step3_time+step4_time)*1000:8.2f} ms")
    else:
        print(f"  Extract examples:     {step1_time*1000:8.2f} ms")
        print(f"  Transform to conv:    {step2_time*1000:8.2f} ms")


def profile_tokenization_strategies(dataset, num_samples: int = 100):
    """Compare different tokenization strategies."""
    print(f"\n{'#'*70}")
    print(f"PROFILING: Tokenization Strategies")
    print(f"Samples: {num_samples}")
    print(f"{'#'*70}")

    if dataset._tokenizer is None:
        print("No tokenizer available")
        return

    dataset._ensure_loaded()
    if dataset._raw_hf_data is None:
        print("No raw HF data available")
        return

    # Get sample data
    batch = dataset._raw_hf_data[:num_samples]
    examples = [{k: v[i] for k, v in batch.items()} for i in range(num_samples)]
    conversations = [dataset.transform_conversation(ex) for ex in examples]

    # Strategy 1: Serial tokenization (one at a time)
    gc.collect()
    start = time.perf_counter()
    serial_results = []
    for conv in conversations:
        text = dataset._tokenizer.apply_chat_template(conv, tokenize=False)
        tokens = dataset._tokenizer(text, padding=False, truncation=True)
        serial_results.append(tokens)
    serial_time = time.perf_counter() - start

    # Strategy 2: Batched tokenization
    gc.collect()
    start = time.perf_counter()
    texts = [dataset._tokenizer.apply_chat_template(conv, tokenize=False) for conv in conversations]
    batch_tokens = dataset._tokenizer(texts, padding=False, truncation=True)
    batch_time = time.perf_counter() - start

    # Strategy 3: Batched chat template (if supported)
    gc.collect()
    start = time.perf_counter()
    try:
        texts_batched = dataset._tokenizer.apply_chat_template(
            conversations, tokenize=False
        )
        if isinstance(texts_batched, str):
            texts_batched = [texts_batched]  # Single result, not batched
            batched_template_time = None
        else:
            batch_tokens_v2 = dataset._tokenizer(texts_batched, padding=False, truncation=True)
            batched_template_time = time.perf_counter() - start
    except Exception as e:
        batched_template_time = None
        print(f"  Batched chat template not supported: {e}")

    print(f"\n  Strategy Comparison ({num_samples} samples):")
    print(f"  {'Strategy':<30} | {'Time':>10} | {'Speedup':>10}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'Serial (one at a time)':<30} | {serial_time*1000:>8.1f}ms | {'1.0x':>10}")
    print(f"  {'Batched tokenization':<30} | {batch_time*1000:>8.1f}ms | {serial_time/batch_time:>9.1f}x")
    if batched_template_time:
        print(f"  {'Batched template + tokenize':<30} | {batched_template_time*1000:>8.1f}ms | {serial_time/batched_template_time:>9.1f}x")


def profile_oversampling(num_samples: int = 1000, oversample_factor: int = 5):
    """Profile different oversampling strategies."""
    import copy
    import datasets

    print(f"\n{'#'*70}")
    print(f"PROFILING: Oversampling Strategies")
    print(f"Base samples: {num_samples}, Factor: {oversample_factor}x")
    print(f"{'#'*70}")

    # Create sample dataset
    data = {"text": [f"Sample text {i}" for i in range(num_samples)]}
    ds = datasets.Dataset.from_dict(data)

    # Strategy 1: Deep copy (current implementation)
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    copies = [copy.deepcopy(ds) for _ in range(oversample_factor)]
    result1 = datasets.concatenate_datasets(copies)
    deepcopy_time = time.perf_counter() - start
    _, deepcopy_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Strategy 2: Index-based selection
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    indices = list(range(len(ds))) * oversample_factor
    result2 = ds.select(indices)
    select_time = time.perf_counter() - start
    _, select_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Strategy 3: Concatenate same reference (unsafe but fast)
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    result3 = datasets.concatenate_datasets([ds] * oversample_factor)
    concat_time = time.perf_counter() - start
    _, concat_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  {'Strategy':<25} | {'Time':>12} | {'Memory':>12} | {'Speedup':>10}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    print(f"  {'Deep copy (current)':<25} | {deepcopy_time*1000:>10.2f}ms | {format_bytes(deepcopy_mem):>12} | {'1.0x':>10}")
    print(f"  {'Index selection':<25} | {select_time*1000:>10.2f}ms | {format_bytes(select_mem):>12} | {deepcopy_time/select_time:>9.1f}x")
    print(f"  {'Concatenate refs':<25} | {concat_time*1000:>10.2f}ms | {format_bytes(concat_mem):>12} | {deepcopy_time/concat_time:>9.1f}x")


def profile_full_pipeline(dataset_name: str, split: str, num_samples: int):
    """Run full pipeline profiling."""
    from transformers import AutoTokenizer
    from oumi.core.registry import REGISTRY

    print(f"\n{'='*70}")
    print(f"FULL PIPELINE PROFILING")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {num_samples}")
    print(f"{'='*70}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    # Get dataset class
    dataset_class = REGISTRY.get_dataset(dataset_name)
    if dataset_class is None:
        print(f"Dataset {dataset_name} not in registry, using AlpacaDataset")
        from oumi.datasets import AlpacaDataset
        dataset_class = AlpacaDataset
        dataset_name = "yahma/alpaca-cleaned"

    # Profile loading
    print("\n" + "="*70)
    print("PHASE 1: Dataset Loading")
    print("="*70)

    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    dataset = dataset_class(
        dataset_name=dataset_name,
        split=f"{split}[:{num_samples}]" if num_samples else split,
        tokenizer=tokenizer,
    )

    load_time = time.perf_counter() - start
    _, load_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Load time: {load_time:.2f}s")
    print(f"  Peak memory: {format_bytes(load_mem)}")
    print(f"  Dataset size: {len(dataset)} examples")

    # Profile transform_batch
    print("\n" + "="*70)
    print("PHASE 2: Transform Batch Analysis")
    print("="*70)
    profile_transform_batch(dataset, batch_size=min(1000, len(dataset)))

    # Profile tokenization strategies
    print("\n" + "="*70)
    print("PHASE 3: Tokenization Strategies")
    print("="*70)
    profile_tokenization_strategies(dataset, num_samples=min(100, len(dataset)))

    # Profile to_hf native
    print("\n" + "="*70)
    print("PHASE 4: to_hf Conversion (Native)")
    print("="*70)

    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    hf_ds = dataset.to_hf(use_native_map=True)

    native_time = time.perf_counter() - start
    _, native_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Conversion time: {native_time:.2f}s")
    print(f"  Peak memory: {format_bytes(native_mem)}")
    print(f"  Throughput: {len(dataset)/native_time:.0f} examples/sec")

    # Profile to_hf legacy (if small enough)
    if len(dataset) <= 5000:
        print("\n" + "="*70)
        print("PHASE 5: to_hf Conversion (Legacy Generator)")
        print("="*70)

        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()

        hf_ds_legacy = dataset.to_hf(use_native_map=False)

        legacy_time = time.perf_counter() - start
        _, legacy_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  Conversion time: {legacy_time:.2f}s")
        print(f"  Peak memory: {format_bytes(legacy_mem)}")
        print(f"  Throughput: {len(dataset)/legacy_time:.0f} examples/sec")
        print(f"\n  Native vs Legacy speedup: {legacy_time/native_time:.1f}x")

    # Profile oversampling
    print("\n" + "="*70)
    print("PHASE 6: Oversampling Analysis")
    print("="*70)
    profile_oversampling(num_samples=1000, oversample_factor=5)

    # Summary
    print("\n" + "="*70)
    print("PROFILING SUMMARY")
    print("="*70)
    print(f"  Dataset: {dataset_name}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  to_hf time (native): {native_time:.2f}s")
    print(f"  Peak memory (load): {format_bytes(load_mem)}")
    print(f"  Peak memory (to_hf): {format_bytes(native_mem)}")
    print(f"  Overall throughput: {len(dataset)/(load_time+native_time):.0f} examples/sec")


def main():
    parser = argparse.ArgumentParser(description="Profile dataset pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="yahma/alpaca-cleaned",
        help="Dataset name to profile",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of samples to profile",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "load", "transform", "to_hf", "oversample"],
        default="all",
        help="Which phase to profile",
    )

    args = parser.parse_args()

    if args.phase == "all":
        profile_full_pipeline(args.dataset, args.split, args.samples)
    elif args.phase == "oversample":
        profile_oversampling()
    else:
        print(f"Profiling phase: {args.phase}")
        # Individual phase profiling
        if args.phase == "load":
            profile_dataset_loading(args.dataset, args.split, args.samples)
        # Add other phases as needed


if __name__ == "__main__":
    main()
