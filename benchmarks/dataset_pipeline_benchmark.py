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

"""Benchmark script for the dataset loading pipeline.

This script profiles the dataset loading pipeline across diverse datasets,
measuring download speed, load speed, transform time, and peak memory usage.

Usage:
    python benchmarks/dataset_pipeline_benchmark.py [--output results.json] [--quick]

Options:
    --output: Path to save JSON results (default: benchmark_results.json)
    --quick: Run quick mode with smaller sample sizes
    --datasets: Comma-separated list of dataset keys to run (default: all)
    --skip-download: Skip datasets that aren't already cached
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class TimingResult:
    """Result of a timed operation."""

    duration_seconds: float
    peak_memory_mb: float
    current_memory_mb: float
    success: bool
    error: Optional[str] = None


@dataclass
class DatasetBenchmarkResult:
    """Complete benchmark result for a single dataset."""

    dataset_key: str
    dataset_name: str
    dataset_config: dict
    num_examples: int = 0
    num_features: int = 0

    # Timing results for each phase
    load_raw: Optional[TimingResult] = None
    to_hf_map: Optional[TimingResult] = None
    to_hf_iterable: Optional[TimingResult] = None
    iteration_first_100: Optional[TimingResult] = None
    iteration_full: Optional[TimingResult] = None
    sampling: Optional[TimingResult] = None
    oversampling: Optional[TimingResult] = None

    # Additional metrics
    examples_per_second: float = 0.0
    bytes_per_example: float = 0.0
    total_benchmark_time: float = 0.0


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""

    timestamp: str
    python_version: str
    platform: str
    oumi_version: str
    quick_mode: bool
    results: list[DatasetBenchmarkResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class MemoryTracker:
    """Context manager for tracking memory usage."""

    def __init__(self):
        self.peak_memory_mb = 0.0
        self.current_memory_mb = 0.0
        self._tracemalloc_started = False

    def __enter__(self):
        gc.collect()
        tracemalloc.start()
        self._tracemalloc_started = True
        return self

    def __exit__(self, *args):
        if self._tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.current_memory_mb = current / (1024 * 1024)
            self.peak_memory_mb = peak / (1024 * 1024)
        gc.collect()


@contextmanager
def timed_operation(name: str = ""):
    """Context manager for timing operations with memory tracking."""
    gc.collect()
    tracker = MemoryTracker()

    result = TimingResult(
        duration_seconds=0.0,
        peak_memory_mb=0.0,
        current_memory_mb=0.0,
        success=False,
    )

    start_time = time.perf_counter()
    try:
        with tracker:
            yield result
        result.success = True
    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {str(e)}"
        if name:
            print(f"  [ERROR] {name}: {result.error}")
    finally:
        result.duration_seconds = time.perf_counter() - start_time
        result.peak_memory_mb = tracker.peak_memory_mb
        result.current_memory_mb = tracker.current_memory_mb

        if name and result.success:
            print(
                f"  {name}: {result.duration_seconds:.2f}s, "
                f"peak_mem={result.peak_memory_mb:.1f}MB"
            )


def get_dataset_configs(quick_mode: bool = False) -> dict[str, dict]:
    """Get diverse dataset configurations for benchmarking.

    Returns a dict mapping dataset keys to their configurations.
    """
    # Sample sizes for quick vs full mode
    small_sample = 100 if quick_mode else 1000
    medium_sample = 500 if quick_mode else 5000
    large_sample = 1000 if quick_mode else 10000

    return {
        # 1. Small SFT dataset - Alpaca format (instruction tuning)
        "alpaca_small": {
            "dataset_name": "yahma/alpaca-cleaned",
            "split": "train",
            "sample_count": small_sample,
            "description": "Small Alpaca-format SFT dataset",
            "category": "sft",
        },
        # 2. Medium SFT dataset - ShareGPT format (multi-turn)
        "sharegpt_medium": {
            "dataset_name": "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "subset": "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry",
            "split": "train",
            "sample_count": medium_sample,
            "description": "ShareGPT multi-turn conversation dataset",
            "category": "sft",
        },
        # 3. Large SFT dataset - OpenAI messages format
        "openhermes_large": {
            "dataset_name": "teknium/OpenHermes-2.5",
            "split": "train",
            "sample_count": large_sample,
            "description": "Large OpenAI-format instruction dataset",
            "category": "sft",
        },
        # 4. DPO/Preference dataset
        "ultrafeedback_dpo": {
            "dataset_name": "HuggingFaceH4/ultrafeedback_binarized",
            "split": "train_prefs",
            "sample_count": medium_sample,
            "description": "DPO preference dataset with chosen/rejected",
            "category": "dpo",
        },
        # 5. Code dataset - longer sequences
        "code_alpaca": {
            "dataset_name": "sahil2801/CodeAlpaca-20k",
            "split": "train",
            "sample_count": small_sample,
            "description": "Code instruction dataset",
            "category": "sft",
        },
        # 6. Math/reasoning dataset
        "metamath": {
            "dataset_name": "meta-math/MetaMathQA",
            "split": "train",
            "sample_count": medium_sample,
            "description": "Math reasoning dataset",
            "category": "sft",
        },
        # 7. Pretraining-style dataset (raw text)
        "wikitext": {
            "dataset_name": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train",
            "sample_count": medium_sample,
            "description": "Raw text pretraining dataset",
            "category": "pretraining",
        },
        # 8. Multilingual dataset
        "aya_multilingual": {
            "dataset_name": "CohereForAI/aya_dataset",
            "split": "train",
            "sample_count": small_sample,
            "description": "Multilingual instruction dataset",
            "category": "sft",
        },
        # 9. Long-context dataset
        "longalpaca": {
            "dataset_name": "Yukang/LongAlpaca-12k",
            "split": "train",
            "sample_count": small_sample,
            "description": "Long-context instruction dataset",
            "category": "sft",
        },
        # 10. Tool/function calling dataset
        "glaive_function": {
            "dataset_name": "glaiveai/glaive-function-calling-v2",
            "split": "train",
            "sample_count": small_sample,
            "description": "Function calling dataset",
            "category": "sft",
        },
    }


def benchmark_raw_hf_load(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    sample_count: int,
) -> tuple[Any, TimingResult]:
    """Benchmark raw HuggingFace dataset loading (no Oumi wrapper)."""
    import datasets

    result_holder = [None]

    with timed_operation("Raw HF load") as timing:
        ds = datasets.load_dataset(
            dataset_name,
            name=subset,
            split=split,
        )
        # Take sample
        if sample_count and len(ds) > sample_count:
            ds = ds.select(range(sample_count))
        result_holder[0] = ds

    return result_holder[0], timing


def get_default_tokenizer():
    """Get a default tokenizer for benchmarking (cached)."""
    if not hasattr(get_default_tokenizer, "_cached"):
        try:
            from transformers import AutoTokenizer

            # Use a tokenizer with chat template support
            # Try Qwen first (small, fast, has chat template)
            try:
                get_default_tokenizer._cached = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2-0.5B-Instruct", use_fast=True
                )
            except Exception:
                # Fallback to TinyLlama
                get_default_tokenizer._cached = AutoTokenizer.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True
                )
        except Exception:
            get_default_tokenizer._cached = None
    return get_default_tokenizer._cached


def benchmark_oumi_load(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    sample_count: int,
    converter: Optional[str] = None,
) -> tuple[Any, TimingResult]:
    """Benchmark Oumi dataset loading via registry or GenericSftDataset."""
    from oumi.core.registry import REGISTRY

    result_holder = [None]
    tokenizer = get_default_tokenizer()

    with timed_operation("Oumi registry load") as timing:
        # Try registry first
        dataset_class = REGISTRY.get_dataset(dataset_name, subset=subset)

        if dataset_class is not None:
            dataset = dataset_class(
                dataset_name=dataset_name,
                split=split,
                subset=subset,
                tokenizer=tokenizer,
            )
        else:
            # Use GenericSftDataset with auto-detection
            from oumi.datasets.sft.generic_sft import GenericSftDataset

            dataset = GenericSftDataset(
                dataset_name=dataset_name,
                split=split,
                subset=subset,
                converter=converter,
                tokenizer=tokenizer,
            )
        result_holder[0] = dataset

    return result_holder[0], timing


def benchmark_to_hf_conversion(
    dataset: Any, return_iterable: bool = False
) -> tuple[Any, TimingResult]:
    """Benchmark converting Oumi dataset to HuggingFace format."""
    result_holder = [None]
    mode = "iterable" if return_iterable else "map"

    with timed_operation(f"to_hf ({mode})") as timing:
        hf_dataset = dataset.to_hf(return_iterable=return_iterable)
        result_holder[0] = hf_dataset

    return result_holder[0], timing


def benchmark_iteration(
    dataset: Any, num_examples: Optional[int] = None, label: str = "Iteration"
) -> TimingResult:
    """Benchmark iterating through dataset examples."""
    count = 0
    with timed_operation(label) as timing:
        limit = num_examples or len(dataset)
        for i, example in enumerate(dataset):
            count += 1
            if i >= limit - 1:
                break
            # Access the example to ensure it's materialized
            _ = example

    return timing


def benchmark_sampling(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    original_size: int,
    target_size: int,
) -> TimingResult:
    """Benchmark dataset sampling (downsampling)."""
    import copy

    import datasets as hf_datasets

    from oumi.builders.data import _sample_dataset
    from oumi.core.configs import DatasetParams

    with timed_operation(f"Sampling {original_size} -> {target_size}") as timing:
        # Load raw dataset
        ds = hf_datasets.load_dataset(
            dataset_name,
            name=subset,
            split=split,
        )
        if len(ds) > original_size:
            ds = ds.select(range(original_size))

        # Create params for sampling
        params = DatasetParams(
            dataset_name=dataset_name,
            subset=subset,
            split=split,
            sample_count=target_size,
            shuffle=True,
            seed=42,
        )

        # Run sampling
        sampled = _sample_dataset(ds, params, stream=False)
        _ = len(sampled)  # Force evaluation

    return timing


def benchmark_oversampling(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    original_size: int,
    target_size: int,
) -> TimingResult:
    """Benchmark dataset oversampling."""
    import datasets as hf_datasets

    from oumi.builders.data import _sample_dataset
    from oumi.core.configs import DatasetParams

    with timed_operation(f"Oversampling {original_size} -> {target_size}") as timing:
        # Load raw dataset
        ds = hf_datasets.load_dataset(
            dataset_name,
            name=subset,
            split=split,
        )
        # Take a small subset to oversample
        if len(ds) > original_size:
            ds = ds.select(range(original_size))

        # Create params for oversampling (target > original)
        params = DatasetParams(
            dataset_name=dataset_name,
            subset=subset,
            split=split,
            sample_count=target_size,
            shuffle=True,
            seed=42,
        )

        # Run oversampling
        oversampled = _sample_dataset(ds, params, stream=False)
        _ = len(oversampled)  # Force evaluation

    return timing


def benchmark_single_dataset(
    key: str,
    config: dict,
    quick_mode: bool = False,
) -> DatasetBenchmarkResult:
    """Run full benchmark suite on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {key}")
    print(f"  {config.get('description', '')}")
    print(f"  Dataset: {config['dataset_name']}")
    print(f"{'='*60}")

    result = DatasetBenchmarkResult(
        dataset_key=key,
        dataset_name=config["dataset_name"],
        dataset_config=config,
    )

    start_time = time.perf_counter()
    sample_count = config.get("sample_count", 1000)

    try:
        # Phase 1: Raw HuggingFace load (baseline)
        print("\n[Phase 1] Raw HuggingFace Loading")
        raw_ds, load_timing = benchmark_raw_hf_load(
            dataset_name=config["dataset_name"],
            subset=config.get("subset"),
            split=config["split"],
            sample_count=sample_count,
        )
        result.load_raw = load_timing

        if raw_ds is not None:
            result.num_examples = len(raw_ds)
            result.num_features = len(raw_ds.column_names)
            print(f"  Loaded {result.num_examples} examples, {result.num_features} features")

        # Phase 2: Oumi dataset loading (if applicable)
        print("\n[Phase 2] Oumi Dataset Loading")
        oumi_ds = None

        # Skip Oumi loading for pretraining datasets (they use different base class)
        if config.get("category") != "pretraining":
            try:
                oumi_ds, oumi_timing = benchmark_oumi_load(
                    dataset_name=config["dataset_name"],
                    subset=config.get("subset"),
                    split=config["split"],
                    sample_count=sample_count,
                )
                # Store as a custom field
                result.dataset_config["oumi_load_timing"] = asdict(oumi_timing)
            except Exception as e:
                print(f"  [SKIP] Oumi load not supported: {e}")

        # Phase 3: to_hf conversion (map dataset)
        if oumi_ds is not None:
            print("\n[Phase 3] to_hf Conversion (Map)")
            try:
                hf_map_ds, to_hf_timing = benchmark_to_hf_conversion(
                    oumi_ds, return_iterable=False
                )
                result.to_hf_map = to_hf_timing
            except Exception as e:
                print(f"  [ERROR] to_hf (map) failed: {e}")

        # Phase 4: to_hf conversion (iterable)
        if oumi_ds is not None:
            print("\n[Phase 4] to_hf Conversion (Iterable)")
            try:
                hf_iter_ds, to_hf_iter_timing = benchmark_to_hf_conversion(
                    oumi_ds, return_iterable=True
                )
                result.to_hf_iterable = to_hf_iter_timing
            except Exception as e:
                print(f"  [ERROR] to_hf (iterable) failed: {e}")

        # Phase 5: Iteration benchmarks
        if raw_ds is not None:
            print("\n[Phase 5] Iteration Benchmarks")

            # First 100 examples
            result.iteration_first_100 = benchmark_iteration(
                raw_ds, num_examples=100, label="First 100 examples"
            )

            # Full iteration (if not too large)
            if result.num_examples <= 5000 or quick_mode:
                result.iteration_full = benchmark_iteration(
                    raw_ds, label=f"Full iteration ({result.num_examples} examples)"
                )

                if result.iteration_full.success and result.iteration_full.duration_seconds > 0:
                    result.examples_per_second = (
                        result.num_examples / result.iteration_full.duration_seconds
                    )

        # Phase 6: Sampling benchmark
        if raw_ds is not None and result.num_examples > 100:
            print("\n[Phase 6] Sampling Benchmark")
            try:
                result.sampling = benchmark_sampling(
                    dataset_name=config["dataset_name"],
                    subset=config.get("subset"),
                    split=config["split"],
                    original_size=min(result.num_examples, 1000),
                    target_size=100,
                )
            except Exception as e:
                print(f"  [ERROR] Sampling benchmark failed: {e}")

        # Phase 7: Oversampling benchmark
        if raw_ds is not None:
            print("\n[Phase 7] Oversampling Benchmark")
            try:
                oversample_base = 100 if quick_mode else 500
                oversample_target = 500 if quick_mode else 2500
                result.oversampling = benchmark_oversampling(
                    dataset_name=config["dataset_name"],
                    subset=config.get("subset"),
                    split=config["split"],
                    original_size=oversample_base,
                    target_size=oversample_target,
                )
            except Exception as e:
                print(f"  [ERROR] Oversampling benchmark failed: {e}")

    except Exception as e:
        print(f"\n[FATAL] Benchmark failed for {key}: {e}")
        traceback.print_exc()

    result.total_benchmark_time = time.perf_counter() - start_time
    print(f"\n  Total benchmark time: {result.total_benchmark_time:.2f}s")

    # Cleanup
    gc.collect()

    return result


def compute_summary(results: list[DatasetBenchmarkResult]) -> dict:
    """Compute summary statistics across all benchmark results."""
    summary = {
        "total_datasets": len(results),
        "successful_datasets": 0,
        "failed_datasets": 0,
        "avg_load_time_seconds": 0.0,
        "avg_to_hf_time_seconds": 0.0,
        "avg_examples_per_second": 0.0,
        "total_benchmark_time_seconds": 0.0,
        "peak_memory_mb": 0.0,
        "by_category": {},
    }

    load_times = []
    to_hf_times = []
    throughputs = []
    peak_memories = []

    for r in results:
        summary["total_benchmark_time_seconds"] += r.total_benchmark_time

        if r.load_raw and r.load_raw.success:
            summary["successful_datasets"] += 1
            load_times.append(r.load_raw.duration_seconds)
            peak_memories.append(r.load_raw.peak_memory_mb)
        else:
            summary["failed_datasets"] += 1

        if r.to_hf_map and r.to_hf_map.success:
            to_hf_times.append(r.to_hf_map.duration_seconds)
            peak_memories.append(r.to_hf_map.peak_memory_mb)

        if r.examples_per_second > 0:
            throughputs.append(r.examples_per_second)

        # Group by category
        category = r.dataset_config.get("category", "unknown")
        if category not in summary["by_category"]:
            summary["by_category"][category] = {"count": 0, "avg_load_time": 0.0}
        summary["by_category"][category]["count"] += 1

    if load_times:
        summary["avg_load_time_seconds"] = sum(load_times) / len(load_times)
    if to_hf_times:
        summary["avg_to_hf_time_seconds"] = sum(to_hf_times) / len(to_hf_times)
    if throughputs:
        summary["avg_examples_per_second"] = sum(throughputs) / len(throughputs)
    if peak_memories:
        summary["peak_memory_mb"] = max(peak_memories)

    return summary


def print_results_table(results: list[DatasetBenchmarkResult]):
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    # Header
    headers = [
        "Dataset",
        "Examples",
        "Load (s)",
        "to_hf (s)",
        "Iter (s)",
        "Peak Mem (MB)",
        "Ex/s",
    ]
    widths = [20, 10, 10, 10, 10, 12, 10]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    for r in results:
        load_time = f"{r.load_raw.duration_seconds:.2f}" if r.load_raw else "N/A"
        to_hf_time = f"{r.to_hf_map.duration_seconds:.2f}" if r.to_hf_map else "N/A"
        iter_time = (
            f"{r.iteration_full.duration_seconds:.2f}" if r.iteration_full else "N/A"
        )
        peak_mem = f"{r.load_raw.peak_memory_mb:.1f}" if r.load_raw else "N/A"
        throughput = f"{r.examples_per_second:.0f}" if r.examples_per_second > 0 else "N/A"

        row = [
            r.dataset_key[:20],
            str(r.num_examples),
            load_time,
            to_hf_time,
            iter_time,
            peak_mem,
            throughput,
        ]
        row_line = " | ".join(str(v).ljust(w) for v, w in zip(row, widths))
        print(row_line)

    print("=" * 100)


def save_results(suite: BenchmarkSuite, output_path: str):
    """Save benchmark results to JSON file."""
    # Convert to dict, handling dataclasses
    def to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj

    data = to_dict(suite)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Oumi dataset loading pipeline"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode with smaller samples",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of dataset keys to benchmark",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip datasets that require downloading",
    )

    args = parser.parse_args()

    # Initialize benchmark suite
    import platform

    try:
        from oumi import __version__ as oumi_version
    except ImportError:
        oumi_version = "unknown"

    suite = BenchmarkSuite(
        timestamp=datetime.now().isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        oumi_version=oumi_version,
        quick_mode=args.quick,
    )

    # Get dataset configs
    all_configs = get_dataset_configs(quick_mode=args.quick)

    # Filter datasets if specified
    if args.datasets:
        requested = set(args.datasets.split(","))
        configs = {k: v for k, v in all_configs.items() if k in requested}
        if not configs:
            print(f"No matching datasets found. Available: {list(all_configs.keys())}")
            sys.exit(1)
    else:
        configs = all_configs

    print("=" * 60)
    print("OUMI DATASET PIPELINE BENCHMARK")
    print("=" * 60)
    print(f"Timestamp: {suite.timestamp}")
    print(f"Python: {suite.python_version}")
    print(f"Platform: {suite.platform}")
    print(f"Oumi version: {suite.oumi_version}")
    print(f"Quick mode: {suite.quick_mode}")
    print(f"Datasets to benchmark: {list(configs.keys())}")
    print("=" * 60)

    # Run benchmarks
    for key, config in configs.items():
        try:
            result = benchmark_single_dataset(key, config, quick_mode=args.quick)
            suite.results.append(result)
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            break
        except Exception as e:
            print(f"\n[ERROR] Failed to benchmark {key}: {e}")
            traceback.print_exc()

    # Compute summary
    suite.summary = compute_summary(suite.results)

    # Print results table
    print_results_table(suite.results)

    # Print summary
    print("\nSUMMARY:")
    print(f"  Successful: {suite.summary['successful_datasets']}/{suite.summary['total_datasets']}")
    print(f"  Avg load time: {suite.summary['avg_load_time_seconds']:.2f}s")
    print(f"  Avg to_hf time: {suite.summary['avg_to_hf_time_seconds']:.2f}s")
    print(f"  Avg throughput: {suite.summary['avg_examples_per_second']:.0f} examples/s")
    print(f"  Peak memory: {suite.summary['peak_memory_mb']:.1f} MB")
    print(f"  Total time: {suite.summary['total_benchmark_time_seconds']:.1f}s")

    # Save results
    save_results(suite, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
