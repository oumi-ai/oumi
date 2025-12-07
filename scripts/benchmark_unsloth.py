#!/usr/bin/env python
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

r"""Benchmark script to compare training speed and VRAM with/without Unsloth.

This script runs training benchmarks comparing:
- Standard HuggingFace model loading vs Unsloth optimized loading
- Full fine-tuning (FFT) and LoRA/Q-LoRA (PEFT) configurations

Uses the existing Unsloth config files from configs/examples/unsloth/ as base
configurations, applying only minimal benchmark-specific overrides.

Requirements:
    - NVIDIA GPU with sufficient VRAM (16GB+ recommended)
    - Install unsloth: `pip install unsloth`
    - Access to Llama 3.2 model: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

Usage:
    python scripts/benchmarks/benchmark_unsloth.py [--max-steps N]

Example:
    python scripts/benchmarks/benchmark_unsloth.py --max-steps 50
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


def get_available_accelerator() -> Optional[str]:
    """Detect available GPU/accelerator backend.

    Returns:
        Backend name ("cuda", "mps", "xpu") or None if no accelerator available.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return None


def get_accelerator_name(backend: str) -> str:
    """Get the name of the accelerator device."""
    if backend == "cuda":
        return torch.cuda.get_device_name(0)
    elif backend == "mps":
        return "Apple Silicon (MPS)"
    elif backend == "xpu":
        return torch.xpu.get_device_name(0)
    return "Unknown"


CONFIG_PATHS = {
    "fft": "configs/examples/unsloth/llama3_2_1b_unsloth_full.yaml",
    "lora": "configs/examples/unsloth/llama3_2_1b_unsloth_lora.yaml",
    "qlora": "configs/examples/unsloth/llama3_2_3b_unsloth_qlora.yaml",
}


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    mode: str  # "standard" or "unsloth"
    training_type: str  # "fft" or "lora" or "qlora"
    total_time_seconds: float
    peak_vram_gb: float
    samples_per_second: Optional[float] = None
    steps_completed: int = 0
    error: Optional[str] = None


def get_gpu_memory_gb(backend: Optional[str] = None) -> float:
    """Get current GPU memory usage in GB.

    Args:
        backend: Accelerator backend. If None, auto-detects.

    Returns:
        Peak memory allocated in GB, or 0.0 if not available.
    """
    if backend is None:
        backend = get_available_accelerator()

    if backend == "cuda":
        return torch.cuda.max_memory_allocated() / (1024**3)
    elif backend == "mps":
        # MPS doesn't have detailed memory tracking, return 0
        # Could use torch.mps.current_allocated_memory() but it's less accurate
        if hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / (1024**3)
        return 0.0
    elif backend == "xpu":
        if hasattr(torch.xpu, "max_memory_allocated"):
            return torch.xpu.max_memory_allocated() / (1024**3)
        return 0.0
    return 0.0


def reset_gpu_memory(backend: Optional[str] = None):
    """Reset GPU memory stats and run garbage collection.

    Args:
        backend: Accelerator backend. If None, auto-detects.
    """
    gc.collect()

    if backend is None:
        backend = get_available_accelerator()

    if backend == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif backend == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    elif backend == "xpu":
        if hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
        if hasattr(torch.xpu, "reset_peak_memory_stats"):
            torch.xpu.reset_peak_memory_stats()


def get_repo_root() -> Path:
    """Get the repository root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find repository root")


def run_benchmark(
    training_type: str,  # "fft", "lora", "qlora"
    max_steps: int,
    enable_unsloth: bool,
    output_dir: str,
) -> BenchmarkResult:
    """Run a single benchmark using existing config files.

    Loads the appropriate config from configs/examples/unsloth/ and applies
    minimal benchmark-specific overrides.

    Args:
        training_type: One of "fft", "lora", "qlora"
        max_steps: Number of training steps to run
        enable_unsloth: Whether to use Unsloth optimization
        output_dir: Directory for benchmark outputs

    Returns:
        BenchmarkResult with timing and VRAM metrics
    """
    from oumi.core.configs import TrainingConfig
    from oumi.train import train

    reset_gpu_memory()

    mode = "unsloth" if enable_unsloth else "standard"
    name = f"{mode}_{training_type}"

    # Load config from existing yaml file
    repo_root = get_repo_root()
    config_path = repo_root / CONFIG_PATHS[training_type]

    if not config_path.exists():
        return BenchmarkResult(
            name=name,
            mode=mode,
            training_type=training_type,
            total_time_seconds=0,
            peak_vram_gb=0,
            error=f"Config file not found: {config_path}",
        )

    config = TrainingConfig.from_yaml(str(config_path))

    # Apply benchmark-specific overrides
    config.training.max_steps = max_steps
    config.training.output_dir = output_dir
    config.training.save_steps = 0  # Don't save checkpoints during benchmark
    config.training.save_final_model = False
    config.training.enable_wandb = False
    config.training.include_performance_metrics = True

    # Toggle Unsloth on/off for comparison
    config.model.enable_unsloth = enable_unsloth
    if not enable_unsloth:
        # Clear unsloth_kwargs when not using Unsloth
        config.model.unsloth_kwargs = {}

    start_time = time.time()
    error_msg = None
    steps_completed = 0

    try:
        train(config)
        steps_completed = max_steps
    except Exception as e:
        error_msg = str(e)

    elapsed = time.time() - start_time
    peak_vram = get_gpu_memory_gb()

    samples_per_second = None
    if steps_completed > 0:
        batch_size = config.training.per_device_train_batch_size
        grad_accum = config.training.gradient_accumulation_steps
        total_samples = steps_completed * batch_size * grad_accum
        samples_per_second = total_samples / elapsed

    return BenchmarkResult(
        name=name,
        mode=mode,
        training_type=training_type,
        total_time_seconds=elapsed,
        peak_vram_gb=peak_vram,
        samples_per_second=samples_per_second,
        steps_completed=steps_completed,
        error=error_msg,
    )


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Group by training type
    by_type: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        if r.training_type not in by_type:
            by_type[r.training_type] = []
        by_type[r.training_type].append(r)

    for training_type, type_results in by_type.items():
        print(f"\n{training_type.upper()} Training:")
        print("-" * 60)
        header = f"{'Mode':<12} {'Time (s)':<12} {'VRAM (GB)':<12} {'Samples/s':<12}"
        print(f"{header} {'Status'}")
        print("-" * 60)

        for r in type_results:
            status = "OK" if r.error is None else f"ERROR: {r.error[:30]}"
            sps = f"{r.samples_per_second:.2f}" if r.samples_per_second else "N/A"
            row = f"{r.mode:<12} {r.total_time_seconds:<12.2f} {r.peak_vram_gb:<12.2f}"
            print(f"{row} {sps:<12} {status}")

        # Calculate speedup/savings if both modes succeeded
        standard = next(
            (r for r in type_results if r.mode == "standard" and r.error is None),
            None,
        )
        unsloth = next(
            (r for r in type_results if r.mode == "unsloth" and r.error is None),
            None,
        )

        if standard and unsloth:
            speedup = standard.total_time_seconds / unsloth.total_time_seconds
            vram_savings = (1 - unsloth.peak_vram_gb / standard.peak_vram_gb) * 100
            print(f"\nUnsloth speedup: {speedup:.2f}x")
            print(f"VRAM savings: {vram_savings:.1f}%")

    print("\n" + "=" * 80)


def main():
    """Run Unsloth vs standard training benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark Unsloth vs standard training"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum training steps per benchmark",
    )
    parser.add_argument(
        "--training-types",
        type=str,
        nargs="+",
        default=["lora", "qlora"],
        choices=["fft", "lora", "qlora"],
        help="Training types to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/benchmark_unsloth",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--skip-standard",
        action="store_true",
        help="Skip standard (non-Unsloth) benchmarks",
    )
    parser.add_argument(
        "--skip-unsloth",
        action="store_true",
        help="Skip Unsloth benchmarks",
    )

    args = parser.parse_args()

    # Check GPU/accelerator availability
    accelerator = get_available_accelerator()
    if accelerator is None:
        print("ERROR: No GPU/accelerator available. This benchmark requires a GPU.")
        print("Supported backends: CUDA (NVIDIA), MPS (Apple Silicon), XPU (Intel)")
        sys.exit(1)

    print(f"Accelerator: {get_accelerator_name(accelerator)} ({accelerator})")
    if accelerator == "cuda":
        print(f"CUDA Version: {torch.version.cuda}")  # type: ignore[attr-defined]
    print(f"Max Steps: {args.max_steps}")
    print(f"Training Types: {args.training_types}")
    print("\nConfig files used:")
    for tt in args.training_types:
        print(f"  {tt}: {CONFIG_PATHS[tt]}")
    print()

    # Check if unsloth is installed
    try:
        import unsloth  # noqa: F401

        unsloth_available = True
        print("Unsloth: installed")
    except ImportError:
        unsloth_available = False
        print("Unsloth: NOT installed (pip install unsloth)")
        if not args.skip_unsloth:
            print("WARNING: Skipping Unsloth benchmarks")
            args.skip_unsloth = True

    results: list[BenchmarkResult] = []

    for training_type in args.training_types:
        output_subdir = Path(args.output_dir) / training_type
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Standard benchmark
        if not args.skip_standard:
            print(f"\nRunning standard {training_type.upper()} benchmark...")
            print(f"  Config: {CONFIG_PATHS[training_type]}")
            result = run_benchmark(
                training_type=training_type,
                max_steps=args.max_steps,
                enable_unsloth=False,
                output_dir=str(output_subdir / "standard"),
            )
            results.append(result)
            t = result.total_time_seconds
            v = result.peak_vram_gb
            print(f"  Time: {t:.2f}s, VRAM: {v:.2f}GB")
            if result.error:
                print(f"  Error: {result.error}")

            # Reset between runs
            reset_gpu_memory()

        # Unsloth benchmark
        if not args.skip_unsloth and unsloth_available:
            print(f"\nRunning Unsloth {training_type.upper()} benchmark...")
            print(f"  Config: {CONFIG_PATHS[training_type]}")
            result = run_benchmark(
                training_type=training_type,
                max_steps=args.max_steps,
                enable_unsloth=True,
                output_dir=str(output_subdir / "unsloth"),
            )
            results.append(result)
            t = result.total_time_seconds
            v = result.peak_vram_gb
            print(f"  Time: {t:.2f}s, VRAM: {v:.2f}GB")
            if result.error:
                print(f"  Error: {result.error}")

            # Reset between runs
            reset_gpu_memory()

    # Print summary
    print_results(results)

    # Save results to JSON
    results_file = Path(args.output_dir) / "benchmark_results.json"
    results_data = [
        {
            "name": r.name,
            "mode": r.mode,
            "training_type": r.training_type,
            "total_time_seconds": r.total_time_seconds,
            "peak_vram_gb": r.peak_vram_gb,
            "samples_per_second": r.samples_per_second,
            "steps_completed": r.steps_completed,
            "error": r.error,
        }
        for r in results
    ]
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
