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

"""Benchmark script for measuring import times of oumi and its submodules.

This script measures import times to help identify slow imports and establish
a baseline for optimization efforts using lazy-loader or other techniques.

Usage:
    python scripts/benchmark_imports.py [--iterations N] [--detailed] [--importtime]

Examples:
    # Quick benchmark with default settings
    python scripts/benchmark_imports.py

    # Detailed benchmark with more iterations
    python scripts/benchmark_imports.py --iterations 10 --detailed

    # Use Python's importtime for deep analysis
    python scripts/benchmark_imports.py --importtime
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImportResult:
    """Result of an import timing measurement."""

    module: str
    time_seconds: float
    success: bool
    error: str | None = None


def clear_import_cache(module_prefix: str = "oumi") -> None:
    """Clear cached imports for accurate measurements."""
    modules_to_remove = [
        name for name in sys.modules if name.startswith(module_prefix)
    ]
    for name in modules_to_remove:
        del sys.modules[name]


def time_import(module_name: str) -> ImportResult:
    """Time how long it takes to import a module in a subprocess.

    Uses subprocess to ensure clean import state each time.
    """
    code = f"""
import time
start = time.perf_counter()
import {module_name}
end = time.perf_counter()
print(end - start)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent,
            env={
                **dict(__import__("os").environ),
                "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
            },
        )
        if result.returncode != 0:
            return ImportResult(
                module=module_name,
                time_seconds=0.0,
                success=False,
                error=result.stderr.strip(),
            )
        return ImportResult(
            module=module_name,
            time_seconds=float(result.stdout.strip()),
            success=True,
        )
    except Exception as e:
        return ImportResult(
            module=module_name,
            time_seconds=0.0,
            success=False,
            error=str(e),
        )


def time_import_in_process(module_name: str) -> ImportResult:
    """Time import within this process (faster but less accurate for repeated tests)."""
    clear_import_cache()
    try:
        start = time.perf_counter()
        __import__(module_name)
        end = time.perf_counter()
        return ImportResult(
            module=module_name,
            time_seconds=end - start,
            success=True,
        )
    except Exception as e:
        return ImportResult(
            module=module_name,
            time_seconds=0.0,
            success=False,
            error=str(e),
        )


def run_importtime_analysis(module_name: str) -> str:
    """Run Python's -X importtime to get detailed import timing."""
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
        },
    )
    return result.stderr


def parse_importtime_output(output: str) -> list[tuple[str, float, float]]:
    """Parse importtime output into list of (module, self_time_us, cumulative_time_us)."""
    results = []
    for line in output.strip().split("\n"):
        if "import time:" in line:
            # Format: "import time:      self [us] |    cumulative | imported"
            continue
        if "|" in line:
            parts = line.split("|")
            if len(parts) >= 3:
                try:
                    self_time = float(parts[0].replace("import time:", "").strip())
                    cumulative = float(parts[1].strip())
                    module = parts[2].strip()
                    results.append((module, self_time, cumulative))
                except ValueError:
                    continue
    return results


def benchmark_modules(
    modules: list[str],
    iterations: int = 3,
    use_subprocess: bool = True,
) -> dict[str, list[float]]:
    """Benchmark multiple modules, returning timing data."""
    results: dict[str, list[float]] = {m: [] for m in modules}

    for i in range(iterations):
        print(f"  Iteration {i + 1}/{iterations}...", end="\r")
        for module in modules:
            if use_subprocess:
                result = time_import(module)
            else:
                result = time_import_in_process(module)

            if result.success:
                results[module].append(result.time_seconds)
            else:
                print(f"\n  Warning: Failed to import {module}: {result.error}")

    print(" " * 40, end="\r")  # Clear progress line
    return results


def compute_statistics(times: list[float]) -> dict[str, float]:
    """Compute basic statistics for timing data."""
    if not times:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    mean = sum(times) / len(times)
    min_val = min(times)
    max_val = max(times)

    if len(times) > 1:
        variance = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
        std = variance**0.5
    else:
        std = 0.0

    return {"mean": mean, "min": min_val, "max": max_val, "std": std}


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def print_results_table(
    results: dict[str, list[float]],
    title: str = "Import Timing Results",
) -> None:
    """Print results in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")
    print(f"{'Module':<40} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-' * 70}")

    # Sort by mean time, descending
    sorted_items = sorted(
        results.items(),
        key=lambda x: compute_statistics(x[1])["mean"] if x[1] else 0,
        reverse=True,
    )

    for module, times in sorted_items:
        if not times:
            print(f"{module:<40} {'FAILED':>10} {'-':>10} {'-':>10}")
            continue
        stats = compute_statistics(times)
        print(
            f"{module:<40} "
            f"{format_time(stats['mean']):>10} "
            f"{format_time(stats['min']):>10} "
            f"{format_time(stats['max']):>10}"
        )

    print(f"{'=' * 70}\n")


def print_importtime_analysis(module: str, top_n: int = 30) -> None:
    """Print detailed importtime analysis for a module."""
    print(f"\n{'=' * 80}")
    print(f" Detailed Import Time Analysis for: {module}")
    print(f" (Using Python's -X importtime)")
    print(f"{'=' * 80}")

    output = run_importtime_analysis(module)
    parsed = parse_importtime_output(output)

    if not parsed:
        print("No import timing data available.")
        return

    # Sort by self time (what each module contributes, not including children)
    sorted_by_self = sorted(parsed, key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} modules by self-time (their own import cost):")
    print(f"{'-' * 80}")
    print(f"{'Module':<55} {'Self':>10} {'Cumulative':>12}")
    print(f"{'-' * 80}")

    for module_name, self_time, cumulative in sorted_by_self[:top_n]:
        # Convert microseconds to more readable format
        self_str = format_time(self_time / 1_000_000)
        cumul_str = format_time(cumulative / 1_000_000)
        # Truncate long module names
        if len(module_name) > 55:
            display_name = "..." + module_name[-52:]
        else:
            display_name = module_name
        print(f"{display_name:<55} {self_str:>10} {cumul_str:>12}")

    print(f"{'=' * 80}\n")


def main() -> None:
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark import times for oumi modules"
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=3,
        help="Number of iterations for each import (default: 3)",
    )
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Run detailed benchmarks on more submodules",
    )
    parser.add_argument(
        "--importtime",
        "-i",
        action="store_true",
        help="Use Python's -X importtime for detailed analysis",
    )
    parser.add_argument(
        "--subprocess",
        "-s",
        action="store_true",
        default=True,
        help="Use subprocess for clean imports (default: True)",
    )
    parser.add_argument(
        "--no-subprocess",
        action="store_false",
        dest="subprocess",
        help="Use in-process imports (faster but less accurate)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" Oumi Import Benchmark")
    print("=" * 70)
    print(f" Python: {sys.version.split()[0]}")
    print(f" Iterations: {args.iterations}")
    print(f" Mode: {'subprocess' if args.subprocess else 'in-process'}")
    print("=" * 70)

    # Core modules to benchmark
    core_modules = [
        "oumi",
        "oumi.core",
        "oumi.core.configs",
        "oumi.core.types",
        "oumi.utils.logging",
    ]

    # Extended modules for detailed benchmark
    extended_modules = [
        "oumi.builders",
        "oumi.datasets",
        "oumi.models",
        "oumi.inference",
        "oumi.judges",
        "oumi.launcher",
        "oumi.core.inference",
        "oumi.core.tokenizers",
        "oumi.core.datasets",
    ]

    # Common heavy dependencies
    dependency_modules = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "numpy",
        "pandas",
    ]

    print("\n📊 Benchmarking core oumi modules...")
    core_results = benchmark_modules(
        core_modules,
        iterations=args.iterations,
        use_subprocess=args.subprocess,
    )
    print_results_table(core_results, "Core Module Import Times")

    if args.detailed:
        print("\n📊 Benchmarking extended oumi modules...")
        extended_results = benchmark_modules(
            extended_modules,
            iterations=args.iterations,
            use_subprocess=args.subprocess,
        )
        print_results_table(extended_results, "Extended Module Import Times")

        print("\n📊 Benchmarking common dependencies...")
        dep_results = benchmark_modules(
            dependency_modules,
            iterations=args.iterations,
            use_subprocess=args.subprocess,
        )
        print_results_table(dep_results, "Dependency Import Times")

    if args.importtime:
        print_importtime_analysis("oumi", top_n=30)

        if args.detailed:
            print_importtime_analysis("oumi.core.configs", top_n=20)
            print_importtime_analysis("oumi.builders", top_n=20)

    # Summary
    print("\n" + "=" * 70)
    print(" Summary & Recommendations")
    print("=" * 70)

    if core_results.get("oumi"):
        oumi_time = compute_statistics(core_results["oumi"])["mean"]
        print(f"\n Total `import oumi` time: {format_time(oumi_time)}")

        if oumi_time > 1.0:
            print("\n ⚠️  Import time is > 1 second - lazy loading recommended!")
        elif oumi_time > 0.5:
            print("\n ⚠️  Import time is > 500ms - consider lazy loading")
        elif oumi_time > 0.1:
            print("\n ✓  Import time is acceptable but could be improved")
        else:
            print("\n ✅ Import time is good!")

    print("\n Potential optimization targets:")
    print("   - Heavy dependencies (torch, transformers) imported at module level")
    print("   - Large __init__.py files with many imports")
    print("   - Modules that register handlers/callbacks on import")
    print("\n Run with --importtime for detailed breakdown of slow imports")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
