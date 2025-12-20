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

"""Compare two benchmark result files and generate a diff report.

Usage:
    python benchmarks/compare_benchmarks.py baseline.json optimized.json

Output:
    - Console table showing improvements/regressions
    - Optional markdown report file
"""

import argparse
import json
import sys
from dataclasses import dataclass


@dataclass
class MetricComparison:
    """Comparison of a single metric between baseline and optimized."""

    metric_name: str
    baseline_value: float
    optimized_value: float
    unit: str = ""

    @property
    def absolute_change(self) -> float:
        return self.optimized_value - self.baseline_value

    @property
    def percent_change(self) -> float:
        if self.baseline_value == 0:
            return 0.0
        return (
            (self.optimized_value - self.baseline_value) / self.baseline_value
        ) * 100

    @property
    def is_improvement(self) -> bool:
        """Determine if the change is an improvement (lower is better for time/memory)."""
        # For time and memory metrics, lower is better
        if any(
            x in self.metric_name.lower()
            for x in ["time", "duration", "memory", "seconds"]
        ):
            return self.absolute_change < 0
        # For throughput metrics, higher is better
        if any(x in self.metric_name.lower() for x in ["throughput", "per_second"]):
            return self.absolute_change > 0
        return False

    def format_change(self) -> str:
        """Format the change with improvement indicator."""
        pct = self.percent_change
        indicator = "+" if pct > 0 else ""

        if self.is_improvement:
            status = "[BETTER]"
        elif abs(pct) < 5:
            status = "[~SAME]"
        else:
            status = "[WORSE]"

        return f"{indicator}{pct:.1f}% {status}"


def load_benchmark_results(path: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_timing_metrics(result: dict) -> dict[str, float]:
    """Extract all timing metrics from a dataset benchmark result."""
    metrics = {}

    # Extract from timing result objects
    timing_fields = [
        "load_raw",
        "to_hf_map",
        "to_hf_iterable",
        "iteration_first_100",
        "iteration_full",
        "sampling",
        "oversampling",
    ]

    for field in timing_fields:
        timing = result.get(field)
        if timing and isinstance(timing, dict):
            if timing.get("success", False):
                metrics[f"{field}_duration_s"] = timing.get("duration_seconds", 0)
                metrics[f"{field}_peak_memory_mb"] = timing.get("peak_memory_mb", 0)

    # Top-level metrics
    if result.get("examples_per_second", 0) > 0:
        metrics["examples_per_second"] = result["examples_per_second"]

    if result.get("total_benchmark_time", 0) > 0:
        metrics["total_benchmark_time_s"] = result["total_benchmark_time"]

    return metrics


def compare_datasets(
    baseline_results: list[dict], optimized_results: list[dict]
) -> dict[str, list[MetricComparison]]:
    """Compare metrics for matching datasets."""
    comparisons = {}

    # Index optimized results by dataset key
    optimized_by_key = {r["dataset_key"]: r for r in optimized_results}

    for baseline in baseline_results:
        key = baseline["dataset_key"]
        if key not in optimized_by_key:
            print(f"Warning: {key} not found in optimized results, skipping")
            continue

        optimized = optimized_by_key[key]

        baseline_metrics = extract_timing_metrics(baseline)
        optimized_metrics = extract_timing_metrics(optimized)

        dataset_comparisons = []
        for metric_name in sorted(baseline_metrics.keys()):
            if metric_name in optimized_metrics:
                unit = (
                    "s" if "duration" in metric_name or "time" in metric_name else "MB"
                )
                if "per_second" in metric_name:
                    unit = "ex/s"

                comparison = MetricComparison(
                    metric_name=metric_name,
                    baseline_value=baseline_metrics[metric_name],
                    optimized_value=optimized_metrics[metric_name],
                    unit=unit,
                )
                dataset_comparisons.append(comparison)

        comparisons[key] = dataset_comparisons

    return comparisons


def compare_summaries(baseline: dict, optimized: dict) -> list[MetricComparison]:
    """Compare summary statistics."""
    comparisons = []

    summary_metrics = [
        ("avg_load_time_seconds", "s"),
        ("avg_to_hf_time_seconds", "s"),
        ("avg_examples_per_second", "ex/s"),
        ("peak_memory_mb", "MB"),
        ("total_benchmark_time_seconds", "s"),
    ]

    baseline_summary = baseline.get("summary", {})
    optimized_summary = optimized.get("summary", {})

    for metric_name, unit in summary_metrics:
        if metric_name in baseline_summary and metric_name in optimized_summary:
            comparisons.append(
                MetricComparison(
                    metric_name=metric_name,
                    baseline_value=baseline_summary[metric_name],
                    optimized_value=optimized_summary[metric_name],
                    unit=unit,
                )
            )

    return comparisons


def print_comparison_table(
    title: str, comparisons: list[MetricComparison], show_all: bool = False
):
    """Print a formatted comparison table."""
    if not comparisons:
        return

    print(f"\n{title}")
    print("-" * 90)

    # Header
    print(f"{'Metric':<35} {'Baseline':>12} {'Optimized':>12} {'Change':>20}")
    print("-" * 90)

    for c in comparisons:
        # Skip unchanged metrics unless show_all
        if not show_all and abs(c.percent_change) < 1:
            continue

        baseline_str = f"{c.baseline_value:.2f} {c.unit}"
        optimized_str = f"{c.optimized_value:.2f} {c.unit}"

        print(
            f"{c.metric_name:<35} {baseline_str:>12} {optimized_str:>12} {c.format_change():>20}"
        )

    print("-" * 90)


def generate_markdown_report(
    baseline: dict,
    optimized: dict,
    dataset_comparisons: dict[str, list[MetricComparison]],
    summary_comparisons: list[MetricComparison],
    output_path: str,
):
    """Generate a markdown report of the comparison."""
    lines = [
        "# Benchmark Comparison Report",
        "",
        "## Overview",
        "",
        f"- **Baseline**: {baseline.get('timestamp', 'N/A')}",
        f"- **Optimized**: {optimized.get('timestamp', 'N/A')}",
        f"- **Oumi Version (baseline)**: {baseline.get('oumi_version', 'N/A')}",
        f"- **Oumi Version (optimized)**: {optimized.get('oumi_version', 'N/A')}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Baseline | Optimized | Change |",
        "|--------|----------|-----------|--------|",
    ]

    for c in summary_comparisons:
        status = (
            "better"
            if c.is_improvement
            else ("worse" if abs(c.percent_change) > 5 else "~same")
        )
        lines.append(
            f"| {c.metric_name} | {c.baseline_value:.2f} {c.unit} | "
            f"{c.optimized_value:.2f} {c.unit} | {c.percent_change:+.1f}% ({status}) |"
        )

    lines.extend(["", "## Per-Dataset Results", ""])

    for dataset_key, comparisons in dataset_comparisons.items():
        lines.append(f"### {dataset_key}")
        lines.append("")
        lines.append("| Metric | Baseline | Optimized | Change |")
        lines.append("|--------|----------|-----------|--------|")

        for c in comparisons:
            if abs(c.percent_change) < 1:
                continue
            status = (
                "better"
                if c.is_improvement
                else ("worse" if abs(c.percent_change) > 5 else "~same")
            )
            lines.append(
                f"| {c.metric_name} | {c.baseline_value:.2f} {c.unit} | "
                f"{c.optimized_value:.2f} {c.unit} | {c.percent_change:+.1f}% ({status}) |"
            )

        lines.append("")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nMarkdown report saved to: {output_path}")


def compute_overall_improvement(
    summary_comparisons: list[MetricComparison],
) -> dict[str, float]:
    """Compute overall improvement percentages."""
    improvements = {}

    for c in summary_comparisons:
        if "time" in c.metric_name.lower() or "memory" in c.metric_name.lower():
            # For time/memory, negative change is improvement
            improvements[c.metric_name] = -c.percent_change
        elif "per_second" in c.metric_name.lower():
            # For throughput, positive change is improvement
            improvements[c.metric_name] = c.percent_change

    return improvements


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("baseline", type=str, help="Path to baseline benchmark JSON")
    parser.add_argument("optimized", type=str, help="Path to optimized benchmark JSON")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for markdown report output",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all metrics including unchanged ones",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading baseline: {args.baseline}")
    baseline = load_benchmark_results(args.baseline)

    print(f"Loading optimized: {args.optimized}")
    optimized = load_benchmark_results(args.optimized)

    # Compare
    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON")
    print("=" * 90)

    print(f"\nBaseline: {baseline.get('timestamp', 'N/A')}")
    print(f"Optimized: {optimized.get('timestamp', 'N/A')}")

    # Summary comparison
    summary_comparisons = compare_summaries(baseline, optimized)
    print_comparison_table("SUMMARY METRICS", summary_comparisons, args.show_all)

    # Dataset comparisons
    dataset_comparisons = compare_datasets(
        baseline.get("results", []), optimized.get("results", [])
    )

    for dataset_key, comparisons in dataset_comparisons.items():
        print_comparison_table(f"Dataset: {dataset_key}", comparisons, args.show_all)

    # Overall improvement
    improvements = compute_overall_improvement(summary_comparisons)
    if improvements:
        print("\n" + "=" * 90)
        print("OVERALL IMPROVEMENT SUMMARY")
        print("=" * 90)
        for metric, improvement in improvements.items():
            status = "IMPROVED" if improvement > 0 else "REGRESSED"
            print(f"  {metric}: {improvement:+.1f}% ({status})")

    # Generate markdown report if requested
    if args.output:
        generate_markdown_report(
            baseline,
            optimized,
            dataset_comparisons,
            summary_comparisons,
            args.output,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
