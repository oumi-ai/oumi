#!/usr/bin/env python3
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

"""Collate evaluation results from a run directory into a flat table.

This script reads evaluation outputs from enterprise eval runs and produces
a CSV with one row per run, with columns for each benchmark-metric combination.

Usage:
```sh
# Collate a single run
python scripts/enterprise/collate_eval_results.py \
    --run-dirs output/enterprise/evaluation/20260105_185708_Qwen3-4B-Instruct-2507

# Collate multiple runs
python scripts/enterprise/collate_eval_results.py \
    --run-dirs output/enterprise/evaluation/20260105_* \
    --output results.csv

# Output both CSV and detailed JSON
python scripts/enterprise/collate_eval_results.py \
    --run-dirs output/enterprise/evaluation/20260105_185708_Qwen3-4B-Instruct-2507 \
    --output results.csv \
    --json results.json
```
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


# Define which metrics to extract for each task
TASK_METRICS = {
    "banking77": ["accuracy", "mean_response_chars", "num_correct", "num_total"],
    "pubmedqa": ["accuracy", "mean_response_chars", "micro_f1", "num_correct", "num_total"],
    "tatqa": ["exact_match", "f1", "boxed_rate", "mean_response_chars", "num_total"],
    "nl2sql": ["edit_similarity", "exact_match", "mean_response_chars", "num_total"],
    "control": [],  # lm_harness has different structure, handle separately
}

# IFEval metrics (from lm_harness)
IFEVAL_METRICS = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc", 
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]

# SimpleSafetyTests metrics
SAFETY_METRICS = ["safe_rate", "num_safe", "num_total"]


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, return None if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def extract_task_metrics(task_dir: Path) -> dict[str, Any]:
    """Extract metrics from a task output directory."""
    result = {}
    
    # Load task result
    task_result = load_json(task_dir / "task_result.json")
    if task_result:
        result["start_time"] = task_result.get("start_time")
        result["duration_sec"] = task_result.get("duration_sec")
        
        # Extract metrics from results
        results = task_result.get("results", {})
        for task_name, metrics in results.items():
            # Flatten metrics, skip nested dicts like per_class
            for key, value in metrics.items():
                if not isinstance(value, dict):
                    result[f"{key}"] = value
    
    # Load generation params
    gen_params = load_json(task_dir / "generation_params.json")
    if gen_params:
        result["temperature"] = gen_params.get("temperature")
        result["batch_size"] = gen_params.get("batch_size")
        result["max_new_tokens"] = gen_params.get("max_new_tokens")
        result["use_sampling"] = gen_params.get("use_sampling")
    
    # Load model params
    model_params = load_json(task_dir / "model_params.json")
    if model_params:
        result["model_name"] = model_params.get("model_name")
        result["model_max_length"] = model_params.get("model_max_length")
        result["torch_dtype"] = model_params.get("torch_dtype_str")
    
    return result


def extract_lm_harness_metrics(task_dir: Path) -> dict[str, Any]:
    """Extract metrics from lm_harness output (IFEval, etc.)."""
    result = {}
    
    task_result = load_json(task_dir / "task_result.json")
    if not task_result:
        return result
    
    result["start_time"] = task_result.get("start_time")
    result["duration_sec"] = task_result.get("duration_sec")
    
    results = task_result.get("results", {})
    
    # Handle IFEval metrics
    if "leaderboard_ifeval" in results:
        ifeval = results["leaderboard_ifeval"]
        for metric in IFEVAL_METRICS:
            full_key = f"{metric},none"
            if full_key in ifeval:
                result[f"ifeval_{metric}"] = ifeval[full_key]
    
    # Handle simple_safety_tests
    if "simple_safety_tests" in results:
        safety = results["simple_safety_tests"]
        for key, value in safety.items():
            if not isinstance(value, dict):
                result[f"safety_{key}"] = value
    
    return result


def collate_run(run_dir: Path) -> dict[str, Any]:
    """Collate all task results from a single run directory."""
    run_dir = Path(run_dir)
    
    # Extract run ID from directory name
    run_id = run_dir.name
    
    row = {
        "run_id": run_id,
        "run_dir": str(run_dir),
    }
    
    # Try to parse model name from run_id (format: YYYYMMDD_HHMMSS_ModelName)
    parts = run_id.split("_", 2)
    if len(parts) >= 3:
        row["timestamp"] = f"{parts[0]}_{parts[1]}"
        row["model_short"] = parts[2]
    
    # Process each task subdirectory
    for task_name in ["banking77", "pubmedqa", "tatqa", "nl2sql"]:
        task_dir = run_dir / task_name
        if task_dir.exists():
            # Look for custom_* subdirectory (new structure)
            custom_subdirs = list(task_dir.glob("custom_*"))
            if custom_subdirs:
                # Use the first (should only be one per task in a run)
                actual_task_dir = custom_subdirs[0]
            else:
                # Fallback to direct files (old structure)
                actual_task_dir = task_dir
            
            metrics = extract_task_metrics(actual_task_dir)
            # Prefix metrics with task name
            for key, value in metrics.items():
                if key not in ["model_name", "torch_dtype", "model_max_length"]:
                    row[f"{task_name}_{key}"] = value
                elif key == "model_name" and "model_name" not in row:
                    row["model_name"] = value
                elif key == "torch_dtype" and "torch_dtype" not in row:
                    row["torch_dtype"] = value
    
    # Process control evals (may have lm_harness subdirs)
    control_dir = run_dir / "control"
    if control_dir.exists():
        # Check for lm_harness subdirectories
        for subdir in control_dir.iterdir():
            if subdir.is_dir():
                if "lm_harness" in subdir.name:
                    metrics = extract_lm_harness_metrics(subdir)
                    for key, value in metrics.items():
                        row[f"control_{key}"] = value
                elif "custom" in subdir.name:
                    # SimpleSafetyTests is custom
                    metrics = extract_task_metrics(subdir)
                    for key, value in metrics.items():
                        if key not in ["model_name", "torch_dtype", "model_max_length"]:
                            row[f"safety_{key}"] = value
    
    return row


def collate_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Collate results from multiple run directories."""
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        # Skip collated output directories
        if run_dir.name == "collated":
            continue
        if run_dir.is_dir():
            row = collate_run(run_dir)
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write results to CSV with only key metrics (excludes gen params, start_time)."""
    if not rows:
        print("No results to write")
        return
    
    # Columns to exclude from CSV (keep in JSON)
    excluded_suffixes = ("_start_time", "_temperature", "_batch_size", 
                         "_max_new_tokens", "_use_sampling", "_model_max_length")
    excluded_exact = {"run_dir", "torch_dtype", "model_short"}
    
    # Get all unique columns across all rows, filtering out excluded ones
    all_columns = set()
    for row in rows:
        for col in row.keys():
            if col in excluded_exact:
                continue
            if any(col.endswith(suffix) for suffix in excluded_suffixes):
                continue
            all_columns.add(col)
    
    # Sort columns: metadata first, then by task
    def column_sort_key(col: str) -> tuple:
        # Priority order for column groups
        priority = {
            "run_id": 0,
            "timestamp": 1,
            "model_short": 2,
            "model_name": 3,
            "torch_dtype": 4,
            "run_dir": 5,
        }
        if col in priority:
            return (priority[col], col)
        
        # Group by task prefix
        task_order = ["banking77", "pubmedqa", "tatqa", "nl2sql", "control", "safety"]
        for i, task in enumerate(task_order):
            if col.startswith(task):
                return (10 + i, col)
        
        return (100, col)
    
    columns = sorted(all_columns, key=column_sort_key)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {len(rows)} row(s) to {output_path}")


def write_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write detailed results to JSON."""
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote detailed results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collate evaluation results into a flat table"
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        help="Run directory or directories to collate",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("eval_results.csv"),
        help="Output CSV file path (default: eval_results.csv)",
    )
    parser.add_argument(
        "--json", "-j",
        type=Path,
        default=None,
        help="Also output detailed JSON file",
    )
    args = parser.parse_args()
    
    # Collate all runs
    rows = collate_runs(args.run_dirs)
    
    if not rows:
        print("No results found in specified directories")
        return
    
    # Write outputs
    write_csv(rows, args.output)
    
    if args.json:
        write_json(rows, args.json)
    
    # Print summary
    print("\nSummary:")
    for row in rows:
        print(f"  {row.get('run_id', 'unknown')}: {row.get('model_name', 'unknown')}")


if __name__ == "__main__":
    main()

