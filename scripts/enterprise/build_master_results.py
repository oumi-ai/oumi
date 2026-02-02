#!/usr/bin/env python3
# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""NB: This is an ad hoc model-generated utility script. Use it for organizing results from experiment bundles as workflows develop.

Build master results CSV from all evaluation runs in evals directory.

This script walks through all eval directories and builds a comprehensive
master CSV with:
- All eval metrics (PubMedQA, TAT-QA, Banking77, NL2SQL, IFEval, Safety)
- Training hyperparameters from experiment_meta.json (if available)
- Model info and run metadata

Results can then be appended to a results tracking spreadsheet.

Usage (on cluster):
    python scripts/enterprise/build_master_results.py \\
        --evals-dir /data/tim/evals/ent \\
        --checkpoints-dir /data/tim/checkpoints \\
        --output /data/tim/evals/ent/master_results.csv

    # Dry run to see what would be processed
    python scripts/enterprise/build_master_results.py \\
        --evals-dir /data/tim/evals/ent --dry-run

    # Filter by model
    python scripts/enterprise/build_master_results.py \\
        --evals-dir /data/tim/evals/ent \\
        --filter smollm2

Usage (locally with downloaded results):
    python scripts/enterprise/build_master_results.py \\
        --evals-dir ~/Downloads/smollm2_1.7b-ft ~/Downloads/llama31-8b-ft \\
        --output ~/Downloads/master_results.csv

    # Combine all downloaded model results
    python scripts/enterprise/build_master_results.py \\
        --evals-dir ~/Downloads/*-ft \\
        --output ~/Downloads/all_results.csv
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Training hyperparameter columns to include
TRAINING_COLUMNS = [
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "weight_decay",
    "warmup_ratio",
    "lr_scheduler_type",
    "optimizer",
    "task_id",  # The task this model was trained on
]

# IFEval metrics (from lm_harness)
IFEVAL_METRICS = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


@dataclass
class EvalRun:
    """Represents a single evaluation run."""
    run_dir: Path
    run_id: str
    timestamp: str = ""
    model_name: str = ""
    model_short: str = ""
    run_type: str = ""  # "baseline" or "finetuned"
    model_family: str = ""  # e.g., "smollm2", "llama32", etc.
    
    # Eval metrics
    metrics: dict = field(default_factory=dict)
    
    # Training hyperparameters (if finetuned)
    training_params: dict = field(default_factory=dict)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, return None if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def extract_task_metrics(task_dir: Path) -> dict[str, Any]:
    """Extract metrics from a task output directory."""
    result = {}

    task_result = load_json(task_dir / "task_result.json")
    if task_result:
        result["duration_sec"] = task_result.get("duration_sec")

        results = task_result.get("results", {})
        for task_name, metrics in results.items():
            for key, value in metrics.items():
                if not isinstance(value, dict):
                    result[key] = value

    model_params = load_json(task_dir / "model_params.json")
    if model_params:
        result["model_name"] = model_params.get("model_name")

    return result


def extract_lm_harness_metrics(task_dir: Path) -> dict[str, Any]:
    """Extract metrics from lm_harness output (IFEval, etc.)."""
    result = {}

    task_result = load_json(task_dir / "task_result.json")
    if not task_result:
        return result

    result["duration_sec"] = task_result.get("duration_sec")

    results = task_result.get("results", {})

    if "leaderboard_ifeval" in results:
        ifeval = results["leaderboard_ifeval"]
        for metric in IFEVAL_METRICS:
            full_key = f"{metric},none"
            if full_key in ifeval:
                result[f"ifeval_{metric}"] = ifeval[full_key]

    return result


def extract_safety_metrics(task_dir: Path) -> dict[str, Any]:
    """Extract SimpleSafetyTests metrics."""
    result = {}

    task_result = load_json(task_dir / "task_result.json")
    if not task_result:
        return result

    result["duration_sec"] = task_result.get("duration_sec")

    results = task_result.get("results", {})
    if "simple_safety_tests" in results:
        safety = results["simple_safety_tests"]
        for key, value in safety.items():
            if not isinstance(value, dict):
                result[key] = value

    return result


def parse_run_id(run_id: str) -> tuple[str, str]:
    """Parse run_id to extract timestamp and model_short."""
    parts = run_id.split("_", 2)
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}", parts[2]
    elif len(parts) == 2:
        return f"{parts[0]}_{parts[1]}", ""
    return "", run_id


def infer_model_family(run_dir: Path, model_name: str) -> str:
    """Infer model family from directory path or model name."""
    path_str = str(run_dir).lower()
    model_lower = model_name.lower()

    if "smollm2" in path_str or "smollm" in model_lower:
        return "smollm2"
    elif "llama31" in path_str or "llama-3.1" in model_lower or "llama-3_1" in path_str:
        return "llama31"
    elif "llama32" in path_str or "llama-3.2" in model_lower or "llama-3_2" in path_str:
        return "llama32"
    elif "gemma" in path_str or "gemma" in model_lower:
        return "gemma3"
    elif "qwen" in path_str or "qwen" in model_lower:
        return "qwen3"
    return "unknown"


def infer_run_type(run_dir: Path, model_name: str = "") -> str:
    """Infer if this is a baseline or finetuned run."""
    path_str = str(run_dir).lower()
    run_id = run_dir.name
    
    # Check path for baseline indicators
    if "baseline" in path_str:
        return "baseline"
    
    # Check if run_id looks like a HF model name (no timestamp prefix)
    # Baseline runs often have names like "SmolLM2-1_7B-Instruct" or "Llama-3_2-1B-Instruct"
    if not re.match(r"^\d{8}_\d{6}_", run_id):
        return "baseline"
    
    # Check if model_name is a HF path (starts with org/) without /data/ prefix
    if model_name and not model_name.startswith("/"):
        return "baseline"
    
    return "finetuned"


def infer_hyperparams_from_suffix(run_id: str) -> dict[str, Any]:
    """Infer training hyperparameters from run_id suffix conventions.
    
    Common patterns:
    - "3ep" → num_train_epochs: 3
    - "8e6" or "8e-6" → learning_rate: 8e-6
    - "1e5" or "1e-5" → learning_rate: 1e-5
    - "ebs32" → effective_batch_size: 32
    - "gas4" → gradient_accumulation_steps: 4
    """
    params = {}
    
    # Extract epochs: "3ep", "5ep" etc
    epoch_match = re.search(r"(\d+)ep", run_id.lower())
    if epoch_match:
        params["num_train_epochs"] = int(epoch_match.group(1))
    
    # Extract learning rate: "8e6", "8e-6", "1e5", "1e-5" etc
    lr_match = re.search(r"(\d+)e-?(\d+)", run_id.lower())
    if lr_match and "ebs" not in run_id[lr_match.start():lr_match.start()+10].lower():
        from decimal import Decimal
        mantissa = int(lr_match.group(1))
        exponent = int(lr_match.group(2))
        # Use Decimal for exact representation to avoid 7.000000000000001e-05
        params["learning_rate"] = float(Decimal(mantissa) * Decimal(10) ** Decimal(-exponent))
    
    # Extract effective batch size: "ebs32", "ebs64" etc
    ebs_match = re.search(r"ebs(\d+)", run_id.lower())
    if ebs_match:
        params["effective_batch_size"] = int(ebs_match.group(1))
    
    # Extract gradient accumulation steps: "gas4", "gas16" etc
    gas_match = re.search(r"gas(\d+)", run_id.lower())
    if gas_match:
        params["gradient_accumulation_steps"] = int(gas_match.group(1))
    
    return params


def round_learning_rate(lr: float) -> float:
    """Round learning rate to a clean scientific notation value.
    
    e.g., 1.9856159103477085e-05 → 2e-05
          9.998532555055942e-06 → 1e-05
    """
    if lr <= 0:
        return lr
    
    import math
    from decimal import Decimal
    
    # Get the exponent
    exponent = math.floor(math.log10(lr))
    # Get the mantissa and round it
    mantissa = lr / (10 ** exponent)
    rounded_mantissa = round(mantissa)
    
    # Handle case where rounding pushes mantissa to 10
    if rounded_mantissa >= 10:
        rounded_mantissa = 1
        exponent += 1
    
    # Use Decimal for exact representation to avoid floating point artifacts
    result = Decimal(rounded_mantissa) * Decimal(10) ** Decimal(exponent)
    return float(result)


def infer_task_from_path(path: Path) -> str | None:
    """Infer task ID from checkpoint path."""
    path_str = str(path).lower()
    for task in ["pubmedqa", "tatqa", "banking77", "nl2sql"]:
        if task in path_str:
            return task
    return None


def infer_task_from_run_id(run_id: str) -> str | None:
    """Infer task ID from run_id string."""
    run_id_lower = run_id.lower()
    for task in ["pubmedqa", "tatqa", "banking77", "nl2sql"]:
        if task in run_id_lower:
            return task
    return None


# Known dataset sizes for EBS/GAS inference (from train.jsonl line counts)
DATASET_SIZES = {
    "pubmedqa": 800,
    "tatqa": 13096,
    "banking77": 9903,
    "nl2sql": 503,
}

# Default number of GPUs (can't always infer this)
DEFAULT_NUM_GPUS = 8


def extract_from_trainer_state(checkpoint_path: Path) -> dict[str, Any]:
    """Extract hyperparameters from trainer_state.json (HuggingFace format)."""
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if not trainer_state_path.exists():
        return {}
    
    state = load_json(trainer_state_path)
    if not state:
        return {}
    
    result = {}
    
    # Direct fields
    num_epochs = state.get("num_train_epochs")
    per_device_bs = state.get("train_batch_size")
    total_steps = state.get("max_steps") or state.get("global_step")
    
    if num_epochs:
        result["num_train_epochs"] = int(num_epochs)
    if per_device_bs:
        result["per_device_train_batch_size"] = per_device_bs
    
    # Extract max learning rate from log_history
    log_history = state.get("log_history", [])
    lrs = [x.get("learning_rate") for x in log_history if x.get("learning_rate")]
    if lrs:
        result["learning_rate"] = round_learning_rate(max(lrs))
    
    # Try to infer GAS and EBS from dataset size
    task = infer_task_from_path(checkpoint_path)
    if task and task in DATASET_SIZES and num_epochs and total_steps and per_device_bs:
        dataset_size = DATASET_SIZES[task]
        # EBS = (dataset_size * num_epochs) / total_steps
        ebs = int(round((dataset_size * num_epochs) / total_steps))
        result["effective_batch_size"] = ebs
        
        # GAS = EBS / (per_device_bs * num_gpus)
        gas = ebs // (per_device_bs * DEFAULT_NUM_GPUS)
        if gas > 0:
            result["gradient_accumulation_steps"] = gas
    
    return result


def find_experiment_metadata(
    model_name: str,
    checkpoints_dir: Path | None,
) -> dict[str, Any]:
    """Find experiment_meta.json for a given model/checkpoint."""
    checkpoint_path = None
    
    # Determine checkpoint path
    if model_name.startswith("/"):
        checkpoint_path = Path(model_name)
    elif checkpoints_dir and checkpoints_dir.exists():
        checkpoint_name = Path(model_name).name
        checkpoint_path = checkpoints_dir / checkpoint_name
    
    if not checkpoint_path or not checkpoint_path.exists():
        return {}
    
    # Try experiment_meta.json first (preferred)
    meta_path = checkpoint_path / "experiment_meta.json"
    if meta_path.exists():
        return load_json(meta_path) or {}
    
    # Fall back to trainer_state.json
    trainer_meta = extract_from_trainer_state(checkpoint_path)
    if trainer_meta:
        return trainer_meta
    
    # Search for matching metadata files in checkpoints dir
    if checkpoints_dir and checkpoints_dir.exists():
        checkpoint_name = Path(model_name).name
        for meta_file in checkpoints_dir.glob("*/experiment_meta.json"):
            meta = load_json(meta_file)
            if meta and meta.get("run_name") == checkpoint_name:
                return meta

    return {}


def collate_run(run_dir: Path, checkpoints_dir: Path | None = None) -> EvalRun:
    """Collate all task results from a single run directory."""
    run_dir = Path(run_dir)
    run_id = run_dir.name

    timestamp, model_short = parse_run_id(run_id)

    run = EvalRun(
        run_dir=run_dir,
        run_id=run_id,
        timestamp=timestamp,
        model_short=model_short,
    )

    # Process each task subdirectory
    for task_name in ["banking77", "pubmedqa", "tatqa", "nl2sql"]:
        task_dir = run_dir / task_name
        if task_dir.exists():
            custom_subdirs = list(task_dir.glob("custom_*"))
            if custom_subdirs:
                actual_task_dir = custom_subdirs[0]
            else:
                actual_task_dir = task_dir

            metrics = extract_task_metrics(actual_task_dir)
            for key, value in metrics.items():
                if key == "model_name":
                    if not run.model_name:
                        run.model_name = value
                else:
                    run.metrics[f"{task_name}_{key}"] = value

    # Process control evals (lm_harness for IFEval, custom for safety)
    control_dir = run_dir / "control"
    if control_dir.exists():
        for subdir in control_dir.iterdir():
            if subdir.is_dir():
                if "lm_harness" in subdir.name:
                    metrics = extract_lm_harness_metrics(subdir)
                    for key, value in metrics.items():
                        run.metrics[f"control_{key}"] = value
                elif "custom" in subdir.name:
                    metrics = extract_safety_metrics(subdir)
                    for key, value in metrics.items():
                        run.metrics[f"safety_{key}"] = value

    # Infer model family and run type
    run.model_family = infer_model_family(run_dir, run.model_name)
    run.run_type = infer_run_type(run_dir, run.model_name)

    # Try to find training hyperparameters from multiple sources
    if run.run_type == "finetuned":
        # 1. Infer task_id from run name
        task_id = infer_task_from_run_id(run.run_id)
        if task_id:
            run.training_params["task_id"] = task_id
        
        # 2. Get suffix inference (explicit in run name, most reliable for GAS)
        suffix_params = infer_hyperparams_from_suffix(run.run_id)
        
        # 3. Then get metadata from checkpoint (experiment_meta.json or trainer_state.json)
        meta_params = {}
        if run.model_name:
            meta = find_experiment_metadata(run.model_name, checkpoints_dir)
            if meta:
                for col in TRAINING_COLUMNS:
                    if col in meta and meta[col] is not None:
                        value = meta[col]
                        # Round learning rate for cleaner display
                        if col == "learning_rate" and isinstance(value, float):
                            value = round_learning_rate(value)
                        meta_params[col] = value
        
        # 4. Merge: start with meta, then override with suffix (suffix is explicit)
        run.training_params.update(meta_params)
        run.training_params.update(suffix_params)  # Suffix overrides inferred values

    return run


def is_eval_run(path: Path) -> bool:
    """Check if a directory looks like an evaluation run."""
    if not path.is_dir():
        return False
    return any(
        (path / task).exists()
        for task in ["pubmedqa", "tatqa", "banking77", "nl2sql", "control"]
    )


def find_all_runs(evals_dir: Path) -> list[Path]:
    """Find all evaluation run directories."""
    runs = []
    evals_dir = Path(evals_dir)

    # First check if evals_dir itself IS an eval run (when glob-expanded paths are passed)
    if is_eval_run(evals_dir):
        return [evals_dir]

    # Check if evals_dir contains runs directly (flat structure)
    for item in evals_dir.iterdir():
        if item.is_dir() and item.name != "collated":
            if is_eval_run(item):
                runs.append(item)

    # If we found runs directly, return them
    if runs:
        return sorted(runs)

    # Otherwise, check known subdirectory patterns (nested structure)
    subdirs_to_check = [
        "baselines",
        "ft",
        "llama31-8b-ft",
        "llama32-3b-ft",
        "llama32-ft",
        "smollm2_1.7b-ft",
    ]

    for subdir_name in subdirs_to_check:
        subdir = evals_dir / subdir_name
        if subdir.exists() and subdir.is_dir():
            for item in subdir.iterdir():
                if item.is_dir() and item.name != "collated":
                    if is_eval_run(item):
                        runs.append(item)
                    else:
                        # Could be a nested structure (e.g., ft/baseline-rerun/*)
                        for nested in item.iterdir():
                            if nested.is_dir() and is_eval_run(nested):
                                runs.append(nested)

    return sorted(runs)


def run_to_row(run: EvalRun) -> dict[str, Any]:
    """Convert EvalRun to a flat dict for CSV output."""
    row = {
        "run_id": run.run_id,
        "timestamp": run.timestamp,
        "model_name": run.model_name,
        "model_family": run.model_family,
        "run_type": run.run_type,
    }

    # Add training params
    for key, value in run.training_params.items():
        row[key] = value

    # Add metrics
    row.update(run.metrics)

    return row


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write results to CSV."""
    if not rows:
        print("No results to write")
        return

    # Get all unique columns
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())

    # Define column ordering
    priority_cols = [
        "run_id",
        "timestamp",
        "model_name",
        "model_family",
        "run_type",
    ] + TRAINING_COLUMNS

    # Key metrics
    key_metrics = [
        "pubmedqa_accuracy",
        "pubmedqa_macro_f1",
        "tatqa_f1",
        "tatqa_exact_match",
        "banking77_accuracy",
        "nl2sql_exact_match",
        "nl2sql_edit_similarity",
        "control_ifeval_prompt_level_strict_acc",
        "control_ifeval_inst_level_strict_acc",
        "safety_safe_rate",
    ]

    # Build ordered columns
    ordered_cols = []
    for col in priority_cols + key_metrics:
        if col in all_columns:
            ordered_cols.append(col)
            all_columns.discard(col)

    # Add remaining columns sorted
    ordered_cols.extend(sorted(all_columns))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


def write_json(rows: list[dict], output_path: Path) -> None:
    """Write detailed results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote detailed results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build master results CSV from all evaluation runs"
    )
    parser.add_argument(
        "--evals-dir",
        type=Path,
        nargs="+",
        default=[Path("/data/tim/evals/ent")],
        help="Root directory or directories containing evaluation results",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("/data/tim/checkpoints"),
        help="Directory containing checkpoints with experiment_meta.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: <evals-dir>/master_results.csv)",
    )
    parser.add_argument(
        "--json",
        "-j",
        type=Path,
        default=None,
        help="Also output detailed JSON file",
    )
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        default=None,
        help="Filter runs by pattern (e.g., 'smollm2', 'llama32')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without writing output",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.evals_dir[0] / "master_results.csv"

    # Find all runs from all directories
    runs = []
    for evals_dir in args.evals_dir:
        print(f"Scanning: {evals_dir}")
        runs.extend(find_all_runs(evals_dir))

    if args.filter:
        pattern = args.filter.lower()
        runs = [r for r in runs if pattern in str(r).lower()]

    print(f"Found {len(runs)} evaluation runs")

    if args.dry_run:
        print("\nRuns that would be processed:")
        for run in runs:
            print(f"  {run}")
        return

    # Collate each run
    results = []
    for run_dir in runs:
        try:
            run = collate_run(run_dir, args.checkpoints_dir)
            row = run_to_row(run)
            results.append(row)
            print(f"  ✓ {run.run_id}")
        except Exception as e:
            print(f"  ✗ {run_dir.name}: {e}")

    if not results:
        print("No results found")
        return

    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""))

    # Write outputs
    write_csv(results, args.output)

    if args.json is None:
        args.json = args.output.with_suffix(".json")
    write_json(results, args.json)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary by model family:")
    print(f"{'='*60}")

    from collections import Counter
    family_counts = Counter(r.get("model_family", "unknown") for r in results)
    type_counts = Counter(r.get("run_type", "unknown") for r in results)

    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count} runs")

    print(f"\nBy run type:")
    for run_type, count in sorted(type_counts.items()):
        print(f"  {run_type}: {count} runs")


if __name__ == "__main__":
    main()

