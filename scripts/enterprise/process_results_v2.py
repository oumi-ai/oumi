"""Collate eval results and generate comparison plot.

```sh
# usage:
python scripts/enterprise/process_results_v2.py \
    --eval-dirs /data/demo/evals/baseline/* /data/demo/evals/sft/* \
    --checkpoint-dir /data/demo/checkpoints \
    --output /data/demo/results
```
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Metrics to extract per benchmark (first is primary for plotting)
METRICS = {
    "pubmedqa": ["accuracy", "macro_f1"],
    "banking77": ["accuracy"],
    "tatqa": ["exact_match", "f1", "boxed_rate"],
    "nl2sql": ["edit_similarity", "exact_match"],
    "ifeval": ["inst_level_strict_acc", "prompt_level_strict_acc"],
    "safety": ["safe_rate"],
}


def load_json(path: Path) -> dict | None:
    return json.load(open(path)) if path.exists() else None


def extract_metrics(run_dir: Path) -> dict:
    """Extract key metrics from an eval run directory."""
    run_dir = Path(run_dir)
    row = {"run_id": run_dir.name, "run_dir": str(run_dir)}
    
    # Parse run_id: YYYYMMDD_HHMMSS_ModelName
    parts = run_dir.name.split("_", 2)
    if len(parts) >= 3:
        row["timestamp"] = f"{parts[0]}_{parts[1]}"
        row["model"] = parts[2]
    
    # Task evals
    for task in ["pubmedqa", "banking77", "tatqa", "nl2sql"]:
        task_dir = run_dir / task
        if not task_dir.exists():
            continue
        # Find task_result.json in subdirectory
        for subdir in task_dir.iterdir():
            result_file = subdir / "task_result.json" if subdir.is_dir() else None
            if result_file and result_file.exists():
                data = load_json(result_file)
                if data and "results" in data:
                    for bench, bench_metrics in data["results"].items():
                        for metric in METRICS.get(task, []):
                            if metric in bench_metrics:
                                row[f"{task}_{metric}"] = bench_metrics[metric]
                break
    
    # Control evals (IFEval + SimpleSafetyTests)
    control_dir = run_dir / "control"
    if control_dir.exists():
        for subdir in control_dir.iterdir():
            if not subdir.is_dir():
                continue
            result_file = subdir / "task_result.json"
            if not result_file.exists():
                continue
            data = load_json(result_file)
            if not data or "results" not in data:
                continue
            
            # IFEval
            if "leaderboard_ifeval" in data["results"]:
                ifeval = data["results"]["leaderboard_ifeval"]
                for metric in METRICS.get("ifeval", []):
                    key = f"{metric},none"
                    if key in ifeval:
                        row[f"ifeval_{metric}"] = ifeval[key]
            
            # SimpleSafetyTests
            if "simple_safety_tests" in data["results"]:
                safety = data["results"]["simple_safety_tests"]
                for metric in METRICS.get("safety", []):
                    if metric in safety:
                        row[f"safety_{metric}"] = safety[metric]
    
    return row


def load_experiment_meta(checkpoint_dir: Path, run_id: str) -> dict:
    """Try to load experiment metadata from checkpoint dir."""
    if not checkpoint_dir:
        return {}
    
    # Try to match run_id to checkpoint directory
    for ckpt_dir in checkpoint_dir.iterdir():
        if not ckpt_dir.is_dir():
            continue
        # Check if run_id contains the checkpoint name or vice versa
        if ckpt_dir.name in run_id or run_id.split("_")[-1] in ckpt_dir.name:
            meta_file = ckpt_dir / "experiment_meta.json"
            if meta_file.exists():
                meta = load_json(meta_file)
                # Prefix with meta_ to avoid column name collisions
                return {f"meta_{k}": v for k, v in meta.items() 
                        if v is not None and k not in ("run_name", "checkpoint_dir", 
                                                         "train_dataset", "val_dataset",
                                                         "training_config")}
    return {}


def collate(eval_dirs: list[Path], checkpoint_dir: Path = None) -> pd.DataFrame:
    """Collate all eval results into a DataFrame."""
    rows = []
    for eval_dir in eval_dirs:
        eval_dir = Path(eval_dir)
        if eval_dir.name == "collated" or not eval_dir.is_dir():
            continue
        row = extract_metrics(eval_dir)
        if checkpoint_dir:
            row.update(load_experiment_meta(checkpoint_dir, row.get("run_id", "")))
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Sort: baselines first (no task suffix), then by model name
    if "model" in df.columns:
        df["is_baseline"] = ~df["model"].str.contains("-", na=False)
        df = df.sort_values(["is_baseline", "model"], ascending=[False, True])
        df = df.drop(columns=["is_baseline"])
    return df


BENCHMARK_CONFIG = {
    "pubmedqa": {"title": "PubMedQA", "metrics": ["accuracy", "macro_f1"]},
    "banking77": {"title": "Banking77", "metrics": ["accuracy"]},
    "tatqa": {"title": "TAT-QA", "metrics": ["exact_match", "f1", "boxed_rate"]},
    "nl2sql": {"title": "NL2SQL", "metrics": ["edit_similarity", "exact_match"]},
    "ifeval": {"title": "IFEval", "metrics": ["inst_level_strict_acc"]},
    "safety": {"title": "Safety", "metrics": ["safe_rate"]},
}


def plot_comparison(df: pd.DataFrame, output_path: Path):
    """Generate multi-panel comparison chart (one panel per benchmark)."""
    if df.empty:
        print("No data to plot")
        return
    
    models = df["model"].tolist() if "model" in df.columns else df["run_id"].tolist()
    
    # Filter to benchmarks with data
    benchmarks = []
    for bench in BENCHMARK_CONFIG:
        for metric in BENCHMARK_CONFIG[bench]["metrics"]:
            col = f"{bench}_{metric}"
            if col in df.columns and df[col].notna().any():
                benchmarks.append(bench)
                break
    
    if not benchmarks:
        print("No benchmark data to plot")
        return
    
    # Create subplots
    n_cols = min(3, len(benchmarks))
    n_rows = (len(benchmarks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if len(benchmarks) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]
        config = BENCHMARK_CONFIG[bench]
        metrics = [m for m in config["metrics"] if f"{bench}_{m}" in df.columns]
        
        x = np.arange(len(models))
        width = 0.8 / len(metrics)
        colors = plt.colormaps["Set2"](np.linspace(0, 1, len(metrics)))
        
        for i, metric in enumerate(metrics):
            col = f"{bench}_{metric}"
            values = df[col].fillna(0).tolist()
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, 
                         label=metric.replace("_", " ").title(),
                         color=colors[i], edgecolor="white", linewidth=0.5)
            # Value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        
        ax.set_title(config["title"], fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        
        # Bold the baseline and task-matched model labels for visual comparison
        # TODO fix this -- won't be robust for ad hoc model names
        for tick_label, model_name in zip(ax.get_xticklabels(), models):
            is_baseline = "train" not in model_name.lower() and "ft" not in model_name.lower()
            is_task_match = bench in model_name.lower()
            if is_baseline or is_task_match:
                tick_label.set_fontweight("bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Hide unused subplots
    for idx in range(len(benchmarks), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collate eval results and plot")
    parser.add_argument("--eval-dirs", nargs="+", type=Path, required=True,
                        help="Eval result directories")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Checkpoint dir to find experiment metadata")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output directory (will create results.csv and comparison.png)")
    args = parser.parse_args()
    
    # Collate
    df = collate(args.eval_dirs, args.checkpoint_dir)
    
    if df.empty:
        print("No results found")
        return
    
    # Output
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path} ({len(df)} rows)")
    
    # Plot
    plot_path = args.output / "comparison.png"
    plot_comparison(df, plot_path)
    
    # Print summary (primary metrics only)
    print(f"\nResults summary:")
    primary_cols = ["model"] + [f"{t}_{m[0]}" for t, m in METRICS.items() 
                                 if f"{t}_{m[0]}" in df.columns]
    print(df[primary_cols].to_string(index=False))


if __name__ == "__main__":
    main()

