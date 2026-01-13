"""WiP -- initial draft of eval results plotting script from Opus

Visualize evaluation results across models and benchmarks.

Creates a multi-panel plot with one panel per benchmark, showing
performance metrics for each model.

Usage:

```sh
python scripts/enterprise/plot_eval_results.py \
  --results-json $BASELINE_EVALS_DEST/collated/results.json \
  --output $BASELINE_EVALS_DEST/collated/results-plot.png
```
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Define metrics to plot for each benchmark
# Each benchmark can have a response_length_key for the paired response length panel
BENCHMARK_METRICS = {
    "banking77": {
        "metrics": ["accuracy"],
        "title": "Banking77 (77-way Classification)",
        "ylabel": "Accuracy",
        "response_length_key": "banking77_mean_response_chars",
    },
    "pubmedqa": {
        "metrics": ["accuracy", "macro_f1"],
        "title": "PubMedQA (Yes/No/Maybe)",
        "ylabel": "Score",
        "response_length_key": "pubmedqa_mean_response_chars",
    },
    "tatqa": {
        "metrics": ["exact_match", "f1", "boxed_rate"],
        "title": "TAT-QA (Tabular QA)",
        "ylabel": "Score",
        "response_length_key": "tatqa_mean_response_chars",
    },
    "nl2sql": {
        "metrics": ["edit_similarity", "exact_match"],
        "title": "NL2SQL",
        "ylabel": "Score",
        "response_length_key": "nl2sql_mean_response_chars",
    },
    "safety": {
        "metrics": ["safe_rate"],
        "title": "Simple Safety Tests",
        "ylabel": "Safe Rate",
        "response_length_key": "safety_mean_response_chars",
    },
    "ifeval": {
        "metrics": ["prompt_level_strict_acc", "inst_level_strict_acc"],
        "title": "IFEval (Instruction Following)",
        "ylabel": "Accuracy",
        # No response length for ifeval (uses lm-harness, not custom eval)
    },
}


def load_results(json_path: Path) -> list[dict]:
    """Load results from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def sort_models_base_first(results: list[dict]) -> list[dict]:
    """Sort results so base models come before fine-tuned models.
    
    Base models are identified by NOT having common fine-tuning suffixes.
    """
    def is_base_model(row: dict) -> bool:
        """Check if a model is a base model (not fine-tuned)."""
        model_name = row.get("model_short", row.get("model_name", ""))
        # Common patterns for fine-tuned models
        ft_patterns = ["-tatqa", "-pubmedqa", "-banking", "-nl2sql", "-ft", "-sft", "-finetune"]
        return not any(pattern in model_name.lower() for pattern in ft_patterns)
    
    # Separate base and fine-tuned models
    base_models = sorted(
        [r for r in results if is_base_model(r)],
        key=lambda r: r.get("model_short", ""),
    )
    ft_models = sorted(
        [r for r in results if not is_base_model(r)],
        key=lambda r: r.get("model_short", ""),
    )
    
    # Return base models first, then fine-tuned
    return base_models + ft_models


def extract_benchmark_data(results: list[dict], benchmark: str) -> dict:
    """Extract data for a specific benchmark from results."""
    config = BENCHMARK_METRICS[benchmark]
    metrics = config["metrics"]
    
    data = {
        "models": [],
        "metrics": {m: [] for m in metrics},
        "response_length": [],
    }
    
    for row in results:
        model_name = row.get("model_short", row.get("model_name", "Unknown"))
        data["models"].append(model_name)
        
        for metric in metrics:
            # Handle different key patterns
            if benchmark == "ifeval":
                key = f"control_ifeval_{metric}"
            else:
                key = f"{benchmark}_{metric}"
            
            value = row.get(key)
            if value is not None:
                data["metrics"][metric].append(value)
            else:
                data["metrics"][metric].append(0)  # Missing data
        
        # Extract response length if configured
        resp_len_key = config.get("response_length_key")
        if resp_len_key:
            data["response_length"].append(row.get(resp_len_key, 0))
        else:
            data["response_length"].append(0)
    
    return data


def plot_benchmark(ax, data: dict, config: dict, benchmark: str):
    """Plot a single benchmark panel."""
    models = data["models"]
    metrics = list(data["metrics"].keys())
    resp_lengths = data.get("response_length", [])
    
    if not models:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(config["title"])
        return
    
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    
    colors = plt.colormaps["Set2"](np.linspace(0, 1, len(metrics)))
    
    for i, (metric, values) in enumerate(data["metrics"].items()):
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace("_", " ").title(), 
                      color=colors[i], edgecolor="white", linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f"{val:.2f}",
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha="center", va="bottom", fontsize=8)
    
    ax.set_ylabel(config["ylabel"])
    ax.set_title(config["title"], fontweight="bold")
    ax.set_xticks(x)
    
    # Create x-axis labels with response length annotations
    labels = []
    for i, model in enumerate(models):
        if resp_lengths and i < len(resp_lengths) and resp_lengths[i] > 0:
            labels.append(f"{model}\n({int(resp_lengths[i])} mean chars)")
        else:
            labels.append(model)
    
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_visualization(results: list[dict], output_path: Path):
    """Create the full visualization."""
    # Filter to benchmarks that have data
    benchmarks_with_data = []
    for benchmark in BENCHMARK_METRICS:
        data = extract_benchmark_data(results, benchmark)
        if any(any(v > 0 for v in values) for values in data["metrics"].values()):
            benchmarks_with_data.append(benchmark)
    
    if not benchmarks_with_data:
        print("No benchmark data found!")
        return
    
    # Create figure with subplots
    n_benchmarks = len(benchmarks_with_data)
    n_cols = min(3, n_benchmarks)
    n_rows = (n_benchmarks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Handle single subplot case
    if n_benchmarks == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each benchmark
    for i, benchmark in enumerate(benchmarks_with_data):
        data = extract_benchmark_data(results, benchmark)
        config = BENCHMARK_METRICS[benchmark]
        plot_benchmark(axes[i], data, config, benchmark)
    
    # Hide unused subplots
    for i in range(n_benchmarks, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    model_names = [r.get("model_short", "?") for r in results]
    fig.suptitle(f"Evaluation Results: {', '.join(model_names)}", 
                 fontsize=14, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results across models and benchmarks"
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Path to collated results.json file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.png"),
        help="Output image path (default: eval_results.png)",
    )
    args = parser.parse_args()
    
    if not args.results_json.exists():
        print(f"Error: {args.results_json} not found")
        return
    
    results = load_results(args.results_json)
    print(f"Loaded {len(results)} run(s)")
    
    # Sort so base models come first
    results = sort_models_base_first(results)
    
    create_visualization(results, args.output)


if __name__ == "__main__":
    main()

