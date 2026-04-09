"""Format tool call evaluation scores as markdown tables and bar charts.

Reads all *.scores.json files from the output directory and produces:
1. A markdown table printed to stdout
2. Bar chart PNGs saved to the output directory

Usage:
    python format_scores.py
    python format_scores.py --output_dir output
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scores(output_dir: Path) -> dict:
    """Load all score files. Returns {model_name: scores_dict}."""
    scores = {}
    for path in sorted(output_dir.glob("*.scores.json")):
        name = path.stem.replace(".scores", "")
        with open(path) as f:
            scores[name] = json.loads(f.read())
    return scores


def print_markdown_table(scores: dict):
    """Print overall results as a markdown table."""
    print("## Overall Results\n")
    print("| Model | Total | Parse Fail | Name Acc | Args Acc | Exact Match |")
    print("|-------|------:|----------:|---------:|---------:|------------:|")
    for name, s in scores.items():
        o = s["overall"]
        print(
            f"| {name} | {o['total']} | {o['parse_failures']} "
            f"| {o['name_accuracy']:.1%} | {o['args_accuracy']:.1%} "
            f"| {o['exact_match']:.1%} |"
        )

    # Per-category tables
    categories = list(next(iter(scores.values()))["per_category"].keys())
    for cat in categories:
        print(f"\n### {cat.capitalize()}\n")
        print("| Model | Total | Parse Fail | Name Acc | Args Acc | Exact Match |")
        print("|-------|------:|----------:|---------:|---------:|------------:|")
        for name, s in scores.items():
            c = s["per_category"][cat]
            print(
                f"| {name} | {c['total']} | {c['parse_failures']} "
                f"| {c['name_accuracy']:.1%} | {c['args_accuracy']:.1%} "
                f"| {c['exact_match']:.1%} |"
            )


def plot_overall(scores: dict, output_dir: Path):
    """Bar chart of overall metrics across models."""
    models = list(scores.keys())
    metrics = ["name_accuracy", "args_accuracy", "exact_match"]
    labels = ["Name Acc", "Args Acc", "Exact Match"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [scores[m]["overall"][metric] for m in models]
        bars = ax.bar(x + i * width, values, width, label=label, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Accuracy")
    ax.set_title("Tool Call Accuracy — Overall")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "chart_overall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_per_category(scores: dict, output_dir: Path):
    """Bar chart of exact match broken down by category."""
    models = list(scores.keys())
    categories = list(next(iter(scores.values()))["per_category"].keys())
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    x = np.arange(len(models))
    width = 0.8 / len(categories)

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))
    for i, cat in enumerate(categories):
        values = [scores[m]["per_category"][cat]["exact_match"] for m in models]
        bars = ax.bar(
            x + i * width, values, width, label=cat, color=colors[i % len(colors)]
        )
        for bar, val in zip(bars, values):
            if val > 0.02:  # skip tiny labels
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_ylabel("Exact Match")
    ax.set_title("Tool Call Exact Match — Per Category")
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "chart_per_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_parse_failures(scores: dict, output_dir: Path):
    """Bar chart of parse failure rates."""
    models = list(scores.keys())
    fail_rates = [
        scores[m]["overall"]["parse_failures"] / scores[m]["overall"]["total"]
        for m in models
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    bars = ax.bar(models, fail_rates, color="#C44E52")
    for bar, val in zip(bars, fail_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Parse Failure Rate")
    ax.set_title("Tool Call Parse Failures")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "chart_parse_failures.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Format tool call eval scores")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/shanghong/oumi/tool_call_project/output",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    scores = load_scores(output_dir)
    if not scores:
        print(f"No *.scores.json files found in {output_dir}")
        return

    print(f"Found {len(scores)} models: {list(scores.keys())}\n")

    # Markdown tables
    print_markdown_table(scores)

    # Bar charts
    print()
    plot_overall(scores, output_dir)
    plot_per_category(scores, output_dir)
    plot_parse_failures(scores, output_dir)


if __name__ == "__main__":
    main()
