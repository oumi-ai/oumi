#!/usr/bin/env python3
"""
Plot IFEval metrics from control_tatqa output directory.
Plots prompt level strict accuracy and instruction level strict accuracy.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define the output directory
OUTPUT_DIR = Path("/data/shanghong/oumi/gold/output/control_tatqa")

# Define the ordering of models (based on the reference image)
MODEL_ORDER = [
    "Llama-3.1-8B-Instruct",
    "tatqa_llama_lambda0.5_1epoch_inversekl_lora",
    "tatqa_llama3.1_8b_sft",  # New model added after lambda0.5
    "tinker_llama3.1_8b_ckpt6200",
    "Llama-3.3-70B-Instruct",
    # "Qwen2.5-1.5B-Instruct",
    # "qwen2.5_1.5b_lambda0.5",
    # "qwen2.5_1.5b.sft_completions_only",
    # "Qwen3-4B-Instruct",
]

# Display names for models (clean up underscores and make more readable)
DISPLAY_NAMES = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "tatqa_llama_lambda0.5_1epoch_inversekl_lora": "GOLD inverse KL w/ lora",
    "tatqa_llama3.1_8b_sft": "SFT",
    "tinker_llama3.1_8b_ckpt6200": "Tinker",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "qwen2.5_1.5b_lambda0.5": "qwen2.5_1.5b_lambda0.5",
    "qwen2.5_1.5b.sft_completions_only": "qwen2.5_1.5b.sft_completions_only",
    "Qwen3-4B-Instruct": "Qwen3-4B-Instruct",
}


def collect_metrics():
    """Collect metrics from all task_result.json files."""
    metrics = {}

    # Iterate through all model directories
    for model_dir in OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Find the task_result.json file
        task_result_files = list(model_dir.glob("*/task_result.json"))

        if not task_result_files:
            print(f"Warning: No task_result.json found for {model_name}")
            continue

        # Use the first (should be only) task_result.json file
        task_result_file = task_result_files[0]

        with open(task_result_file, 'r') as f:
            data = json.load(f)

        # Extract metrics
        results = data.get("results", {}).get("leaderboard_ifeval", {})

        metrics[model_name] = {
            "prompt_level_strict_acc": results.get("prompt_level_strict_acc,none", 0) * 100,
            "inst_level_strict_acc": results.get("inst_level_strict_acc,none", 0) * 100,
        }

    return metrics


def plot_metrics(metrics):
    """Plot the metrics in a grouped bar chart."""
    # Filter and order models based on MODEL_ORDER
    ordered_models = [m for m in MODEL_ORDER if m in metrics]

    # Get display names and metric values
    display_names = [DISPLAY_NAMES.get(m, m) for m in ordered_models]
    prompt_acc = [metrics[m]["prompt_level_strict_acc"] for m in ordered_models]
    inst_acc = [metrics[m]["inst_level_strict_acc"] for m in ordered_models]

    # Create figure with grouped bars
    fig, ax = plt.subplots(figsize=(16, 8))

    x_pos = np.arange(len(ordered_models))
    bar_width = 0.35

    # Create grouped bars
    bars1 = ax.bar(x_pos - bar_width/2, prompt_acc, bar_width,
                   label='Prompt Level Strict Acc', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + bar_width/2, inst_acc, bar_width,
                   label='Instruction Level Strict Acc', color='coral', alpha=0.8)

    # Customize the plot
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('IFEval Llama', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, fontsize=14, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=14)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=14)

    plt.tight_layout()

    # Save the figure
    output_path = OUTPUT_DIR / "tatqa_ifeval_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()


def print_metrics_table(metrics):
    """Print metrics in a formatted table."""
    ordered_models = [m for m in MODEL_ORDER if m in metrics]

    print("\n" + "="*80)
    print("IFEval Metrics Summary")
    print("="*80)
    print(f"{'Model':<40} {'Prompt Strict Acc':<20} {'Inst Strict Acc':<20}")
    print("-"*80)

    for model in ordered_models:
        display_name = DISPLAY_NAMES.get(model, model)
        prompt_acc = metrics[model]["prompt_level_strict_acc"]
        inst_acc = metrics[model]["inst_level_strict_acc"]
        print(f"{display_name:<40} {prompt_acc:>18.2f}% {inst_acc:>18.2f}%")

    print("="*80)


def main():
    print("Collecting metrics from:", OUTPUT_DIR)
    metrics = collect_metrics()

    print(f"\nFound metrics for {len(metrics)} models:")
    for model in sorted(metrics.keys()):
        print(f"  - {model}")

    # Print table
    print_metrics_table(metrics)

    # Create plots
    plot_metrics(metrics)


if __name__ == "__main__":
    main()
