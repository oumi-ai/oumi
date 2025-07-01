"""Analyze command for dataset analysis and insights."""

import json
import re
from pathlib import Path
from typing import Optional

import typer
from langdetect import detect
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from sentence_transformers import SentenceTransformer

from oumi.cli import cli_utils

# Initialize console
console = Console()

# Load safety patterns
SAFETY_PATTERNS = {
    "curse_words": r"\b(fuck|shit|damn|bitch|ass)\b",  # Add more patterns as needed
    "hate_speech": r"\b(hate|kill|stupid|idiot)\b",  # Add more patterns as needed
}


def detect_language(text: str) -> str:
    """Detect the language of a text sample."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def calculate_length_metrics(text: str) -> dict:
    """Calculate various length metrics for a text sample."""
    return {
        "characters": len(text),
        "words": len(text.split()),
        "sentences": len(re.split(r"[.!?]+", text)),
    }


def check_safety(text: str) -> dict:
    """Check text for safety concerns."""
    results = {}
    for category, pattern in SAFETY_PATTERNS.items():
        matches = re.findall(pattern, text.lower())
        results[category] = len(matches)
    return results


def analyze(
    ctx: typer.Context,
    input_file: Path = typer.Argument(
        ..., help="Path to the input dataset file (JSONL format)"
    ),
    output_file: Optional[Path] = typer.Option(
        None, help="Path to save analysis results"
    ),
    batch_size: int = typer.Option(100, help="Batch size for processing"),
    use_gpu: bool = typer.Option(False, help="Use GPU for embeddings if available"),
):
    """Analyze dataset composition, language, length, and safety metrics."""
    # Parse extra arguments from context
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Parse batch_size from extra args
    for arg in extra_args:
        if arg.startswith("batch_size="):
            batch_size = int(arg.split("=")[1])
            break

    # Parse use_gpu from extra args
    for arg in extra_args:
        if arg.startswith("use_gpu="):
            use_gpu = arg.split("=")[1].lower() == "true"
            break

    # Load the dataset
    console.print(f"Loading dataset from {input_file}...")
    console.print(f"Using batch size: {batch_size}")
    data = []
    with open(input_file) as f:
        for line in f:
            data.append(json.loads(line))

    # Initialize results storage
    results = {
        "languages": {},
        "length_stats": {
            "characters": {"min": float("inf"), "max": 0, "total": 0},
            "words": {"min": float("inf"), "max": 0, "total": 0},
            "sentences": {"min": float("inf"), "max": 0, "total": 0},
        },
        "safety_issues": {
            "curse_words": 0,
            "hate_speech": 0,
        },
        "samples": [],
    }

    # Load embedding model
    console.print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Check GPU availability and usage
    import torch

    if use_gpu and torch.cuda.is_available():
        console.print("Using GPU for embeddings")
        model = model.to("cuda")
    elif use_gpu:
        console.print(
            "[yellow]Warning: GPU requested but not available. "
            "Using CPU instead.[/yellow]"
        )
    else:
        console.print("Using CPU for embeddings")

    # Process samples
    with Progress() as progress:
        task = progress.add_task("Analyzing samples...", total=len(data))

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            for sample in batch:
                text = sample.get("text", "")

                # Language detection
                lang = detect_language(text)
                results["languages"][lang] = results["languages"].get(lang, 0) + 1

                # Length metrics
                length_metrics = calculate_length_metrics(text)
                for metric, value in length_metrics.items():
                    results["length_stats"][metric]["min"] = min(
                        results["length_stats"][metric]["min"], value
                    )
                    results["length_stats"][metric]["max"] = max(
                        results["length_stats"][metric]["max"], value
                    )
                    results["length_stats"][metric]["total"] += value

                # Safety checks
                safety_results = check_safety(text)
                for category, count in safety_results.items():
                    results["safety_issues"][category] += count

                # Store sample analysis
                results["samples"].append(
                    {
                        "text": text,
                        "language": lang,
                        "length_metrics": length_metrics,
                        "safety_issues": safety_results,
                    }
                )

            progress.update(task, advance=len(batch))

    # Calculate averages
    n_samples = len(data)
    for metric in results["length_stats"]:
        results["length_stats"][metric]["avg"] = (
            results["length_stats"][metric]["total"] / n_samples
        )

    # Display results
    display_results(results)

    # Save results if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\nResults saved to {output_file}")


def display_results(results: dict):
    """Display analysis results in a formatted way."""
    # Language distribution
    console.print("\n[bold]Language Distribution:[/bold]")
    lang_table = Table(show_header=True)
    lang_table.add_column("Language")
    lang_table.add_column("Count")
    lang_table.add_column("Percentage")

    total_samples = sum(results["languages"].values())
    for lang, count in sorted(
        results["languages"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / total_samples) * 100
        lang_table.add_row(lang, str(count), f"{percentage:.1f}%")
    console.print(lang_table)

    # Length statistics
    console.print("\n[bold]Length Statistics:[/bold]")
    length_table = Table(show_header=True)
    length_table.add_column("Metric")
    length_table.add_column("Min")
    length_table.add_column("Max")
    length_table.add_column("Average")

    for metric, stats in results["length_stats"].items():
        length_table.add_row(
            metric.capitalize(),
            str(stats["min"]),
            str(stats["max"]),
            f"{stats['avg']:.1f}",
        )
    console.print(length_table)

    # Safety issues
    console.print("\n[bold]Safety Issues:[/bold]")
    safety_table = Table(show_header=True)
    safety_table.add_column("Category")
    safety_table.add_column("Count")

    for category, count in results["safety_issues"].items():
        safety_table.add_row(category.replace("_", " ").title(), str(count))
    console.print(safety_table)
