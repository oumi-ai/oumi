"""Curate command for dataset manipulation and filtering."""

import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from sentence_transformers import SentenceTransformer, util

# Initialize console
console = Console()


def load_analysis(analysis_file: Path) -> dict[str, Any]:
    """Load analysis results from file."""
    with open(analysis_file) as f:
        return json.load(f)


def filter_by_language(samples: list[dict], target_language: str) -> list[dict]:
    """Filter samples by language."""
    return [s for s in samples if s["language"] == target_language]


def filter_by_length(
    samples: list[dict], min_length: int, max_length: int, metric: str = "words"
) -> list[dict]:
    """Filter samples by length metrics."""
    return [s for s in samples if min_length <= s["length_stats"][metric] <= max_length]


def filter_by_safety(samples: list[dict], max_issues: int = 0) -> list[dict]:
    """Filter samples by safety issues."""
    return [s for s in samples if len(s.get("safety_issues", [])) <= max_issues]


def find_similar_samples(
    samples: list[dict],
    query: str,
    model: SentenceTransformer,
    top_k: int = 5,
    threshold: float = 0.7,
) -> list[dict]:
    """Find samples similar to the query text."""
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Encode all samples
    texts = [s["text"] for s in samples]
    embeddings = model.encode(texts, convert_to_tensor=True)
    # Calculate similarities
    import torch

    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = torch.tensor(query_embedding)
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    # Get top-k similar samples
    top_indices = similarities.argsort(descending=True)[:top_k]
    similar_samples = []
    for idx in top_indices:
        similarity = similarities[idx].item()
        if similarity >= threshold:
            similar_samples.append({"sample": samples[idx], "similarity": similarity})
    return similar_samples


def curate(
    analysis_file: Path = typer.Argument(..., help="Path to the analysis results file"),
    output_file: Path = typer.Argument(..., help="Path to save curated dataset"),
    language: Optional[str] = typer.Option(
        None, help="Filter by language code (e.g., 'en' for English)"
    ),
    min_length: Optional[int] = typer.Option(None, help="Minimum length (in words)"),
    max_length: Optional[int] = typer.Option(None, help="Maximum length (in words)"),
    max_safety_issues: int = typer.Option(
        0, help="Maximum number of safety issues allowed"
    ),
    query: Optional[str] = typer.Option(None, help="Find samples similar to this text"),
    similarity_threshold: float = typer.Option(
        0.7, help="Similarity threshold for query matching"
    ),
    use_gpu: bool = typer.Option(False, help="Use GPU for embeddings if available"),
):
    """Curate and manipulate datasets based on analysis results."""
    # Load analysis results
    console.print(f"Loading analysis from {analysis_file}...")
    analysis = load_analysis(analysis_file)
    samples = analysis["samples"]

    # Apply filters
    if language:
        console.print(f"Filtering by language: {language}")
        samples = filter_by_language(samples, language)

    if min_length is not None or max_length is not None:
        min_len = min_length or 0
        max_len = max_length if max_length is not None else 1_000_000_000
        console.print(f"Filtering by length: {min_len} to {max_len} words")
        samples = filter_by_length(samples, min_len, max_len)

    if max_safety_issues >= 0:
        console.print(f"Filtering by safety issues (max: {max_safety_issues})")
        samples = filter_by_safety(samples, max_safety_issues)

    # Find similar samples if query provided
    if query:
        console.print("Loading embedding model for similarity search...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        if use_gpu:
            model = model.to("cuda")

        console.print(f"Finding samples similar to: {query}")
        similar_samples = find_similar_samples(
            samples, query, model, threshold=similarity_threshold
        )

        # Display similar samples
        console.print("\n[bold]Similar Samples:[/bold]")
        for i, result in enumerate(similar_samples, 1):
            console.print(f"\n{i}. Similarity: {result['similarity']:.2f}")
            console.print(f"Text: {result['sample']['text'][:200]}...")

        # Use similar samples as the final dataset
        samples = [result["sample"] for result in similar_samples]

    # Save curated dataset
    console.print(f"\nSaving {len(samples)} samples to {output_file}")
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    console.print("\n[bold green]Curation complete![/bold green]")
