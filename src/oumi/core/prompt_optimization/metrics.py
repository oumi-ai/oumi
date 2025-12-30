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

"""Metrics for prompt optimization evaluation."""

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oumi.utils.logging import logger

# Type alias for metric functions
MetricFn = Callable[[list[str], list[str]], float]


def accuracy_metric(predictions: list[str], references: list[str]) -> float:
    """Calculate exact match accuracy.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        Accuracy score between 0 and 1.
    """
    _validate_lengths(predictions, references)
    if len(predictions) == 0:
        return 0.0

    correct = sum(
        pred.strip().lower() == ref.strip().lower()
        for pred, ref in zip(predictions, references)
    )
    return correct / len(predictions)


def f1_metric(predictions: list[str], references: list[str]) -> float:
    """Calculate token-level F1 score.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        Average F1 score between 0 and 1.
    """
    _validate_lengths(predictions, references)
    if len(predictions) == 0:
        return 0.0

    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            f1_scores.append(1.0)
        elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
        else:
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * (precision * recall) / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


def bleu_metric(predictions: list[str], references: list[str]) -> float:
    """Calculate BLEU score using sacrebleu.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        BLEU score between 0 and 1.
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError(
            "sacrebleu is required for BLEU metric. "
            "Install it with: pip install sacrebleu"
        )

    _validate_lengths(predictions, references)
    if len(predictions) == 0:
        return 0.0

    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
    return bleu.score / 100.0


def rouge_metric(predictions: list[str], references: list[str]) -> float:
    """Calculate ROUGE-L score.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        ROUGE-L F1 score between 0 and 1.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for ROUGE metric. "
            "Install it with: pip install rouge-score"
        )

    _validate_lengths(predictions, references)
    if len(predictions) == 0:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


def bertscore_metric(predictions: list[str], references: list[str]) -> float:
    """Calculate BERTScore F1.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        Average BERTScore F1 between 0 and 1.
    """
    try:
        from bert_score import score
    except ImportError:
        raise ImportError(
            "bert-score is required for BERTScore metric. "
            "Install it with: pip install bert-score"
        )

    _validate_lengths(predictions, references)
    if len(predictions) == 0:
        return 0.0

    P, R, F1 = score(predictions, references, lang="en", verbose=False)
    return F1.mean().item()


def _validate_lengths(predictions: list[str], references: list[str]) -> None:
    """Validate that prediction and reference lists have the same length."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have same length, "
            f"got {len(predictions)} and {len(references)}"
        )


# =============================================================================
# Parameterized Metric Factories
# =============================================================================


def _create_embedding_similarity_metric(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> MetricFn:
    """Create an embedding similarity metric with specified model.

    Args:
        model_name: Name of the sentence transformer model to use.

    Returns:
        Metric function that computes embedding cosine similarity.
    """

    def metric(predictions: list[str], references: list[str]) -> float:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding similarity metric. "
                "Install it with: pip install sentence-transformers"
            )

        _validate_lengths(predictions, references)
        if len(predictions) == 0:
            return 0.0

        model = SentenceTransformer(model_name)
        pred_embeddings = model.encode(predictions, convert_to_numpy=True)
        ref_embeddings = model.encode(references, convert_to_numpy=True)

        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            cos_sim = np.dot(pred_emb, ref_emb) / (
                np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb)
            )
            similarities.append((cos_sim + 1) / 2)  # Convert [-1, 1] to [0, 1]

        return sum(similarities) / len(similarities)

    return metric


def _create_llm_judge_metric(
    judge_model: str = "gpt-3.5-turbo",
    api_key: str | None = None,
) -> MetricFn:
    """Create an LLM judge metric with specified model.

    Args:
        judge_model: Name of the model to use as judge.
        api_key: API key for the judge model.

    Returns:
        Metric function that uses LLM as judge.
    """

    def metric(predictions: list[str], references: list[str]) -> float:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for LLM judge metric. "
                "Install it with: pip install openai"
            )

        _validate_lengths(predictions, references)
        if len(predictions) == 0:
            return 0.0

        if api_key:
            openai.api_key = api_key

        scores = []
        for pred, ref in zip(predictions, references):
            prompt = (
                "You are an objective judge evaluating the quality of a response.\n\n"
                f"Reference answer: {ref}\n\n"
                f"Candidate answer: {pred}\n\n"
                "Rate the candidate answer on a scale from 0 to 10, where:\n"
                "- 0-2: Completely wrong or irrelevant\n"
                "- 3-4: Partially correct but missing key information\n"
                "- 5-6: Mostly correct with minor errors\n"
                "- 7-8: Correct with good quality\n"
                "- 9-10: Excellent, comprehensive answer\n\n"
                "Output ONLY a single number from 0 to 10, nothing else."
            )

            try:
                response = openai.ChatCompletion.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                score_text = response.choices[0].message.content.strip()
                scores.append(float(score_text) / 10.0)
            except Exception as e:
                logger.warning(f"LLM judge failed for prediction: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    return metric


def _load_custom_metric(metric_path: str) -> MetricFn:
    """Load a custom metric function from a Python file.

    The file should define a function named 'metric_fn' that takes
    predictions and references and returns a score between 0 and 1.

    Args:
        metric_path: Path to the Python file containing the metric function.

    Returns:
        The custom metric function.
    """
    path = Path(metric_path)
    if not path.exists():
        raise FileNotFoundError(f"Metric file not found: {metric_path}")

    if path.suffix != ".py":
        raise ValueError(f"Metric file must be a Python file: {metric_path}")

    spec = importlib.util.spec_from_file_location("custom_metric", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metric module from: {metric_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_metric"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "metric_fn"):
        raise AttributeError(
            f"Metric module must define a 'metric_fn' function: {metric_path}"
        )

    metric_fn = getattr(module, "metric_fn")
    if not callable(metric_fn):
        raise TypeError(f"metric_fn must be callable: {metric_path}")

    return metric_fn  # type: ignore[return-value]


@dataclass(frozen=True)
class _MetricSpec:
    """Internal specification for a metric type."""

    metric_fn: MetricFn | None = None
    """Direct metric function (for simple metrics)."""

    factory_fn: Callable[..., MetricFn] | None = None
    """Factory function for parameterized metrics."""

    requires_path: bool = False
    """Whether this metric requires a file path."""


_METRIC_REGISTRY: dict[str, _MetricSpec] = {
    "accuracy": _MetricSpec(metric_fn=accuracy_metric),
    "f1": _MetricSpec(metric_fn=f1_metric),
    "bleu": _MetricSpec(metric_fn=bleu_metric),
    "rouge": _MetricSpec(metric_fn=rouge_metric),
    "bertscore": _MetricSpec(metric_fn=bertscore_metric),
    "embedding_similarity": _MetricSpec(factory_fn=_create_embedding_similarity_metric),
    "llm_judge": _MetricSpec(factory_fn=_create_llm_judge_metric),
    "custom": _MetricSpec(factory_fn=_load_custom_metric, requires_path=True),
}


# =============================================================================
# Public API
# =============================================================================


def get_metric_fn(
    metric_name: str,
    custom_metric_path: str | None = None,
    **metric_kwargs: Any,
) -> MetricFn:
    """Get a metric function by name.

    Args:
        metric_name: Name of the metric. Supported values:
            - accuracy: Exact match accuracy
            - f1: Token-level F1 score
            - bleu: BLEU score (requires sacrebleu)
            - rouge: ROUGE-L score (requires rouge-score)
            - bertscore: BERTScore F1 (requires bert-score)
            - embedding_similarity: Cosine similarity of embeddings
            - llm_judge: LLM-as-a-judge evaluation
            - custom: Custom metric from file
        custom_metric_path: Path to custom metric file (required for 'custom').
        **metric_kwargs: Additional arguments for parameterized metrics:
            - model_name: For embedding_similarity (default: all-MiniLM-L6-v2)
            - judge_model: For llm_judge (default: gpt-3.5-turbo)
            - api_key: For llm_judge

    Returns:
        The metric function.

    Raises:
        ValueError: If metric_name is unknown or required path is missing.
    """
    if metric_name not in _METRIC_REGISTRY:
        available = ", ".join(_METRIC_REGISTRY.keys())
        raise ValueError(f"Unknown metric: {metric_name}. Supported: {available}")

    spec = _METRIC_REGISTRY[metric_name]

    # Handle custom metric path requirement
    if spec.requires_path:
        if custom_metric_path is None:
            raise ValueError(f"custom_metric_path required for '{metric_name}' metric")
        return spec.factory_fn(custom_metric_path)  # type: ignore[misc]

    # Simple metrics - return directly
    if spec.metric_fn is not None:
        return spec.metric_fn

    # Parameterized metrics - use factory
    if spec.factory_fn is not None:
        return spec.factory_fn(**metric_kwargs)

    raise ValueError(f"Invalid metric spec for '{metric_name}'")
