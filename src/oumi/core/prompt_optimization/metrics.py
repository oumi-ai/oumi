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

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Optional


def accuracy_metric(predictions: list[str], references: list[str]) -> float:
    """Calculate exact match accuracy.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        Accuracy score between 0 and 1.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have same length, "
            f"got {len(predictions)} and {len(references)}"
        )

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
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have same length, "
            f"got {len(predictions)} and {len(references)}"
        )

    if len(predictions) == 0:
        return 0.0

    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            f1_scores.append(1.0)
            continue

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
            continue

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def bleu_metric(predictions: list[str], references: list[str]) -> float:
    """Calculate BLEU score using sacrebleu if available.

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

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have same length, "
            f"got {len(predictions)} and {len(references)}"
        )

    if len(predictions) == 0:
        return 0.0

    # sacrebleu expects references as list of lists
    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
    return bleu.score / 100.0  # Convert to 0-1 range


def rouge_metric(predictions: list[str], references: list[str]) -> float:
    """Calculate ROUGE-L score using rouge-score if available.

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

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have same length, "
            f"got {len(predictions)} and {len(references)}"
        )

    if len(predictions) == 0:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)

    return sum(scores) / len(scores)


def load_custom_metric(metric_path: str) -> Callable[[list[str], list[str]], float]:
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

    if not path.suffix == ".py":
        raise ValueError(f"Metric file must be a Python file: {metric_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("custom_metric", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metric module from: {metric_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_metric"] = module
    spec.loader.exec_module(module)

    # Get the metric function
    if not hasattr(module, "metric_fn"):
        raise AttributeError(
            f"Metric module must define a 'metric_fn' function: {metric_path}"
        )

    metric_fn = getattr(module, "metric_fn")
    if not callable(metric_fn):
        raise TypeError(f"metric_fn must be callable: {metric_path}")

    return metric_fn  # type: ignore[return-value]


def embedding_similarity_metric(
    predictions: list[str],
    references: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """Calculate cosine similarity between embeddings.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.
        model_name: Name of the sentence transformer model to use.

    Returns:
        Average cosine similarity score between 0 and 1.
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding similarity metric. "
            "Install it with: pip install sentence-transformers"
        )

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have same length, "
            f"got {len(predictions)} and {len(references)}"
        )

    if len(predictions) == 0:
        return 0.0

    # Load model (cached after first use)
    model = SentenceTransformer(model_name)

    # Encode all texts
    pred_embeddings = model.encode(predictions, convert_to_numpy=True)
    ref_embeddings = model.encode(references, convert_to_numpy=True)

    # Calculate cosine similarities
    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        # Cosine similarity
        cos_sim = np.dot(pred_emb, ref_emb) / (
            np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb)
        )
        # Convert from [-1, 1] to [0, 1]
        normalized_sim = (cos_sim + 1) / 2
        similarities.append(normalized_sim)

    return sum(similarities) / len(similarities)


def bertscore_metric(predictions: list[str], references: list[str]) -> float:
    """Calculate BERTScore F1.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        Average BERTScore F1 between 0 and 1.
    """
    try:
        from bert_score import score  # type: ignore
    except ImportError:
        raise ImportError(
            "bert-score is required for BERTScore metric. "
            "Install it with: pip install bert-score"
        )

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have same length, "
            f"got {len(predictions)} and {len(references)}"
        )

    if len(predictions) == 0:
        return 0.0

    # Calculate BERTScore
    # Returns precision, recall, F1
    P, R, F1 = score(predictions, references, lang="en", verbose=False)

    # Return average F1 score
    return F1.mean().item()


def llm_judge_metric(
    predictions: list[str],
    references: list[str],
    judge_model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
) -> float:
    """Use an LLM as a judge to evaluate predictions.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.
        judge_model: Name of the model to use as judge.
        api_key: API key for the judge model (if needed).

    Returns:
        Average quality score between 0 and 1.

    Note:
        This metric requires an API key and will make API calls.
        It can be expensive for large datasets.
    """
    try:
        import openai  # type: ignore
    except ImportError:
        raise ImportError(
            "openai is required for LLM judge metric. "
            "Install it with: pip install openai"
        )

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have same length, "
            f"got {len(predictions)} and {len(references)}"
        )

    if len(predictions) == 0:
        return 0.0

    if api_key:
        openai.api_key = api_key

    scores = []
    for pred, ref in zip(predictions, references):
        prompt = f"""You are an objective judge evaluating the quality of a response.

Reference answer: {ref}

Candidate answer: {pred}

Rate the candidate answer on a scale from 0 to 10, where:
- 0-2: Completely wrong or irrelevant
- 3-4: Partially correct but missing key information
- 5-6: Mostly correct with minor errors
- 7-8: Correct with good quality
- 9-10: Excellent, comprehensive answer

Output ONLY a single number from 0 to 10, nothing else."""

        try:
            response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            # Extract score
            score_text = response.choices[0].message.content.strip()  # type: ignore[attr-defined]
            score = float(score_text)

            # Normalize to 0-1 range
            normalized_score = score / 10.0
            scores.append(normalized_score)

        except Exception as e:
            # If judgment fails, return 0
            from oumi.utils.logging import logger

            logger.warning(f"LLM judge failed for prediction: {e}")
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def get_metric_fn(
    metric_name: str, custom_metric_path: Optional[str] = None, **metric_kwargs
) -> Callable[[list[str], list[str]], float]:
    """Get a metric function by name.

    Args:
        metric_name: Name of the metric (accuracy, f1, bleu, rouge,
            embedding_similarity, bertscore, llm_judge, custom).
        custom_metric_path: Path to custom metric file (required for custom metric).
        **metric_kwargs: Additional keyword arguments for specific metrics
            (e.g., model_name for embedding_similarity, judge_model for llm_judge).

    Returns:
        The metric function.
    """
    if metric_name == "accuracy":
        return accuracy_metric
    elif metric_name == "f1":
        return f1_metric
    elif metric_name == "bleu":
        return bleu_metric
    elif metric_name == "rouge":
        return rouge_metric
    elif metric_name == "embedding_similarity":
        # Create a partial function with the model_name
        model_name = metric_kwargs.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )

        def wrapped_embedding_metric(preds: list[str], refs: list[str]) -> float:
            return embedding_similarity_metric(preds, refs, model_name=model_name)

        return wrapped_embedding_metric
    elif metric_name == "bertscore":
        return bertscore_metric
    elif metric_name == "llm_judge":
        # Create a partial function with judge model and API key
        judge_model = metric_kwargs.get("judge_model", "gpt-3.5-turbo")
        api_key = metric_kwargs.get("api_key")

        def wrapped_llm_judge(preds: list[str], refs: list[str]) -> float:
            return llm_judge_metric(
                preds, refs, judge_model=judge_model, api_key=api_key
            )

        return wrapped_llm_judge
    elif metric_name == "custom":
        if custom_metric_path is None:
            raise ValueError("custom_metric_path required for custom metric")
        return load_custom_metric(custom_metric_path)
    else:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Supported: accuracy, f1, bleu, rouge, embedding_similarity, "
            f"bertscore, llm_judge, custom"
        )
