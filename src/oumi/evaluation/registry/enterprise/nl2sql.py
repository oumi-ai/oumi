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

"""NL2SQL evaluation function for text-to-SQL generation."""

import json
import re
from pathlib import Path
from typing import Any

from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger


def _load_test_data(test_data_path: str) -> list[dict]:
    """Load test data from JSONL file."""
    data = []
    with open(test_data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _extract_ground_truth(conversation: dict) -> str:
    """Extract ground truth from the assistant message in the conversation."""
    messages = conversation.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return ""


def _create_input_conversation(conversation: dict) -> Conversation:
    """Create input conversation (user message only) for inference."""
    messages = []
    for msg in conversation.get("messages", []):
        if msg.get("role") == "user":
            messages.append(Message(role=Role.USER, content=msg.get("content", "")))
            break
    return Conversation(messages=messages)


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    # Remove markdown code blocks if present
    sql = re.sub(r"```sql\s*", "", sql)
    sql = re.sub(r"```\s*", "", sql)

    # Convert to lowercase
    sql = sql.lower()

    # Normalize whitespace
    sql = " ".join(sql.split())

    # Remove trailing semicolon
    sql = sql.rstrip(";")

    # Normalize common SQL patterns
    sql = re.sub(r"\s*,\s*", ", ", sql)
    sql = re.sub(r"\s*=\s*", " = ", sql)
    sql = re.sub(r"\s*<>\s*", " <> ", sql)
    sql = re.sub(r"\s*!=\s*", " != ", sql)
    sql = re.sub(r"\s*>=\s*", " >= ", sql)
    sql = re.sub(r"\s*<=\s*", " <= ", sql)
    sql = re.sub(r"\s*>\s*", " > ", sql)
    sql = re.sub(r"\s*<\s*", " < ", sql)

    return sql.strip()


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _compute_normalized_edit_distance(pred: str, gt: str) -> float:
    """Compute normalized edit distance (0 = identical, 1 = completely different)."""
    pred_normalized = _normalize_sql(pred)
    gt_normalized = _normalize_sql(gt)

    if pred_normalized == gt_normalized:
        return 0.0

    distance = _levenshtein_distance(pred_normalized, gt_normalized)
    max_len = max(len(pred_normalized), len(gt_normalized))

    if max_len == 0:
        return 0.0

    return distance / max_len


def _compute_exact_match(pred: str, gt: str) -> bool:
    """Check if normalized SQL matches exactly."""
    return _normalize_sql(pred) == _normalize_sql(gt)


@register_evaluation_function("enterprise_nl2sql")
def enterprise_nl2sql(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
) -> dict[str, Any]:
    """Evaluate NL2SQL text-to-SQL generation task.

    Computes edit distance and exact match metrics for SQL generation.

    Args:
        task_params: Evaluation task parameters. Expected eval_kwargs:
            - test_data_path: Path to test JSONL file
        inference_engine: Inference engine for generating predictions

    Returns:
        Dictionary with edit distance and exact match metrics
    """
    test_data_path = task_params.eval_kwargs.get(
        "test_data_path", "data/enterprise/nl2sql/test.jsonl"
    )

    if not Path(test_data_path).exists():
        raise FileNotFoundError(
            f"Test data not found at {test_data_path}. "
            "Run scripts/enterprise/prepare_datasets.py first."
        )

    logger.info(f"Loading NL2SQL test data from {test_data_path}")
    test_data = _load_test_data(test_data_path)

    # Apply sample limit if specified
    num_samples = task_params.num_samples
    if num_samples is not None and num_samples > 0:
        test_data = test_data[:num_samples]

    logger.info(f"Evaluating on {len(test_data)} samples")

    # Extract ground truths and create input conversations
    ground_truths = [_extract_ground_truth(conv) for conv in test_data]
    input_conversations = [_create_input_conversation(conv) for conv in test_data]

    # Run inference
    logger.info("Running inference...")
    output_conversations = inference_engine.infer(input_conversations)

    # Extract predictions
    predictions = []
    for conv in output_conversations:
        response = conv.last_message()
        if response and isinstance(response.content, str):
            predictions.append(response.content)
        else:
            predictions.append("")

    # Compute metrics
    edit_distances = []
    exact_matches = 0

    for pred, gt in zip(predictions, ground_truths):
        edit_dist = _compute_normalized_edit_distance(pred, gt)
        edit_distances.append(edit_dist)

        if _compute_exact_match(pred, gt):
            exact_matches += 1

    total = len(predictions)
    avg_edit_distance = (
        sum(edit_distances) / len(edit_distances) if edit_distances else 0.0
    )
    # Edit similarity is 1 - edit_distance (higher is better)
    avg_edit_similarity = 1.0 - avg_edit_distance
    exact_match_accuracy = exact_matches / total if total > 0 else 0.0

    metrics = {
        "edit_similarity": avg_edit_similarity,
        "edit_distance": avg_edit_distance,
        "exact_match": exact_match_accuracy,
        "num_exact_match": exact_matches,
        "num_total": total,
    }

    logger.info(
        f"NL2SQL results: EditSim={metrics['edit_similarity']:.4f}, "
        f"EM={metrics['exact_match']:.4f} "
        f"(EM: {metrics['num_exact_match']}/{metrics['num_total']})"
    )

    return metrics
