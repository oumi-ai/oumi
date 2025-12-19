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

"""Classification evaluation functions for Banking77 and PubMedQA tasks."""

import json
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
            break  # Only take user message for inference
    return Conversation(messages=messages)


def _normalize_label(label: str) -> str:
    """Normalize label for comparison."""
    return label.strip().lower()


def _compute_accuracy(predictions: list[str], ground_truths: list[str]) -> dict:
    """Compute classification accuracy metrics."""
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Predictions ({len(predictions)}) and ground truths "
            f"({len(ground_truths)}) must have same length"
        )

    total = len(predictions)
    correct = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_normalized = _normalize_label(pred)
        gt_normalized = _normalize_label(gt)

        # Check for exact match or if prediction contains the ground truth
        if pred_normalized == gt_normalized or gt_normalized in pred_normalized:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
        "num_incorrect": total - correct,
    }


@register_evaluation_function("enterprise_banking77")
def enterprise_banking77(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    test_data_path: str = "data/enterprise/banking77/test.jsonl",
) -> dict[str, Any]:
    """Evaluate Banking77 classification task.

    Args:
        task_params: Evaluation task parameters
        inference_engine: Inference engine for generating predictions
        test_data_path: Path to test JSONL file
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Dictionary with accuracy metrics
    """
    if not Path(test_data_path).exists():
        raise FileNotFoundError(
            f"Test data not found at {test_data_path}. "
            "Run scripts/enterprise/prepare_datasets.py first."
        )

    logger.info(f"Loading Banking77 test data from {test_data_path}")
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
    metrics = _compute_accuracy(predictions, ground_truths)

    logger.info(
        f"Banking77 results: accuracy={metrics['accuracy']:.4f} "
        f"({metrics['num_correct']}/{metrics['num_total']})"
    )

    return metrics


@register_evaluation_function("enterprise_pubmedqa")
def enterprise_pubmedqa(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    test_data_path: str = "data/enterprise/pubmedqa/test.jsonl",
) -> dict[str, Any]:
    """Evaluate PubMedQA classification task.

    Args:
        task_params: Evaluation task parameters
        inference_engine: Inference engine for generating predictions
        test_data_path: Path to test JSONL file
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Dictionary with accuracy metrics
    """
    if not Path(test_data_path).exists():
        raise FileNotFoundError(
            f"Test data not found at {test_data_path}. "
            "Run scripts/enterprise/prepare_datasets.py first."
        )

    logger.info(f"Loading PubMedQA test data from {test_data_path}")
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

    # Extract predictions - normalize to yes/no/maybe
    valid_labels = {"yes", "no", "maybe"}
    predictions = []
    for conv in output_conversations:
        response = conv.last_message()
        if response and isinstance(response.content, str):
            pred = response.content.strip().lower()
            # Try to extract yes/no/maybe from response
            for label in valid_labels:
                if label in pred:
                    pred = label
                    break
            predictions.append(pred)
        else:
            predictions.append("")

    # Compute metrics
    metrics = _compute_accuracy(predictions, ground_truths)

    # Add per-class breakdown
    class_counts = {label: {"correct": 0, "total": 0} for label in valid_labels}
    for pred, gt in zip(predictions, ground_truths):
        gt_normalized = _normalize_label(gt)
        if gt_normalized in class_counts:
            class_counts[gt_normalized]["total"] += 1
            if _normalize_label(pred) == gt_normalized:
                class_counts[gt_normalized]["correct"] += 1

    metrics["per_class"] = {
        label: {
            "accuracy": (
                counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            ),
            "correct": counts["correct"],
            "total": counts["total"],
        }
        for label, counts in class_counts.items()
    }

    logger.info(
        f"PubMedQA results: accuracy={metrics['accuracy']:.4f} "
        f"({metrics['num_correct']}/{metrics['num_total']})"
    )

    return metrics
