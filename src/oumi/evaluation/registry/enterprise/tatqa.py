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

"""TAT-QA evaluation function for tabular question answering."""

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


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove extra whitespace
    answer = " ".join(answer.split())
    # Convert to lowercase
    answer = answer.lower()
    # Remove punctuation at the end
    answer = re.sub(r"[.,;:!?]+$", "", answer)
    # Remove common prefixes like "the answer is"
    answer = re.sub(r"^(the answer is|answer:|the result is)\s*", "", answer)
    return answer.strip()


def _compute_exact_match(pred: str, gt: str) -> bool:
    """Check if prediction exactly matches ground truth."""
    return _normalize_answer(pred) == _normalize_answer(gt)


def _compute_f1(pred: str, gt: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = set(_normalize_answer(pred).split())
    gt_tokens = set(_normalize_answer(gt).split())

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    overlap = pred_tokens & gt_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@register_evaluation_function("enterprise_tatqa")
def enterprise_tatqa(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    test_data_path: str = "data/enterprise/tatqa/test.jsonl",
) -> dict[str, Any]:
    """Evaluate TAT-QA tabular question answering task.

    Computes both exact match accuracy and token-level F1 score.

    Args:
        task_params: Evaluation task parameters
        inference_engine: Inference engine for generating predictions
        test_data_path: Path to test JSONL file
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Dictionary with exact match and F1 metrics
    """

    if not Path(test_data_path).exists():
        raise FileNotFoundError(
            f"Test data not found at {test_data_path}. "
            "Run scripts/enterprise/prepare_datasets.py first."
        )

    logger.info(f"Loading TAT-QA test data from {test_data_path}")
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
    exact_matches = 0
    f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        if _compute_exact_match(pred, gt):
            exact_matches += 1
        f1_scores.append(_compute_f1(pred, gt))

    total = len(predictions)
    exact_match_accuracy = exact_matches / total if total > 0 else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    metrics = {
        "exact_match": exact_match_accuracy,
        "f1": avg_f1,
        "num_exact_match": exact_matches,
        "num_total": total,
    }

    logger.info(
        f"TAT-QA results: EM={metrics['exact_match']:.4f}, F1={metrics['f1']:.4f} "
        f"(EM: {metrics['num_exact_match']}/{metrics['num_total']})"
    )

    return metrics
