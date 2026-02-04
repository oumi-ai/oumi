import json
import re
from pathlib import Path
from typing import Any

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger
import argparse

def _load_test_data(test_data_path: str) -> list[dict]:
    """Load test data from JSONL file."""
    data = []
    with open(test_data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def _extract_ground_truth(conversation: dict) -> str:
    """Extract ground truth from the assistant message in the conversation."""
    return conversation["metadata"]["ground_truth"]


def _get_output_conversation(conversation: dict) -> Conversation:
    """Create input conversation (user message only) for inference."""
    messages = []
    for msg in conversation.get("messages", []):
        if msg.get("role") == "user":
            messages.append(Message(role=Role.USER, content=msg.get("content", "")))
            break
    return Conversation(messages=messages)


def _extract_sql(response: str) -> str:
    """Extract SQL from response, handling markdown code blocks.

    Looks for <ans blocks and extracts the content.
    Falls back to the full response if no markdown block found.
    """
    ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    matches = ANSWER_RE.findall(response or "")
    if not matches:
        return ""
    return matches[-1]


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    # Remove any remaining markdown artifacts
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_data_path", type=str, required=True, help="Path to test data JSONL file")
    ap.add_argument("--out", type=str, default=None, help="Write augmented JSONL to this path")
    ap.add_argument("--summary-out", type=str, default=None, help="Write dataset-level metrics JSON to this path")
    args = ap.parse_args()

    logger.info(f"Loading NL2SQL test data from {args.test_data_path}")
    test_data = _load_test_data(args.test_data_path)

    # Extract ground truths and create input conversations
    ground_truths = [_extract_ground_truth(conv) for conv in test_data]


    # Extract predictions (raw and extracted SQL)
    raw_predictions = []
    extracted_predictions = []
    for item in test_data:
        response = item["messages"][-1]["content"]
        raw_predictions.append(response)
        extracted_predictions.append(_extract_sql(response))

    # Compute metrics on extracted SQL
    edit_distances = []
    exact_matches = []

    for extracted, gt in zip(extracted_predictions, ground_truths):
        edit_dist = _compute_normalized_edit_distance(extracted, gt)
        edit_distances.append(edit_dist)

        if _compute_exact_match(extracted, gt):
            exact_matches.append(1)
        else:
            exact_matches.append(0)

    total = len(extracted_predictions)
    avg_edit_distance = (
        sum(edit_distances) / len(edit_distances) if edit_distances else 0.0
    )
    # Edit similarity is 1 - edit_distance (higher is better)
    avg_edit_similarity = 1.0 - avg_edit_distance
    exact_match_accuracy = sum(exact_matches) / total if total > 0 else 0.0

    # Mean response length (raw, before any extraction/normalization)
    mean_response_chars = (
        sum(len(r) for r in raw_predictions) / len(raw_predictions)
        if raw_predictions else 0.0
    )

    metrics = {
        "edit_similarity": avg_edit_similarity,
        "edit_distance": avg_edit_distance,
        "exact_match": exact_match_accuracy,
        "mean_response_chars": mean_response_chars,
        "num_exact_match": exact_matches,
        "num_total": total,
    }

    # Add predictions to metrics (will be saved separately by evaluator)
    metrics["_predictions"] = []
    for item, raw, extracted, edit_distance, exact_match in zip(
        test_data, raw_predictions, extracted_predictions, edit_distances, exact_matches
    ):
        new_item = item.copy()
        new_item["metadata"]["extracted_sql"] = extracted
        new_item["metadata"]["edit_distance"] = edit_distance
        new_item["metadata"]["exact_match"] = exact_match
        metrics["_predictions"].append(new_item)

    logger.info(
        f"NL2SQL results: EditSim={metrics['edit_similarity']:.4f}, "
        f"EM={metrics['exact_match']:.4f} "
        f"(EM: {metrics['num_exact_match']}/{metrics['num_total']})"
    )

    if args.out:
        with open(args.out, "w") as f:
            for item in metrics["_predictions"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if args.summary_out:
        with open(args.summary_out, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()