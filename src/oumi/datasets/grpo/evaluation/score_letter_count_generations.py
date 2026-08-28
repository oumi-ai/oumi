"""Offline scorer for letter-counting generations saved as JSONL conversations.

Unlike the `count_letters` evaluation function (see
`oumi.evaluation.registry.count_letters_task`), this script does not run
inference. It grades generations that were already saved to disk, e.g. the
`generations_trained.jsonl` file produced alongside a GRPO run, so a base and a
trained checkpoint can be compared without another inference pass.

Each line is expected to be a conversation with the model's answer as the last
assistant message, and the target count in `metadata.letter_count_integer`.

Usage:
  python src/oumi/datasets/grpo/evaluation/score_letter_count_generations.py \
      outputs/gemma4_e2b_grpo/generations_trained.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

# `_extract_prediction` is reused (rather than re-implementing the `\boxed{...}`
# regex here) so this script parses answers exactly like the GRPO reward does.
from oumi.datasets.grpo.rewards.count_letters_rewards import (
    _extract_prediction,
    compute_letter_count_reward,
)

_TARGET_KEY = "letter_count_integer"


def _last_assistant_content(conversation: dict[str, Any]) -> str | None:
    """Returns the content of the last assistant message, or None if there's none."""
    for message in reversed(conversation.get("messages", [])):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        # Ignore multimodal responses; this task is text-only.
        return content if isinstance(content, str) else None
    return None


def score_file(path: Path) -> dict[str, Any]:
    """Scores a single JSONL file of letter-counting conversations."""
    total = 0  # All examples.
    count = 0  # The number of examples with correct answers extracted.
    valid_count = 0  # The number of examples with valid answers extracted.
    reward_sum = 0.0
    errors: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            conversation = json.loads(line)
            metadata = conversation.get("metadata") or {}
            if _TARGET_KEY not in metadata:
                raise ValueError(
                    f"{path}:{line_number} is missing `metadata.{_TARGET_KEY}`, so it "
                    "can't be graded."
                )
            target = int(metadata[_TARGET_KEY])

            total += 1
            response = _last_assistant_content(conversation)
            if response is None:
                errors.append(
                    {
                        "conversation_id": conversation.get("conversation_id"),
                        "word": metadata.get("word"),
                        "letter": metadata.get("letter"),
                        "target": target,
                        "prediction": None,
                    }
                )
                # Missing responses count as unparseable, matching the reward's
                # lowest value.
                reward_sum += -3.0
                continue

            reward_sum += compute_letter_count_reward(response, target)
            prediction = _extract_prediction(response)
            if prediction is None:
                errors.append(
                    {
                        "conversation_id": conversation.get("conversation_id"),
                        "word": metadata.get("word"),
                        "letter": metadata.get("letter"),
                        "target": target,
                        "prediction": None,
                    }
                )
                continue

            valid_count += 1
            if prediction == target:
                count += 1
            else:
                errors.append(
                    {
                        "conversation_id": conversation.get("conversation_id"),
                        "word": metadata.get("word"),
                        "letter": metadata.get("letter"),
                        "target": target,
                        "prediction": prediction,
                    }
                )

    return {
        # Metric names match the `count_letters` evaluation function.
        "accuracy": count / total if total > 0 else 0,
        "properly_extracted_accuracy": count / valid_count if valid_count > 0 else 0,
        "num_samples": total,
        # These three values sum up to num_samples.
        "num_correct_answers": count,
        "num_incorrect_answers": valid_count - count,
        "num_invalid_answers": total - valid_count,
        "mean_reward": reward_sum / total if total > 0 else 0,
        "errors": errors,
    }


def _print_metrics(path: Path, metrics: dict[str, Any], num_errors: int) -> None:
    """Prints one file's metrics, plus up to `num_errors` failing examples."""
    print(f"\n{path}")
    for key, value in metrics.items():
        if key == "errors":
            continue
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key:<28} {formatted}")

    errors = metrics["errors"]
    if num_errors <= 0 or not errors:
        return
    print(f"  first {min(num_errors, len(errors))} of {len(errors)} failures:")
    for error in errors[:num_errors]:
        prediction = error["prediction"]
        predicted = "unparseable" if prediction is None else prediction
        print(
            f"    {error['conversation_id']}: '{error['letter']}' in "
            f"'{error['word']}' -> predicted {predicted}, expected {error['target']}"
        )


def main() -> None:
    """Scores each JSONL file passed on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="JSONL files of generations to score, e.g. generations_trained.jsonl.",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=0,
        help="Print this many incorrect or unparseable examples per file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write all metrics to, keyed by input file.",
    )
    args = parser.parse_args()

    all_metrics = {}
    for path in args.paths:
        metrics = score_file(path)
        all_metrics[str(path)] = metrics
        _print_metrics(path, metrics, args.show_errors)

    if args.output_json:
        args.output_json.write_text(json.dumps(all_metrics, indent=2))
        print(f"\nWrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
