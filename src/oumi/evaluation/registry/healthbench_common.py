# Copyright 2026 - Oumi
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

"""Shared pieces of the HealthBench evaluations.

Two harnesses build on this module: `healthbench_task` grades each example
against its own physician-written rubric (the faithful benchmark), and
`healthbench_global_task` grades every example against one consolidated
rubric. They share dataset loading, response caching and the scoring
primitives so the two numbers stay comparable.
"""

import json
import random
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np

from oumi.core.types.conversation import Conversation, Role
from oumi.utils.logging import logger

DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/openai/healthbench/resolve/main/"
    "2025-05-07-06-14-12_oss_eval.jsonl"
)

HEALTHBENCH_AXES = (
    "accuracy",
    "completeness",
    "context_awareness",
    "communication_quality",
    "instruction_following",
)


def calculate_healthbench_score(
    rubrics: Sequence[dict[str, Any]],
    criteria_met: Sequence[bool],
) -> float | None:
    """Calculates one HealthBench example score using the reference formula."""
    if len(rubrics) != len(criteria_met):
        raise ValueError("rubrics and criteria_met must have the same length")

    total_possible_points = sum(
        float(rubric["points"]) for rubric in rubrics if float(rubric["points"]) > 0
    )
    if total_possible_points == 0:
        return None

    achieved_points = sum(
        float(rubric["points"])
        for rubric, is_met in zip(rubrics, criteria_met, strict=True)
        if is_met
    )
    return achieved_points / total_possible_points


def clipped_mean(values: Sequence[float]) -> float:
    """Returns the mean of `values`, clipped to [0, 1] like the reference eval."""
    return float(np.clip(np.mean(values), 0.0, 1.0))


def bootstrap_std(
    values: Sequence[float], *, num_bootstrap_samples: int, seed: int
) -> float:
    """Returns the bootstrap standard deviation of the clipped mean."""
    if not values or num_bootstrap_samples <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    value_array = np.asarray(values, dtype=float)
    indices = rng.integers(
        0, len(value_array), size=(num_bootstrap_samples, len(value_array))
    )
    means = np.clip(value_array[indices].mean(axis=1), 0.0, 1.0)
    return float(np.std(means))


def download_dataset(dataset_url: str, dataset_path: Path) -> None:
    """Downloads the HealthBench jsonl to `dataset_path`."""
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = dataset_path.with_suffix(f"{dataset_path.suffix}.tmp")
    logger.info(f"Downloading HealthBench from {dataset_url} to {dataset_path}")
    urllib.request.urlretrieve(dataset_url, temporary_path)
    temporary_path.replace(dataset_path)


def load_examples(dataset_path: Path) -> list[dict[str, Any]]:
    """Loads HealthBench examples from a jsonl file."""
    with jsonlines.open(dataset_path) as reader:
        examples = list(reader)
    if not examples:
        raise ValueError(f"No HealthBench examples found in {dataset_path}")
    return examples


def select_examples(
    examples: list[dict[str, Any]], num_samples: int | None, seed: int
) -> list[dict[str, Any]]:
    """Selects `num_samples` examples using the reference sampling procedure."""
    if num_samples is None:
        return examples
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if num_samples > len(examples):
        raise ValueError(
            f"Requested {num_samples} samples, but HealthBench has {len(examples)}"
        )
    if num_samples == len(examples):
        return examples
    return random.Random(seed).sample(examples, num_samples)


def resolve_dataset(
    dataset_path: str, dataset_url: str, num_samples: int | None, sample_seed: int
) -> list[dict[str, Any]]:
    """Downloads the dataset if needed and returns the selected examples."""
    resolved_dataset_path = Path(dataset_path)
    if not resolved_dataset_path.exists():
        download_dataset(dataset_url, resolved_dataset_path)
    return select_examples(
        load_examples(resolved_dataset_path), num_samples, sample_seed
    )


def build_input_conversations(
    examples: Sequence[dict[str, Any]],
) -> list[Conversation]:
    """Builds the model-input conversations, carrying rubrics in the metadata."""
    conversations = []
    for example in examples:
        conversations.append(
            Conversation.from_dict(
                {
                    "messages": example["prompt"],
                    "metadata": {
                        "prompt_id": example["prompt_id"],
                        "example_tags": example["example_tags"],
                        "rubrics": example["rubrics"],
                    },
                }
            )
        )
    return conversations


def save_conversations(
    conversations: Sequence[Conversation], output_path: Path
) -> None:
    """Writes conversations to a jsonl file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode="w") as writer:
        for conversation in conversations:
            writer.write(conversation.to_dict())


def load_completed_conversations(output_path: Path) -> list[Conversation]:
    """Reads cached model responses from a jsonl file."""
    with jsonlines.open(output_path) as reader:
        return [Conversation.from_dict(row) for row in reader]


def validate_completed_conversations(
    conversations: Sequence[Conversation], examples: Sequence[dict[str, Any]]
) -> None:
    """Checks that cached responses line up with the selected examples."""
    if len(conversations) != len(examples):
        raise ValueError(
            f"Cached inference has {len(conversations)} rows; expected {len(examples)}"
        )
    for index, (conversation, example) in enumerate(
        zip(conversations, examples, strict=True)
    ):
        if conversation.metadata.get("prompt_id") != example["prompt_id"]:
            raise ValueError(f"Cached inference prompt_id mismatch at row {index}")
        response = conversation.last_message()
        if response is None or response.role != Role.ASSISTANT:
            raise ValueError(f"Cached inference row {index} has no assistant response")
        if not isinstance(response.content, str):
            raise ValueError(f"Cached inference row {index} has a non-text response")


def subset_cached_conversations(
    conversations: Sequence[Conversation], examples: Sequence[dict[str, Any]]
) -> list[Conversation]:
    """Reorders a cached superset of responses to match `examples`.

    A cache generated over the full dataset is reusable by any subset run --
    a pilot, or the validation subset -- but `select_examples` draws a random
    sample, so the cache order does not match. Matching on prompt_id lets one
    expensive generation pass serve every later subset without regenerating.
    """
    by_prompt_id: dict[str, Conversation] = {}
    for conversation in conversations:
        prompt_id = conversation.metadata.get("prompt_id")
        if prompt_id is not None:
            by_prompt_id[prompt_id] = conversation
    missing = [
        example["prompt_id"]
        for example in examples
        if example["prompt_id"] not in by_prompt_id
    ]
    if missing:
        raise ValueError(
            f"Cached inference is missing {len(missing)} of {len(examples)} "
            f"requested prompt_ids (first: {missing[0]})"
        )
    return [by_prompt_id[example["prompt_id"]] for example in examples]


def obtain_responses(
    *,
    examples: Sequence[dict[str, Any]],
    inference_engine: Any,
    inference_path: Path,
) -> list[Conversation]:
    """Returns model responses, reusing the cache at `inference_path` if present."""
    if inference_path.exists():
        logger.info(f"Loading cached model responses from {inference_path}")
        conversations = load_completed_conversations(inference_path)
        if len(conversations) != len(examples):
            logger.info(
                f"Cache holds {len(conversations)} responses for {len(examples)} "
                "requested examples; selecting by prompt_id"
            )
            conversations = subset_cached_conversations(conversations, examples)
    else:
        conversations = inference_engine.infer(build_input_conversations(examples))
        save_conversations(conversations, inference_path)
        logger.info(f"Saved {len(conversations)} model responses to {inference_path}")
    validate_completed_conversations(conversations, examples)
    return conversations


def conversation_to_text(conversation: Conversation) -> str:
    """Renders a conversation as the plain text shown to the judge."""
    parts = []
    for message in conversation.messages:
        if not isinstance(message.content, str):
            raise ValueError("HealthBench only supports text conversation messages")
        parts.append(f"{message.role.value}: {message.content}")
    return "\n\n".join(parts)


def write_json(path: Path, value: Any) -> None:
    """Writes `value` as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_jsonlines(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Writes `rows` to a jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(rows)
