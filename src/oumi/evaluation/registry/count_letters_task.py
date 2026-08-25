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

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonlines

from oumi.core.configs import EvaluationConfig
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.core.types.conversation import Conversation
from oumi.datasets.grpo.letter_count import LetterCountGrpoDataset
from oumi.utils.logging import logger


def _extract_prediction(response: str) -> int | None:
    r"""Returns the numeric answer extracted from `\boxed{...}`, or None otherwise."""
    regex_result = re.findall(r"\\boxed\{([-+]?\d+)\}", response)
    if not regex_result or len(regex_result) != 1:
        return None
    number_str = regex_result[0]
    # Except clause shouldn't trigger because the regex should only find ints.
    try:
        return int(number_str)
    except ValueError:
        return None


def _save_conversations(conversations: list[Conversation], output_path: str) -> None:
    """Saves conversations to a JSONL file in Oumi chat format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for conversation in conversations:
            writer.write(conversation.to_dict())


@register_evaluation_function("count_letters")
def count_letters(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    config: EvaluationConfig | None = None,
    dataset_name: str = "oumi-ai/oumi-letter-count-clean",
    split: str = "test",
    inference_output_path: str | None = None,
) -> dict[str, Any]:
    """Custom evaluation function registered as `count_letters`.

    The dataset is configurable from the task's `eval_kwargs`::

        tasks:
          - evaluation_backend: custom
            task_name: count_letters
            num_samples: 1000
            eval_kwargs:
              dataset_name: oumi-ai/oumi-letter-count-clean
              split: test

    The full inference output (each conversation, annotated in its metadata with
    the extracted `prediction` and `is_correct`) is saved as JSONL to
    `inference_output_path` (also settable via `eval_kwargs`). If unset, it
    defaults to `<config.output_dir>/count_letters_inference_<timestamp>.jsonl`;
    pass an empty string to disable saving.
    """
    dataset = LetterCountGrpoDataset(dataset=dataset_name, split=split)
    # TODO: OPE-1155: Add support for using Oumi dataset code to create the dataset.
    # dataset = build_dataset("oumi-ai/oumi-letter-count", tokenizer=None, sample_count=10)  # noqa: E501
    num_samples = task_params.num_samples
    if num_samples is None:
        num_samples = len(dataset)
    input_conversations = [dataset.conversation(i) for i in range(num_samples)]
    conversations = inference_engine.infer(input_conversations)
    logger.info(f"Finished inference on {len(conversations)} conversations!")
    if len(conversations) > 0:
        logger.info(f"Sample conversation: {conversations[0]}")

    count = 0  # The number of examples with correct answers extracted.
    total = 0  # All examples.
    valid_count = 0  # The number of examples with valid answers extracted.
    for conversation in conversations:
        total += 1
        # Grab the model's response
        response = conversation.last_message()
        # Ignore cases where model didn't respond or it's a multimodal response.
        # For now, we focus on text-only responses.
        prediction = None
        if response and isinstance(response.content, str):
            prediction = _extract_prediction(response.content)
        is_correct = (
            prediction is not None
            and prediction == conversation.metadata["letter_count_integer"]
        )
        conversation.metadata["prediction"] = prediction
        conversation.metadata["is_correct"] = is_correct
        # Count the example as correct if the extracted prediction is correct.
        if prediction is None:
            continue
        valid_count += 1
        if is_correct:
            count += 1

    if inference_output_path is None and config is not None and config.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inference_output_path = str(
            Path(config.output_dir) / f"count_letters_inference_{timestamp}.jsonl"
        )
    if inference_output_path:
        _save_conversations(conversations, inference_output_path)
        logger.info(f"Saved annotated inference output to: {inference_output_path}")

    return {
        # Accuracy across all examples.
        "accuracy": count / total if total > 0 else 0,
        # Accuracy when only counting examples with properly extracted answers.
        "properly_extracted_accuracy": count / valid_count if valid_count > 0 else 0,
        "num_samples": num_samples,
        # These three values sum up to num_samples.
        "num_correct_answers": count,
        "num_incorrect_answers": valid_count - count,
        "num_invalid_answers": total - valid_count,
    }
