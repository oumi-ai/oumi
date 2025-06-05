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

import os
from typing import Union

import datasets as hf_datasets

from oumi.core.types.conversation import Conversation, Role


def extract_prompt_images_completion_from_single_turn_conversation(
    example: dict,
) -> tuple[str, list, str]:
    """Finds prompt, completion, and optional images in a single-turn conversation.

    Args:
        example: A dictionary containing the conversation JSON.

    Returns:
        A tuple containing the prompt, images, and completion.
        The list of images is empty for text-only conversations.
    """
    if "conversation_json" not in example:
        raise ValueError(
            f"Example doesn't contain 'conversation_json' key. "
            f"Available keys: {example.keys()}"
        )

    conversation_json = example["conversation_json"]
    conversation = Conversation.from_json(conversation_json)

    user_messages = conversation.filter_messages(role=Role.USER)
    if len(user_messages) != 1:
        raise ValueError(f"Expected 1 user message, but got {len(user_messages)}.")

    assistant_messages = conversation.filter_messages(role=Role.ASSISTANT)
    if len(assistant_messages) != 1:
        raise ValueError(
            f"Expected 1 assistant message, but got {len(assistant_messages)}."
        )

    user_message = user_messages[0]
    assistant_message = assistant_messages[0]
    prompt: str = user_message.text_content_items[-1].content or ""
    images = [{"bytes": item.binary} for item in user_message.image_content_items]
    answer: str = assistant_message.text_content_items[-1].content or ""

    return (prompt, images, answer)


def try_prepare_trl_grpo_example(
    example: dict,
) -> dict:
    """Prepares an example for GRPO_TRL processing.

    This function checks if the input example is one of known special cases
    e.g., SFT example, and transforms it into a GRPO compatible format.
    Otherwise, it returns the original example.

    Args:
        example (dict): The input example.

    Returns:
        GRPO compatible example, or an original example.
    """
    if "conversation_json" in example:
        prompt, images, answer = (
            extract_prompt_images_completion_from_single_turn_conversation(example)
        )
        if len(images) > 0:
            raise ValueError(
                f"Image content is not supported in GRPO_TRL yet. "
                f"Found {len(images)} image(s) in an example."
            )
        return {
            "prompt": prompt,
            "completion": answer,
        }

    return example


def try_prepare_trl_grpo_dataset(
    dataset: Union[hf_datasets.Dataset, hf_datasets.IterableDataset],
) -> Union[hf_datasets.Dataset, hf_datasets.IterableDataset]:
    """Prepares a dataset for GRPO_TRL processing."""
    column_names = dataset.column_names
    if column_names and ("conversation_json" not in column_names):
        return dataset
    if isinstance(dataset, hf_datasets.Dataset):
        # Limit the max number of sub-processes to 8 to avoid overloading the system
        # with too many processes.
        # TODO: Make this configurable.
        num_proc = min(8, os.cpu_count() or 1)
        dataset = dataset.map(
            function=try_prepare_trl_grpo_example,
            with_indices=False,
            num_proc=num_proc,
            remove_columns=["conversation_json"],
        )
    else:
        dataset = dataset.map(
            function=try_prepare_trl_grpo_example,
            with_indices=False,
            remove_columns=["conversation_json"],
        )

    print(f"Transformed GRPO Dataset columns: {dataset.column_names}")
    return dataset
