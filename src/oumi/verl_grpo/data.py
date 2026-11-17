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

"""Turns oumi conversation datasets into the parquet rows verl reads."""

import json
import os
from pathlib import Path
from typing import Any, Final

from datasets import Dataset

from oumi.builders.data import build_dataset_mixture
from oumi.core.configs import DataParams
from oumi.core.configs.params.data_params import DatasetSplit
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.types.conversation import Conversation
from oumi.core.types.conversation import Role as ConversationRole
from oumi.utils.conversation_utils import create_list_of_message_json_dicts
from oumi.utils.grpo_utils import extract_prompt_images_completion_from_conversation

SINGLE_TURN_AGENT: Final = "single_turn_agent"
TOOL_AGENT: Final = "tool_agent"

_PROMPT_JSON_EXTRA_INFO_KEY: Final = "prompt_json"
_IMAGE_MARKER: Final = "<image>"
_CONVERSATION_COLUMN: Final = "conversation_json"
_MAX_CONVERT_PROCESSES: Final = 8


def _extract_prompt_images_answer(
    example: dict,
) -> tuple[list[dict], list, str]:
    """Finds the prompt, images and answer of a conversation.

    The prompt is in verl's chat format and the answer is the final assistant
    message's text. Prompt-only conversations have an empty answer.

    Raises:
        ValueError: If the conversation has images but is multi-turn.
    """
    prompt_messages, images, answer = (
        extract_prompt_images_completion_from_conversation(
            example, allow_prompt_only=True
        )
    )

    if len(images) > 0:
        user_messages = [m for m in prompt_messages if m["role"] == "user"]
        if len(user_messages) != 1:
            raise ValueError(
                "Images are only supported for single-turn conversations, "
                f"but the prompt has {len(user_messages)} user messages "
                "(multi-turn). Please use a text-only multi-turn conversation."
            )
        content = user_messages[0]["content"]
        if not content.startswith(_IMAGE_MARKER):
            user_messages[0]["content"] = _IMAGE_MARKER + content
    return (prompt_messages, images, answer)


def _tool_agent_row(
    conversation: Conversation,
    ground_truth: str,
    tools_kwargs: dict[str, Any],
    ability: str,
    index: int,
    data_source: str,
    split: str,
) -> dict[str, Any]:
    """Builds a verl row from a structured tool-agent prompt."""
    if any(
        message.count_content_items().image_items for message in conversation.messages
    ):
        raise ValueError("Tool-agent conversations do not support images.")
    if not conversation.messages or conversation.messages[-1].role not in (
        ConversationRole.USER,
        ConversationRole.TOOL,
    ):
        raise ValueError(
            "Tool-agent conversation prompts must end with a user or tool message."
        )

    prompt_messages = create_list_of_message_json_dicts(
        conversation.messages,
        group_adjacent_same_role_turns=False,
    )
    return {
        "data_source": data_source,
        "prompt": prompt_messages,
        "images": [],
        "ability": ability,
        "agent_name": TOOL_AGENT,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": index,
            _PROMPT_JSON_EXTRA_INFO_KEY: json.dumps(prompt_messages),
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
    }


def conversation_to_verl_row(
    conversation_json: str, *, index: int, data_source: str, split: str
) -> dict[str, Any]:
    """Converts a serialized oumi conversation to a verl dataset row.

    Args:
        conversation_json: The conversation, as produced by `Conversation.to_json()`.
        index: The row's position in its split.
        data_source: The name verl's reward manager sees as the row's data source.
        split: The split name stored in the row's `extra_info`.

    Returns:
        A verl row. Conversations whose metadata has `agent_name: tool_agent` become
        tool-agent rows; all others are single-turn rows. Every row names its agent
        explicitly, so verl's `default_agent_loop` setting does not apply to them.

    Raises:
        ValueError: If the conversation or its tool-agent metadata is invalid.
    """
    # Peek at metadata off the raw JSON so only the branch that needs a
    # `Conversation` pays for building one.
    metadata = json.loads(conversation_json).get("metadata") or {}
    if metadata.get("agent_name") == TOOL_AGENT:
        conversation = Conversation.from_json(conversation_json)
        ground_truth = metadata.get("ground_truth")
        if not isinstance(ground_truth, str) or not ground_truth:
            raise ValueError(
                "Tool-agent conversation metadata must include a non-empty "
                "string 'ground_truth'."
            )
        tools_kwargs = metadata.get("tools_kwargs")
        tools_kwargs_error = (
            "Tool-agent conversation metadata 'tools_kwargs' must be a "
            "mapping of tool names to dictionaries."
        )
        if not isinstance(tools_kwargs, dict):
            raise ValueError(f"{tools_kwargs_error} Got {type(tools_kwargs).__name__}.")
        for tool_name, tool_kwargs in tools_kwargs.items():
            if not isinstance(tool_name, str) or not isinstance(tool_kwargs, dict):
                raise ValueError(
                    f"{tools_kwargs_error} Got {tool_name!r}: "
                    f"{type(tool_kwargs).__name__}."
                )
        ability = metadata.get("ability", TOOL_AGENT)
        if not isinstance(ability, str):
            raise ValueError(
                "Tool-agent conversation metadata 'ability' must be a string."
            )
        return _tool_agent_row(
            conversation,
            ground_truth,
            tools_kwargs,
            ability,
            index,
            data_source,
            split,
        )
    prompt_messages, images, answer = _extract_prompt_images_answer(
        {"conversation_json": conversation_json}
    )
    return {
        "data_source": data_source,
        "prompt": prompt_messages,
        "images": images,
        "ability": "math",
        "agent_name": SINGLE_TURN_AGENT,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "answer": answer,
            _PROMPT_JSON_EXTRA_INFO_KEY: json.dumps(prompt_messages),
            "metadata": json.dumps(metadata),
        },
    }


def write_verl_parquet(
    data_params: DataParams,
    *,
    split: DatasetSplit,
    tokenizer: BaseTokenizer | None,
    output_path: Path,
) -> int:
    """Builds one split of `data_params` and writes it as a verl parquet file.

    Conversation rows are converted with `conversation_to_verl_row`. Rows that are
    already in verl's format are written unchanged.

    Args:
        data_params: The dataset configuration.
        split: The split to write.
        tokenizer: The tokenizer used to build the datasets, if any.
        output_path: Where to write the parquet file.

    Returns:
        The number of rows written.

    Raises:
        ValueError: If the split mixes dataset names, is streamed, or has no rows.
    """
    dataset_names = {ds.dataset_name for ds in data_params.get_split(split).datasets}
    if len(dataset_names) != 1:
        raise ValueError(
            f"Expected a single dataset name for the {split.value} split, "
            f"got {sorted(dataset_names)}."
        )
    (data_source,) = dataset_names

    dataset = build_dataset_mixture(data_params, tokenizer, split)
    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"The {split.value} split must be a map-style dataset; streaming "
            "datasets cannot be written to parquet."
        )
    if len(dataset) == 0:
        raise ValueError(f"The {split.value} split has no rows.")

    if _CONVERSATION_COLUMN in dataset.column_names:
        dataset = dataset.map(
            lambda example, index: conversation_to_verl_row(
                example[_CONVERSATION_COLUMN],
                index=index,
                data_source=data_source,
                split=split.value,
            ),
            with_indices=True,
            remove_columns=dataset.column_names,
            num_proc=min(_MAX_CONVERT_PROCESSES, os.cpu_count() or 1),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path)
    return len(dataset)
