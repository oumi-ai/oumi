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

"""Turns oumi conversations into the rows verl reads from parquet."""

import json
import os
from pathlib import Path
from typing import Any, Final

from datasets import Dataset

from oumi.builders.data import build_dataset_mixture
from oumi.core.configs import DataParams, DatasetSplit
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.types.conversation import Conversation
from oumi.utils.conversation_utils import create_list_of_message_json_dicts
from oumi.utils.grpo_utils import extract_prompt_images_completion_from_conversation

_CONVERSATION_COLUMN: Final = "conversation_json"


def conversation_to_verl_row(
    conversation_json: str, *, index: int, data_source: str, split: str
) -> dict[str, Any]:
    """Converts a serialized oumi conversation to a verl dataset row.

    A conversation whose metadata has `agent_name: tool_agent` becomes a tool-agent
    row. Any other becomes a single-turn row, with the final assistant message as
    the ground truth.
    """
    metadata = json.loads(conversation_json).get("metadata") or {}
    extra_info: dict[str, Any] = {"split": split, "index": index}
    if metadata.get("agent_name") == "tool_agent":
        messages = Conversation.from_json(conversation_json).messages
        prompt = create_list_of_message_json_dicts(
            messages, group_adjacent_same_role_turns=False
        )
        images = []
        agent_name = "tool_agent"
        ability = metadata.get("ability", "tool_agent")
        ground_truth = metadata["ground_truth"]
        extra_info |= {
            "need_tools_kwargs": True,
            "tools_kwargs": metadata["tools_kwargs"],
        }
    else:
        prompt, images, ground_truth = (
            extract_prompt_images_completion_from_conversation(
                {"conversation_json": conversation_json}, allow_prompt_only=True
            )
        )
        if images:
            # verl places the images at an <image> marker in the user message.
            user_messages = [m for m in prompt if m["role"] == "user"]
            if len(user_messages) != 1:
                raise ValueError("Images need a single-turn conversation.")
            if not user_messages[0]["content"].startswith("<image>"):
                user_messages[0]["content"] = "<image>" + user_messages[0]["content"]
        agent_name = "single_turn_agent"
        ability = "math"
        extra_info |= {"answer": ground_truth, "metadata": json.dumps(metadata)}
    # The platform's reward reads the prompt from here.
    extra_info["prompt_json"] = json.dumps(prompt)
    return {
        "data_source": data_source,
        "prompt": prompt,
        "images": images,
        "ability": ability,
        "agent_name": agent_name,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
    }


def write_verl_parquet(
    data_params: DataParams,
    *,
    split: DatasetSplit,
    tokenizer: BaseTokenizer | None,
    output_path: Path,
) -> int:
    """Builds one split of `data_params` and writes it as a verl parquet file.

    Conversation rows are converted with `conversation_to_verl_row`, which requires
    the split to name a single dataset. The source `conversation_json` column is
    dropped after conversion, so the file holds only verl's columns. Rows that are
    already in verl's format are written unchanged and may mix datasets, since they
    carry their own `data_source`.

    Args:
        data_params: The dataset configuration.
        split: The split to write.
        tokenizer: The tokenizer used to build the datasets, if any.
        output_path: Where to write the parquet file.

    Returns:
        The number of rows written.

    Raises:
        ValueError: If the split is streamed or has no rows, if conversations come
            from zero or several dataset names, or if a conversation is invalid.
    """
    dataset = build_dataset_mixture(data_params, tokenizer, split)
    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"The {split.value} split must be a map-style dataset; streaming "
            "datasets cannot be written to parquet."
        )
    if len(dataset) == 0:
        raise ValueError(f"The {split.value} split has no rows.")

    if _CONVERSATION_COLUMN in dataset.column_names:
        dataset_names = {
            ds.dataset_name for ds in data_params.get_split(split).datasets
        }
        if len(dataset_names) != 1:
            raise ValueError(
                f"Expected a single dataset name for the {split.value} split, "
                f"got {sorted(dataset_names)}."
            )
        (data_source,) = dataset_names
        dataset = dataset.map(
            lambda example, index: conversation_to_verl_row(
                example[_CONVERSATION_COLUMN],
                index=index,
                data_source=data_source,
                split=split.value,
            ),
            with_indices=True,
            remove_columns=dataset.column_names,
            # Limit the max number of sub-processes to 8 to avoid overloading the
            # system with too many processes.
            num_proc=min(8, os.cpu_count() or 1),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path)
    return len(dataset)
