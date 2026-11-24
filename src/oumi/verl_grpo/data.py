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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from datasets import Dataset

from oumi.builders.data import build_dataset_mixture
from oumi.core.configs import DataParams, DatasetSplit
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.conversation_utils import create_list_of_message_json_dicts
from oumi.utils.grpo_utils import extract_prompt_images_completion_from_conversation
from oumi.utils.logging import logger
from oumi_verl_ext.tool_episode import TOOL_AGENT_LOOP_NAME

_CONVERSATION_COLUMN: Final = "conversation_json"


def conversation_to_verl_row(
    conversation_json: str,
    *,
    index: int,
    data_source: str,
    split: str,
    multi_turn: bool = False,
) -> dict[str, Any]:
    """Converts a serialized oumi conversation to a verl dataset row.

    A conversation whose metadata has `agent_name: tool_agent` becomes a tool-agent
    row. Any other takes the final assistant message as the ground truth and runs
    in verl's single-turn loop, or in oumi's tool agent loop when `multi_turn` is
    set.
    """
    conversation = Conversation.from_json(conversation_json)
    metadata = conversation.metadata or {}
    _validate(conversation.messages, metadata)
    extra_info: dict[str, Any] = {"split": split, "index": index}
    if metadata.get("agent_name") == "tool_agent":
        prompt = create_list_of_message_json_dicts(
            conversation.messages, group_adjacent_same_role_turns=False
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
            user = next(message for message in prompt if message["role"] == "user")
            if not user["content"].startswith("<image>"):
                user["content"] = "<image>" + user["content"]
        agent_name = TOOL_AGENT_LOOP_NAME if multi_turn else "single_turn_agent"
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


def _validate(messages: list[Message], metadata: dict[str, Any]) -> None:
    """Refuses conversations that would silently train on bad data or fail mid-run."""
    has_images = any(message.count_content_items().image_items for message in messages)
    if metadata.get("agent_name") != "tool_agent":
        if has_images and sum(message.role == Role.USER for message in messages) != 1:
            raise ValueError("Images need a single-turn conversation.")
        return
    ground_truth = metadata.get("ground_truth")
    if not isinstance(ground_truth, str) or not ground_truth:
        raise ValueError("A tool-agent conversation needs a 'ground_truth'.")
    tools_kwargs = metadata.get("tools_kwargs")
    if not isinstance(tools_kwargs, dict) or not all(
        isinstance(kwargs, dict) for kwargs in tools_kwargs.values()
    ):
        raise ValueError("'tools_kwargs' must map tool names to dicts.")
    if not messages or messages[-1].role not in (Role.USER, Role.TOOL):
        raise ValueError("A tool-agent prompt must end with a user or tool message.")
    if has_images:
        raise ValueError("Tool-agent conversations can't have images.")


def write_verl_parquet(
    data_params: DataParams,
    *,
    split: DatasetSplit,
    tokenizer: BaseTokenizer | None,
    output_path: Path,
    multi_turn: bool = False,
    max_prompt_length: int | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    min_rows: int = 1,
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
        tokenizer: The tokenizer used to build the datasets and measure prompts.
        output_path: Where to write the parquet file.
        multi_turn: Whether conversation rows run in oumi's tool agent loop.
        max_prompt_length: If set, rows whose chat-templated prompt, with
            `tool_schemas` rendered, is longer than this many tokens are dropped.
        chat_template_kwargs: Extra chat template arguments for measuring prompts.
        tool_schemas: The tools rendered into every prompt.
        min_rows: The fewest rows the split may keep.

    Returns:
        The number of rows written.

    Raises:
        ValueError: If the split is streamed or has no rows, if conversations come
            from zero or several dataset names, if a conversation is invalid, or if
            fewer than `min_rows` rows are kept.
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
                multi_turn=multi_turn,
            ),
            with_indices=True,
            remove_columns=dataset.column_names,
            # Limit the max number of sub-processes to 8 to avoid overloading the
            # system with too many processes.
            num_proc=min(8, os.cpu_count() or 1),
        )

    total = len(dataset)
    if max_prompt_length is not None:
        if tokenizer is None:
            raise ValueError("Measuring prompts needs a tokenizer.")
        # verl's own overlong-prompt filter tokenizes the same way.
        kwargs = {
            key: value
            for key, value in (chat_template_kwargs or {}).items()
            if key not in ("tokenize", "return_dict", "return_tensors")
        }
        lengths = [
            len(
                tokenizer.apply_chat_template(
                    prompt,
                    tools=tool_schemas,  # type: ignore[arg-type]
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **kwargs,
                )
            )
            for prompt in dataset["prompt"]
        ]
        kept = [i for i, length in enumerate(lengths) if length <= max_prompt_length]
        if len(kept) < total:
            dataset = dataset.select(kept)
            logger.info(
                f"Dropped {total - len(kept)} of {total} {split.value} rows with "
                f"prompts over max_prompt_length={max_prompt_length} tokens."
            )
    if len(dataset) < min_rows:
        raise ValueError(
            f"The {split.value} split keeps {len(dataset)} rows, fewer than the "
            f"{min_rows} it needs; {total - len(dataset)} were dropped with prompts "
            f"over max_prompt_length={max_prompt_length} tokens."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path)
    return len(dataset)
