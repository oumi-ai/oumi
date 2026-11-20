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
from typing import Any

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.conversation_utils import create_list_of_message_json_dicts
from oumi.utils.grpo_utils import extract_prompt_images_completion_from_conversation


def conversation_to_verl_row(
    conversation_json: str, *, index: int, data_source: str, split: str
) -> dict[str, Any]:
    """Converts a serialized oumi conversation to a verl dataset row.

    A conversation whose metadata has `agent_name: tool_agent` becomes a tool-agent
    row. Any other becomes a single-turn row, with the final assistant message as
    the ground truth.
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
