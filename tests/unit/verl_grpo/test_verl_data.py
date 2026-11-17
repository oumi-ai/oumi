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

import json
from typing import Any

import pytest

from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.core.types.tool_call import FunctionCall, ToolCall
from oumi.verl_grpo.data import conversation_to_verl_row


def _row(
    conversation: Conversation,
    *,
    index: int = 0,
    data_source: str = "my_dataset",
    split: str = "train",
) -> dict[str, Any]:
    return conversation_to_verl_row(
        conversation.to_json(), index=index, data_source=data_source, split=split
    )


def _image_message(text: str = "Describe this.") -> Message:
    return Message(
        role=Role.USER,
        content=[
            ContentItem(type=Type.IMAGE_BINARY, binary=b"imgbytes"),
            ContentItem(type=Type.TEXT, content=text),
        ],
    )


def _tool_agent_conversation(**metadata_overrides: Any) -> Conversation:
    metadata: dict[str, Any] = {
        "agent_name": "tool_agent",
        "ground_truth": "SELECT count(*) FROM t",
        "tools_kwargs": {"run_sql": {}},
    }
    metadata.update(metadata_overrides)
    return Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You can call run_sql."),
            Message(role=Role.USER, content="How many rows in t?"),
        ],
        metadata=metadata,
    )


def test_single_turn_row_is_the_0_7_row_plus_agent_name():
    prompt = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Pick one."},
    ]
    convo = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="Be brief."),
            Message(role=Role.USER, content="Pick one."),
            Message(role=Role.ASSISTANT, content="A"),
        ],
        metadata={"k": "v"},
    )

    assert _row(convo, index=3, data_source="ds") == {
        "data_source": "ds",
        "prompt": prompt,
        "images": [],
        "ability": "math",
        "agent_name": "single_turn_agent",
        "reward_model": {"style": "rule", "ground_truth": "A"},
        "extra_info": {
            "split": "train",
            "index": 3,
            "answer": "A",
            "prompt_json": json.dumps(prompt),
            "metadata": json.dumps({"k": "v"}),
        },
    }


def test_multi_turn_prompt_keeps_history():
    convo = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Hello!"),
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="4"),
        ]
    )

    row = _row(convo, split="validation")

    assert row["prompt"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert row["reward_model"]["ground_truth"] == "4"
    assert json.loads(row["extra_info"]["prompt_json"]) == row["prompt"]


def test_prompt_only_conversation_has_empty_ground_truth():
    convo = Conversation(messages=[Message(role=Role.USER, content="What is 2+2?")])

    row = _row(convo)

    assert row["reward_model"]["ground_truth"] == ""
    assert row["prompt"] == [{"role": "user", "content": "What is 2+2?"}]


def test_single_turn_image_row_prepends_the_image_marker():
    convo = Conversation(
        messages=[_image_message(), Message(role=Role.ASSISTANT, content="A cat.")]
    )

    row = _row(convo)

    assert row["prompt"] == [{"role": "user", "content": "<image>Describe this."}]
    assert row["images"] == [{"bytes": b"imgbytes"}]


def test_single_turn_image_row_with_system_prompt():
    convo = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            _image_message(),
            Message(role=Role.ASSISTANT, content="A cat."),
        ]
    )

    row = _row(convo)

    assert row["prompt"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "<image>Describe this."},
    ]
    assert row["images"] == [{"bytes": b"imgbytes"}]


def test_multi_turn_with_images_raises():
    convo = Conversation(
        messages=[
            _image_message(),
            Message(role=Role.ASSISTANT, content="A cat."),
            Message(role=Role.USER, content="And this one?"),
            Message(role=Role.ASSISTANT, content="A dog."),
        ]
    )

    with pytest.raises(ValueError, match="multi-turn"):
        _row(convo)


def test_tool_agent_row_carries_tools_kwargs():
    tools_kwargs = {
        "run_sql": {"create_kwargs": {"schema_sql": "CREATE TABLE t(x INTEGER);"}}
    }
    convo = _tool_agent_conversation(tools_kwargs=tools_kwargs)

    row = _row(convo, data_source="nl2sql")

    assert row["agent_name"] == "tool_agent"
    assert row["ability"] == "tool_agent"
    assert row["reward_model"]["ground_truth"] == "SELECT count(*) FROM t"
    assert row["extra_info"]["need_tools_kwargs"] is True
    assert row["extra_info"]["tools_kwargs"] == tools_kwargs
    assert "answer" not in row["extra_info"]
    assert "metadata" not in row["extra_info"]
    assert row["prompt"][-1]["role"] == "user"


def test_tool_agent_preserves_structured_history():
    tool_call = ToolCall(
        id="call_1",
        function=FunctionCall(
            name="run_sql", arguments='{"query":"SELECT count(*) FROM t"}'
        ),
    )
    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="How many rows in t?"),
            Message(role=Role.ASSISTANT, content=None, tool_calls=[tool_call]),
            Message(role=Role.TOOL, content='{"rows":[[2]]}', tool_call_id="call_1"),
        ],
        metadata=_tool_agent_conversation().metadata,
    )

    row = _row(convo, data_source="nl2sql")

    assert row["prompt"] == [
        {"role": "user", "content": "How many rows in t?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_sql",
                        "arguments": '{"query":"SELECT count(*) FROM t"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": '{"rows":[[2]]}', "tool_call_id": "call_1"},
    ]
    assert json.loads(row["extra_info"]["prompt_json"]) == row["prompt"]


@pytest.mark.parametrize(
    ("metadata_overrides", "message"),
    [
        ({"ground_truth": ""}, "non-empty string 'ground_truth'"),
        ({"tools_kwargs": ["run_sql"]}, "mapping of tool names to dictionaries"),
        ({"tools_kwargs": {"run_sql": "x"}}, "mapping of tool names to dictionaries"),
        ({"ability": 3}, "'ability' must be a string"),
    ],
)
def test_tool_agent_rejects_bad_metadata(metadata_overrides: dict, message: str):
    with pytest.raises(ValueError, match=message):
        _row(_tool_agent_conversation(**metadata_overrides))


def test_tool_agent_rejects_missing_ground_truth():
    convo = _tool_agent_conversation()
    del convo.metadata["ground_truth"]

    with pytest.raises(ValueError, match="non-empty string 'ground_truth'"):
        _row(convo)


def test_tool_agent_rejects_a_prompt_ending_with_the_assistant():
    convo = _tool_agent_conversation()
    convo.messages.append(Message(role=Role.ASSISTANT, content="done"))

    with pytest.raises(ValueError, match="must end with a user or tool message"):
        _row(convo)


def test_tool_agent_rejects_images():
    convo = _tool_agent_conversation()
    convo.messages[-1] = _image_message()

    with pytest.raises(ValueError, match="do not support images"):
        _row(convo)
