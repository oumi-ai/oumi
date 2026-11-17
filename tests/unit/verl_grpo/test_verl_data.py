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
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from datasets import Dataset, IterableDataset

from oumi.core.configs import DataParams, DatasetParams, DatasetSplitParams
from oumi.core.configs.params.data_params import DatasetSplit
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.core.types.tool_call import FunctionCall, ToolCall
from oumi.verl_grpo.data import conversation_to_verl_row, write_verl_parquet


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


def _text_sft_params(path: Path, *, names: tuple[str, ...] = ("text_sft",)):
    return DataParams(
        train=DatasetSplitParams(
            datasets=[
                DatasetParams(
                    dataset_name=name,
                    dataset_path=str(path),
                    dataset_kwargs={"return_conversations": True},
                )
                for name in names
            ]
        )
    )


def _write_jsonl(path: Path, num_rows: int) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
        }
        for i in range(num_rows)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _prebuilt_params() -> DataParams:
    return DataParams(
        train=DatasetSplitParams(datasets=[DatasetParams(dataset_name="prebuilt")])
    )


def _returning(dataset: Any):
    return patch("oumi.verl_grpo.data.build_dataset_mixture", return_value=dataset)


def test_conversation_rows_are_converted_and_counted(tmp_path: Path):
    _write_jsonl(tmp_path / "data.jsonl", 3)
    output_path = tmp_path / "train.parquet"

    count = write_verl_parquet(
        _text_sft_params(tmp_path / "data.jsonl"),
        split=DatasetSplit.TRAIN,
        tokenizer=None,
        output_path=output_path,
    )

    table = pd.read_parquet(output_path)
    assert count == 3
    assert list(table["agent_name"]) == ["single_turn_agent"] * 3
    assert [e["index"] for e in table["extra_info"]] == [0, 1, 2]
    assert [e["split"] for e in table["extra_info"]] == ["train"] * 3
    assert set(table["data_source"]) == {"text_sft"}


def test_rows_already_in_verl_format_are_written_unchanged(tmp_path: Path):
    rows = [
        {
            "data_source": "prebuilt",
            "prompt": [{"role": "user", "content": "Hi"}],
            "reward_model": {"style": "rule", "ground_truth": "yo"},
        }
    ]
    params = _prebuilt_params()
    output_path = tmp_path / "train.parquet"

    with _returning(Dataset.from_list(rows)):
        count = write_verl_parquet(
            params, split=DatasetSplit.TRAIN, tokenizer=None, output_path=output_path
        )

    table = pd.read_parquet(output_path)
    assert count == 1
    assert list(table.columns) == ["data_source", "prompt", "reward_model"]
    assert table["reward_model"][0]["ground_truth"] == "yo"


def test_a_split_with_two_dataset_names_is_refused(tmp_path: Path):
    params = _text_sft_params(tmp_path / "x.jsonl", names=("text_sft", "other"))

    with pytest.raises(ValueError, match="single dataset name"):
        write_verl_parquet(
            params,
            split=DatasetSplit.TRAIN,
            tokenizer=None,
            output_path=tmp_path / "train.parquet",
        )


def test_an_empty_split_is_refused(tmp_path: Path):
    empty = Dataset.from_list([{"prompt": []}]).select([])

    with _returning(empty), pytest.raises(ValueError, match="no rows"):
        write_verl_parquet(
            _prebuilt_params(),
            split=DatasetSplit.TRAIN,
            tokenizer=None,
            output_path=tmp_path / "train.parquet",
        )


def test_an_iterable_dataset_is_refused(tmp_path: Path):
    iterable = IterableDataset.from_generator(lambda: iter([{"prompt": []}]))
    with _returning(iterable), pytest.raises(ValueError, match="streaming"):
        write_verl_parquet(
            _prebuilt_params(),
            split=DatasetSplit.TRAIN,
            tokenizer=None,
            output_path=tmp_path / "train.parquet",
        )
