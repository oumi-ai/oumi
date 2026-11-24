import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from datasets import Dataset, IterableDataset

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
)
from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type
from oumi.verl_grpo.data import conversation_to_verl_row, write_verl_parquet
from oumi_verl_ext.tool_episode import TOOL_AGENT_LOOP_NAME


def _row(conversation: Conversation, multi_turn: bool = False) -> dict:
    return conversation_to_verl_row(
        conversation.to_json(),
        index=3,
        data_source="ds",
        split="train",
        multi_turn=multi_turn,
    )


def test_single_turn_row():
    prompt = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Pick one."},
    ]
    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="Be brief."),
            Message(role=Role.USER, content="Pick one."),
            Message(role=Role.ASSISTANT, content="A"),
        ],
        metadata={"k": "v"},
    )

    assert _row(conversation) == {
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
            "metadata": json.dumps({"k": "v"}),
            "prompt_json": json.dumps(prompt),
        },
    }


def test_image_row_gets_the_image_marker():
    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=b"img"),
                    ContentItem(type=Type.TEXT, content="Describe this."),
                ],
            ),
            Message(role=Role.ASSISTANT, content="A cat."),
        ]
    )

    row = _row(conversation)

    assert row["prompt"] == [{"role": "user", "content": "<image>Describe this."}]
    assert row["images"] == [{"bytes": b"img"}]


def test_tool_agent_row():
    tools_kwargs = {"run_sql": {"create_kwargs": {"schema_sql": "CREATE TABLE t;"}}}
    conversation = Conversation(
        messages=[Message(role=Role.USER, content="How many rows in t?")],
        metadata={
            "agent_name": "tool_agent",
            "ground_truth": "SELECT count(*) FROM t",
            "tools_kwargs": tools_kwargs,
        },
    )

    row = _row(conversation)

    assert row["agent_name"] == "tool_agent"
    assert row["ability"] == "tool_agent"
    assert row["reward_model"]["ground_truth"] == "SELECT count(*) FROM t"
    assert row["extra_info"]["need_tools_kwargs"] is True
    assert row["extra_info"]["tools_kwargs"] == tools_kwargs


def test_multi_turn_runs_conversation_rows_in_oumi_s_tool_loop():
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Pick one."),
            Message(role=Role.ASSISTANT, content="A"),
        ],
        metadata={"k": "v"},
    )

    assert _row(conversation, multi_turn=True) == {
        **_row(conversation),
        "agent_name": TOOL_AGENT_LOOP_NAME,
    }


_IMAGE = ContentItem(type=Type.IMAGE_BINARY, binary=b"img")


def _tool_conversation(*more: Message, **metadata) -> Conversation:
    return Conversation(
        messages=[Message(role=Role.USER, content="How many rows in t?"), *more],
        metadata={
            "agent_name": "tool_agent",
            "ground_truth": "SELECT count(*) FROM t",
            "tools_kwargs": {"run_sql": {}},
            **metadata,
        },
    )


def test_multi_turn_keeps_the_agent_a_row_names():
    conversation = _tool_conversation()

    assert _row(conversation, multi_turn=True) == _row(conversation)


@pytest.mark.parametrize(
    ("conversation", "error"),
    [
        (_tool_conversation(ground_truth=""), "ground_truth"),
        (_tool_conversation(tools_kwargs={"run_sql": "x"}), "tools_kwargs"),
        (
            _tool_conversation(Message(role=Role.ASSISTANT, content="3")),
            "must end with",
        ),
        (_tool_conversation(Message(role=Role.USER, content=[_IMAGE])), "images"),
        (
            Conversation(
                messages=[
                    Message(role=Role.USER, content=[_IMAGE]),
                    Message(role=Role.ASSISTANT, content="A cat."),
                    Message(role=Role.USER, content="And now?"),
                ]
            ),
            "single-turn",
        ),
    ],
)
def test_bad_conversations_are_refused(conversation, error):
    with pytest.raises(ValueError, match=error):
        _row(conversation)


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
    assert "conversation_json" not in table.columns


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


def _two_dataset_params() -> DataParams:
    return DataParams(
        train=DatasetSplitParams(
            datasets=[
                DatasetParams(dataset_name="first"),
                DatasetParams(dataset_name="second"),
            ]
        )
    )


def test_conversations_from_two_dataset_names_are_refused(tmp_path: Path):
    convo = Conversation(messages=[Message(role=Role.USER, content="Hi")])
    dataset = Dataset.from_list([{"conversation_json": convo.to_json()}])

    with _returning(dataset), pytest.raises(ValueError, match="single dataset name"):
        write_verl_parquet(
            _two_dataset_params(),
            split=DatasetSplit.TRAIN,
            tokenizer=None,
            output_path=tmp_path / "train.parquet",
        )


def test_a_mixed_split_already_in_verl_format_is_accepted(tmp_path: Path):
    rows = [
        {"data_source": "first", "prompt": [{"role": "user", "content": "a"}]},
        {"data_source": "second", "prompt": [{"role": "user", "content": "b"}]},
    ]

    with _returning(Dataset.from_list(rows)):
        count = write_verl_parquet(
            _two_dataset_params(),
            split=DatasetSplit.TRAIN,
            tokenizer=None,
            output_path=tmp_path / "train.parquet",
        )

    assert count == 2
    assert list(pd.read_parquet(tmp_path / "train.parquet")["data_source"]) == [
        "first",
        "second",
    ]


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
