import json

import pytest

from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type
from oumi.verl_grpo.data import conversation_to_verl_row


def _row(conversation: Conversation) -> dict:
    return conversation_to_verl_row(
        conversation.to_json(), index=3, data_source="ds", split="train"
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
