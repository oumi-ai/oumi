import asyncio

import pytest

from oumi.core.rollout.user_sim import (
    DONE_SENTINEL,
    RolloutState,
    build_user_turn_prompt,
    messages_to_history,
    next_user_turn,
)
from oumi.core.types.conversation import Message, Role


def test_build_user_turn_prompt_shape():
    history = [
        Message(role=Role.USER, content="hi"),
        Message(role=Role.ASSISTANT, content="hello"),
    ]
    state = RolloutState(persona="You are Jane.", max_turns=6, goal="Refund.")
    state.turn_idx = 2

    conv = build_user_turn_prompt(state, history)

    assert conv.messages[0].content == (
        "You are Jane.\n\nYour goal for this conversation: Refund."
    )
    assert conv.messages[1:3] == history
    instruction = conv.messages[-1].content
    assert conv.messages[-1].role == Role.USER
    assert isinstance(instruction, str)
    assert "reply 2" in instruction
    assert "at most 6" in instruction
    assert DONE_SENTINEL in instruction


def test_messages_to_history_keeps_only_user_and_assistant_text():
    out = messages_to_history(
        [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "where is order 4421"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": '{"status": "delayed"}'},
            {"role": "assistant", "content": "It is delayed."},
        ]
    )

    assert [(m.role, m.content) for m in out] == [
        (Role.USER, "where is order 4421"),
        (Role.ASSISTANT, "It is delayed."),
    ]


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("  my reply  ", (False, "my reply")),
        (f"thanks {DONE_SENTINEL}", (True, "thanks")),
    ],
)
def test_next_user_turn(reply, expected):
    state = RolloutState(persona="p", max_turns=5)

    async def generate(conversation):
        return reply

    assert asyncio.run(next_user_turn(state, [], generate)) == expected
    assert state.turn_idx == 1
