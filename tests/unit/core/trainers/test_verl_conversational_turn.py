from oumi.core.trainers.verl_conversational_turn import (
    DEFAULT_DONE_SENTINEL,
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
    conv = build_user_turn_prompt(
        "You are Jane, a customer.", history, current_turn=2, target_turns=6
    )
    assert conv.messages[0].role == Role.SYSTEM
    assert conv.messages[0].content == "You are Jane, a customer."
    assert [m.content for m in conv.messages[1:3]] == ["hi", "hello"]
    assert conv.messages[-1].role == Role.USER
    assert DEFAULT_DONE_SENTINEL in conv.messages[-1].content


def test_messages_to_history_maps_roles():
    out = messages_to_history(
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert [(m.role, m.content) for m in out] == [
        (Role.USER, "a"),
        (Role.ASSISTANT, "b"),
    ]


def test_next_user_turn_normal():
    state = RolloutState(persona="p", max_turns=5)
    done, text, score = next_user_turn(
        state, [{"role": "assistant", "content": "hi"}], lambda c: "  my reply  "
    )
    assert (done, text, score) == (False, "my reply", 0.0)
    assert state.turn_idx == 1


def test_next_user_turn_sentinel_ends():
    state = RolloutState(persona="p", max_turns=5)
    done, text, score = next_user_turn(
        state, [], lambda c: f"thanks {DEFAULT_DONE_SENTINEL}"
    )
    assert done is True
    assert text == "thanks"


def test_next_user_turn_hits_cap():
    state = RolloutState(persona="p", max_turns=1)  # first call already at/over cap
    done, text, score = next_user_turn(state, [], lambda c: "should not be used")
    assert (done, text) == (True, "")


def test_next_user_turn_threads_correct_turn_number():
    state = RolloutState(persona="p", max_turns=5)
    captured = {}

    def stub(conv):
        captured["conv"] = conv
        return "reply"

    next_user_turn(state, [], stub)
    turn_info = captured["conv"].messages[-1].content
    assert "turn 1 of at most 5" in turn_info
