from oumi.core.rollout.user_sim import (
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
        "You are Jane, a customer.",
        history,
        current_turn=2,
        max_turns=6,
        goal="Get a refund.",
    )
    assert conv.messages[0].role == Role.SYSTEM
    assert conv.messages[0].content == (
        "You are Jane, a customer.\n\nYour goal for this conversation: Get a refund."
    )
    assert [m.content for m in conv.messages[1:3]] == ["hi", "hello"]
    assert conv.messages[-1].role == Role.USER
    assert isinstance(conv.messages[-1].content, str)
    turn_info = conv.messages[-1].content
    assert "safety cap, not a target" in turn_info
    assert DEFAULT_DONE_SENTINEL in turn_info


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
    assert state.turn_idx == 1


def test_next_user_turn_produces_exactly_max_turns():
    state = RolloutState(persona="p", max_turns=1)

    done, text, _ = next_user_turn(state, [], lambda c: "only turn")
    assert (done, text) == (False, "only turn")

    done, text, _ = next_user_turn(state, [], lambda c: "should not be used")
    assert (done, text) == (True, "")
    assert state.turn_idx == 1


def test_next_user_turn_prompt_uses_custom_sentinel():
    state = RolloutState(persona="p", max_turns=5)
    captured = {}

    def stub(conv):
        captured["conv"] = conv
        return "bye <<STOP>>"

    done, text, _ = next_user_turn(state, [], stub, done_sentinel="<<STOP>>")
    assert "<<STOP>>" in captured["conv"].messages[-1].content
    assert (done, text) == (True, "bye")


def test_next_user_turn_threads_correct_turn_number():
    state = RolloutState(persona="p", max_turns=5)
    captured = {}

    def stub(conv):
        captured["conv"] = conv
        return "reply"

    next_user_turn(state, [], stub)
    turn_info = captured["conv"].messages[-1].content
    assert "reply 1" in turn_info
    assert "at most 5" in turn_info


def test_messages_to_history_drops_system_turns():
    out = messages_to_history(
        [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert [(m.role, m.content) for m in out] == [
        (Role.USER, "hi"),
        (Role.ASSISTANT, "hello"),
    ]


def test_messages_to_history_drops_tool_turns():
    # Chat APIs reject a tool turn that does not follow structured `tool_calls`, and
    # the customer would not have seen it anyway.
    out = messages_to_history(
        [
            {"role": "user", "content": "where is order 4421"},
            {"role": "tool", "content": '{"status": "delayed"}'},
            {"role": "assistant", "content": "It is delayed."},
        ]
    )
    assert [(m.role, m.content) for m in out] == [
        (Role.USER, "where is order 4421"),
        (Role.ASSISTANT, "It is delayed."),
    ]
