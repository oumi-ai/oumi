import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("verl")

from verl.experimental.agent_loop.tool_agent_loop import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    AgentState,
    ToolAgentLoop,
)

from oumi.core.rollout.user_sim import RolloutState  # noqa: E402
from oumi.core.trainers.verl_agentic.user_sim_agent_loop import (  # noqa: E402
    UserSimToolAgentLoop,
)

_MODULE = "oumi.core.trainers.verl_agentic.user_sim_agent_loop"
PROMPT_LEN = 10
# 4 token ids per appended message, from the stub apply_chat_template below.
TOKENS_PER_MESSAGE = 4


def _agent_data(policy_tokens: int = 3):
    """Rollout state after the policy has generated `policy_tokens` tokens."""
    return SimpleNamespace(
        messages=[{"role": "user", "content": "my order is late"}],
        prompt_ids=list(range(PROMPT_LEN)) + [900 + i for i in range(policy_tokens)],
        response_ids=[900 + i for i in range(policy_tokens)],
        response_mask=[1] * policy_tokens,
        response_logprobs=[],
        turn_scores=[],
        user_turns=0,
        assistant_turns=1,
        extra_fields={},
    )


def _loop(response_length: int = 1000, max_turns: int = 4):
    loop = UserSimToolAgentLoop.__new__(UserSimToolAgentLoop)
    loop.response_length = response_length
    loop.max_assistant_turns = None
    loop.max_user_turns = None
    loop.loop = asyncio.new_event_loop()
    loop.tokenizer = SimpleNamespace(decode=lambda ids, **kw: "let me check on that")
    loop._sim_config_path = "unused.yaml"
    loop._sim = RolloutState(persona="Jane", max_turns=max_turns)

    async def _apply_chat_template(messages, **kwargs):
        return [100 + i for i in range(TOKENS_PER_MESSAGE * len(messages))]

    loop.apply_chat_template = _apply_chat_template
    return loop


def _run_sim_turn(loop, data, turn_result):
    with (
        patch(f"{_MODULE}.next_user_turn", return_value=turn_result),
        patch(f"{_MODULE}.user_sim_engine", return_value=(object(), object())),
    ):
        return loop.loop.run_until_complete(
            loop._run_simulated_user_turn(data, loop._sim, "unused.yaml")
        )


def test_simulated_user_tokens_are_masked_zero():
    loop, data = _loop(), _agent_data()
    state = _run_sim_turn(loop, data, (False, "and my refund?", 0.0))

    assert state is AgentState.GENERATING
    assert data.response_mask[:3] == [1, 1, 1], "policy tokens must stay mask 1"
    assert set(data.response_mask[3:]) == {0}, "simulated-user tokens must be mask 0"
    assert len(data.response_mask) == 3 + TOKENS_PER_MESSAGE


def test_boundary_invariant_holds_after_append():
    loop, data = _loop(), _agent_data()
    _run_sim_turn(loop, data, (False, "and my refund?", 0.0))

    assert len(data.prompt_ids) - len(data.response_mask) == PROMPT_LEN


def test_assistant_message_synced_without_duplicating_tokens():
    loop, data = _loop(), _agent_data()
    _run_sim_turn(loop, data, (False, "and my refund?", 0.0))

    roles = [m["role"] for m in data.messages]
    assert roles == ["user", "assistant", "user"]
    assert data.messages[1]["content"] == "let me check on that"
    # Only the simulated-user turn adds tokens; the assistant's are already there.
    assert len(data.prompt_ids) == PROMPT_LEN + 3 + TOKENS_PER_MESSAGE


def test_user_turn_counted_and_score_recorded():
    loop, data = _loop(), _agent_data()
    _run_sim_turn(loop, data, (False, "and my refund?", 0.5))

    assert data.user_turns == 1
    assert data.turn_scores == [0.5]


def test_done_sentinel_terminates_without_tokenizing():
    loop, data = _loop(), _agent_data()
    before = list(data.prompt_ids)
    state = _run_sim_turn(loop, data, (True, "", 0.0))

    assert state is AgentState.TERMINATED
    assert data.prompt_ids == before, "a finished turn must not be tokenized"


def test_overlong_turn_terminates_without_mutating():
    loop, data = _loop(response_length=4), _agent_data()
    before_ids, before_mask = list(data.prompt_ids), list(data.response_mask)
    state = _run_sim_turn(loop, data, (False, "x", 0.0))

    assert state is AgentState.TERMINATED
    assert data.prompt_ids == before_ids
    assert data.response_mask == before_mask


@pytest.mark.parametrize(
    "cap_attr,counter_attr",
    [("max_assistant_turns", "assistant_turns"), ("max_user_turns", "user_turns")],
)
def test_turn_caps_prevent_simulated_user_turn(cap_attr, counter_attr):
    loop, data = _loop(), _agent_data()
    setattr(loop, cap_attr, 1)
    setattr(data, counter_attr, 1)

    assert loop._hard_cap_hit(data) is True


def test_response_length_cap_prevents_simulated_user_turn():
    loop, data = _loop(response_length=3), _agent_data()

    assert loop._hard_cap_hit(data) is True
    assert loop._hard_cap_hit(data, ignore_termination=True) is False


def test_no_caps_hit_allows_simulated_user_turn():
    loop, data = _loop(), _agent_data()

    assert loop._hard_cap_hit(data) is False


def test_importing_this_module_does_not_clobber_yaml_registry_entry():
    """verl's @register overwrites the registry entry with {_target_} alone.

    Importing this module is what resolves `_target_`, so decorating the loop would
    drop the `user_sim_inference` key that agent_loop_config_path supplies.
    """
    import importlib

    from verl.experimental.agent_loop.agent_loop import (  # pyright: ignore[reportMissingImports]
        _agent_loop_registry,
    )

    name = "oumi_user_sim_tool_agent"
    target = "oumi.core.trainers.verl_agentic.user_sim_agent_loop.UserSimToolAgentLoop"
    saved = _agent_loop_registry.get(name)
    try:
        _agent_loop_registry[name] = {
            "_target_": target,
            "user_sim_inference": "configs/user_sim_engine.yaml",
        }
        importlib.reload(importlib.import_module(_MODULE))
        assert (
            _agent_loop_registry[name].get("user_sim_inference")
            == "configs/user_sim_engine.yaml"
        ), "module import stripped the YAML-supplied constructor kwargs"
    finally:
        if saved is None:
            _agent_loop_registry.pop(name, None)
        else:
            _agent_loop_registry[name] = saved


# --- Verification item 9: tools and the simulator compose ---------------------


def _generating_state(loop, data, parent_returns):
    """Runs our override with the parent's decision stubbed to `parent_returns`."""
    with patch.object(
        ToolAgentLoop, "_handle_generating_state", return_value=parent_returns
    ):
        return loop.loop.run_until_complete(
            loop._handle_generating_state(data, {}, False)
        )


def test_tool_path_is_never_intercepted():
    """A pending tool call belongs to the parent; the simulator must stay out of it."""
    loop, data = _loop(), _agent_data()

    with patch(f"{_MODULE}.next_user_turn") as sim:
        state = _generating_state(loop, data, AgentState.PROCESSING_TOOLS)

    assert state is AgentState.PROCESSING_TOOLS
    sim.assert_not_called()
    assert data.user_turns == 0


def test_simulator_skipped_entirely_when_not_configured():
    """Tool-only rows must behave exactly like stock ToolAgentLoop."""
    loop, data = _loop(), _agent_data()
    loop._sim = None

    with patch(f"{_MODULE}.next_user_turn") as sim:
        state = _generating_state(loop, data, AgentState.TERMINATED)

    assert state is AgentState.TERMINATED
    sim.assert_not_called()


def test_tool_turn_then_simulated_user_turn_interleave():
    """user -> assistant(tool_call) -> tool -> assistant(text) -> sim_user -> assistant.

    The tool result and the simulated-user turn are both environment text, so the
    mask must read 1,0,1,0 across the four spans.
    """
    loop, data = _loop(), _agent_data(policy_tokens=3)

    # verl's tool path appends the tool result exactly as we append environment text.
    loop.loop.run_until_complete(
        loop._append_environment_turn(
            data, [{"role": "tool", "content": "status=late"}]
        )
    )
    tool_span = len(data.response_mask)

    # The policy generates again, this time with no tool call.
    data.response_ids = [950, 951]
    data.prompt_ids += data.response_ids
    data.response_mask += [1, 1]
    data.assistant_turns += 1

    state = _run_sim_turn(loop, data, (False, "when will it arrive?", 0.0))

    assert state is AgentState.GENERATING
    assert [m["role"] for m in data.messages] == [
        "user",
        "tool",
        "assistant",
        "user",
    ]
    assert data.response_mask[:3] == [1, 1, 1], "first assistant turn"
    assert set(data.response_mask[3:tool_span]) == {0}, "tool result is environment"
    assert data.response_mask[tool_span : tool_span + 2] == [1, 1], "second assistant"
    assert set(data.response_mask[tool_span + 2 :]) == {0}, "simulated user"
    assert len(data.prompt_ids) - len(data.response_mask) == PROMPT_LEN


# --- Verification item 7: drift guards ----------------------------------------


def test_parent_generating_state_signature_is_unchanged():
    """Our override must keep matching ToolAgentLoop's, which we delegate to."""
    import inspect

    assert list(
        inspect.signature(ToolAgentLoop._handle_generating_state).parameters
    ) == ["self", "agent_data", "sampling_params", "ignore_termination"]


def test_agent_state_members_are_known():
    """An unrecognized state upstream may need handling in _handle_generating_state.

    INTERACTING exists in 0.7 and was removed in 0.8; it is inert for us either way,
    since we never set `interaction_config_path`.
    """
    required = {"PENDING", "GENERATING", "PROCESSING_TOOLS", "TERMINATED"}
    tolerated = required | {"INTERACTING"}
    members = set(AgentState.__members__)

    assert required <= members, (
        f"upstream dropped a state we rely on: {required - members}"
    )
    assert members <= tolerated, f"unhandled new AgentState: {members - tolerated}"
