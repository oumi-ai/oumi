import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("verl")

from verl.experimental.agent_loop.tool_agent_loop import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    AgentState,
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
