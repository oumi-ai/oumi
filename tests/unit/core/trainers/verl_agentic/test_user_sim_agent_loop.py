import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("verl")

from verl.experimental.agent_loop.agent_loop import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    _agent_loop_registry,
)
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
TOKENS_PER_MESSAGE = 4
SEED_TURN = {"role": "user", "content": "my order is late"}
SIM_KWARGS = {"user_persona": "Jane", "goal": "get a refund", "max_turns": 3}


def _agent_data(policy_tokens=3):
    return SimpleNamespace(
        messages=[dict(SEED_TURN)],
        prompt_ids=list(range(PROMPT_LEN)) + [900] * policy_tokens,
        response_ids=[900] * policy_tokens,
        response_mask=[1] * policy_tokens,
        response_logprobs=[],
        user_turns=0,
        assistant_turns=1,
    )


def _loop(response_length=1000, max_turns=4):
    loop = UserSimToolAgentLoop.__new__(UserSimToolAgentLoop)
    loop.response_length = response_length
    loop.max_assistant_turns = None
    loop.max_user_turns = None
    loop.tokenizer = SimpleNamespace(decode=lambda ids, **kw: "let me check")
    loop._sim_config_path = "sim.yaml"
    loop._sim = RolloutState(persona="Jane", max_turns=max_turns)
    loop._sim_history = [dict(SEED_TURN)]

    async def apply_chat_template(messages, **kwargs):
        return [100] * (TOKENS_PER_MESSAGE * len(messages))

    loop.apply_chat_template = apply_chat_template
    return loop


def _generating_state(loop, data, parent_state=AgentState.TERMINATED, reply=None):
    with (
        patch.object(
            ToolAgentLoop, "_handle_generating_state", return_value=parent_state
        ),
        patch(f"{_MODULE}.next_user_turn", return_value=reply) as sim,
    ):
        state = asyncio.run(loop._handle_generating_state(data, {}, False))
    return state, sim


def test_tool_then_simulated_user_turn_masks_only_policy_tokens():
    loop, data = _loop(), _agent_data()
    asyncio.run(
        loop._append_environment_turn(data, {"role": "tool", "content": "late"})
    )
    data.response_ids = [950, 951]
    data.prompt_ids += data.response_ids
    data.response_mask += [1, 1]

    state, _ = _generating_state(loop, data, reply=(False, "when?"))

    assert state is AgentState.GENERATING
    t = TOKENS_PER_MESSAGE
    assert data.response_mask == [1] * 3 + [0] * t + [1] * 2 + [0] * t
    assert len(data.prompt_ids) - len(data.response_mask) == PROMPT_LEN
    assert [m["role"] for m in data.messages] == ["user", "tool", "user"]
    assert loop._sim_history == [
        SEED_TURN,
        {"role": "assistant", "content": "let me check"},
        {"role": "user", "content": "when?"},
    ]
    assert data.user_turns == 1


def test_done_sentinel_terminates_without_appending():
    loop, data = _loop(), _agent_data()
    before = list(data.prompt_ids)

    state, _ = _generating_state(loop, data, reply=(True, "thanks"))

    assert state is AgentState.TERMINATED
    assert data.prompt_ids == before
    assert data.user_turns == 1


def test_turn_that_does_not_fit_terminates_without_appending():
    loop, data = _loop(response_length=4), _agent_data()
    before = (list(data.prompt_ids), list(data.response_mask), list(data.messages))

    state, _ = _generating_state(loop, data, reply=(False, "x"))

    assert state is AgentState.TERMINATED
    assert (data.prompt_ids, data.response_mask, data.messages) == before


@pytest.mark.parametrize(
    "setup",
    [
        lambda loop, data: setattr(loop, "response_length", 3),
        lambda loop, data: setattr(loop, "max_assistant_turns", 1),
        lambda loop, data: (
            setattr(loop, "max_user_turns", 1),
            setattr(data, "user_turns", 1),
        ),
        lambda loop, data: setattr(loop._sim, "turn_idx", loop._sim.max_turns),
    ],
    ids=["response_length", "max_assistant_turns", "max_user_turns", "sim_max_turns"],
)
def test_caps_stop_the_simulated_user(setup):
    loop, data = _loop(), _agent_data()
    setup(loop, data)
    user_turns = data.user_turns

    state, sim = _generating_state(loop, data)

    assert state is AgentState.TERMINATED
    sim.assert_not_called()
    assert data.user_turns == user_turns


@pytest.mark.parametrize(
    "parent_state,has_sim",
    [(AgentState.PROCESSING_TOOLS, True), (AgentState.TERMINATED, False)],
)
def test_parent_state_passes_through(parent_state, has_sim):
    loop, data = _loop(), _agent_data()
    if not has_sim:
        loop._sim = None

    state, sim = _generating_state(loop, data, parent_state=parent_state)

    assert state is parent_state
    sim.assert_not_called()


def _run(loop, extra_info, telemetry=None):
    output = SimpleNamespace(extra_fields=dict(telemetry or {}))

    async def parent_run(_self, sampling_params, **kwargs):
        if loop._sim is not None:
            loop._sim.turn_idx = 2
        return output

    loop._sim, loop._sim_history = None, []
    with patch.object(ToolAgentLoop, "run", parent_run):
        return asyncio.run(loop.run({}, extra_info=extra_info, raw_prompt=[SEED_TURN]))


def test_run_reports_simulated_user_turns_after_engine_telemetry():
    loop = _loop()

    output = _run(loop, {"interaction_kwargs": SIM_KWARGS}, {"num_turns": 5})

    assert output.extra_fields == {"num_turns": 5, "sim_user_turns": 2}
    loop._sim_history[0]["content"] = "mutated"
    assert SEED_TURN["content"] == "my order is late"


def test_run_without_interaction_kwargs_is_plain_tool_agent_loop():
    loop = _loop()

    output = _run(loop, {"tools_kwargs": {"run_sql": {}}})

    assert loop._sim is None
    assert output.extra_fields == {}


def test_run_with_persona_but_no_engine_config_raises():
    loop = _loop()
    loop._sim_config_path = None

    with pytest.raises(ValueError, match="user_sim_inference"):
        _run(loop, {"interaction_kwargs": SIM_KWARGS})


def test_import_keeps_yaml_kwargs_in_verl_registry():
    name = "oumi_user_sim_tool_agent"
    entry = {"_target_": f"{_MODULE}.UserSimToolAgentLoop", "user_sim_inference": "x"}
    saved = _agent_loop_registry.get(name)
    _agent_loop_registry[name] = dict(entry)
    try:
        importlib.reload(importlib.import_module(_MODULE))
        assert _agent_loop_registry[name] == entry
    finally:
        _agent_loop_registry.pop(name, None)
        if saved is not None:
            _agent_loop_registry[name] = saved
