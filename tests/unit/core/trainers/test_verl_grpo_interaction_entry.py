import json

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from oumi.core.rollout.user_sim import DEFAULT_MAX_TURNS
from oumi.core.trainers.verl_grpo_trainer import VerlGrpoTrainer

_SINGLE = VerlGrpoTrainer.SINGLE_TURN_AGENT_LOOP_NAME
_TOOL = VerlGrpoTrainer.TOOL_AGENT_LOOP_NAME
_SIM = VerlGrpoTrainer.USER_SIM_AGENT_LOOP_NAME
_PERSONA = {"user_persona": "Jane", "goal": "refund", "max_turns": 4}
_TOOLS = {
    "agent_name": "tool_agent",
    "ground_truth": "SELECT 1",
    "tools_kwargs": {"run_sql": {}},
}


def _example(metadata=None, last_role="user"):
    messages = [{"role": "user", "content": "my order is late"}]
    if last_role == "assistant":
        messages.append({"role": "assistant", "content": "4"})
    conversation = {"messages": messages}
    if metadata is not None:
        conversation["metadata"] = metadata
    return {"conversation_json": json.dumps(conversation)}


def _entry(example):
    return VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        example, 0, "src", "train"
    )


@pytest.mark.parametrize(
    "metadata,last_role,agent_name,ground_truth,has_tools,has_sim",
    [
        (None, "assistant", _SINGLE, "4", False, False),
        (_TOOLS, "user", _TOOL, "SELECT 1", True, False),
        ({"interaction_kwargs": _PERSONA}, "user", _SIM, "refund", False, True),
        (
            {**_TOOLS, "interaction_kwargs": _PERSONA},
            "user",
            _SIM,
            "SELECT 1",
            True,
            True,
        ),
    ],
    ids=["plain", "tools", "simulated_user", "tools_and_simulated_user"],
)
def test_row_routing(metadata, last_role, agent_name, ground_truth, has_tools, has_sim):
    entry = _entry(_example(metadata, last_role))

    assert entry["agent_name"] == agent_name
    assert entry["reward_model"]["ground_truth"] == ground_truth
    assert entry["extra_info"].get("need_tools_kwargs", False) is has_tools
    assert entry["extra_info"].get("interaction_kwargs") == (
        _PERSONA if has_sim else None
    )


@pytest.mark.parametrize("max_turns", [None, "missing"])
def test_missing_max_turns_uses_default(max_turns):
    kwargs = {"user_persona": "Jane"}
    if max_turns != "missing":
        kwargs["max_turns"] = max_turns

    entry = _entry(_example({"interaction_kwargs": kwargs}))

    assert entry["extra_info"]["interaction_kwargs"] == {
        "user_persona": "Jane",
        "goal": "",
        "max_turns": DEFAULT_MAX_TURNS,
    }


@pytest.mark.parametrize(
    "interaction_kwargs,last_role,match",
    [
        ({"goal": "g"}, "user", "user_persona"),
        ({"user_persona": "Jane"}, "assistant", "end on the user"),
        *[
            ({"user_persona": "Jane", "max_turns": bad}, "user", "max_turns")
            for bad in (0, -1, 2.5, True, "3")
        ],
    ],
)
def test_invalid_simulated_user_row_raises(interaction_kwargs, last_role, match):
    with pytest.raises(ValueError, match=match):
        _entry(_example({"interaction_kwargs": interaction_kwargs}, last_role))


@pytest.mark.parametrize("plain_first", [True, False])
def test_mixed_dataset_keeps_agent_name_column(plain_first, tmp_path):
    rows = [_example(None, "assistant"), _example({"interaction_kwargs": _PERSONA})]
    if not plain_first:
        rows.reverse()

    mapped = Dataset.from_list(rows).map(
        lambda example, idx: VerlGrpoTrainer._create_verl_data_entry_from_conversation(
            example, idx, "src", "train"
        ),
        with_indices=True,
    )
    mapped.to_parquet(str(tmp_path / "train.parquet"))

    assert set(mapped["agent_name"]) == {_SINGLE, _SIM}


def _rollout(mode="async", multi_turn=True, agent_loop_config_path=None):
    return OmegaConf.create(
        {
            "mode": mode,
            "multi_turn": {"enable": multi_turn},
            "agent": {"agent_loop_config_path": agent_loop_config_path},
        }
    )


@pytest.mark.parametrize(
    "rollout,agent_names",
    [
        (_rollout(mode="sync", multi_turn=False), set()),
        (_rollout(mode="sync", multi_turn=False), {_SINGLE}),
        (_rollout(), {_TOOL}),
        (_rollout(agent_loop_config_path="loops.yaml"), {_SINGLE, _TOOL, _SIM}),
    ],
)
def test_valid_agent_loop_config(rollout, agent_names):
    VerlGrpoTrainer._validate_agent_loop_config(rollout, agent_names)


@pytest.mark.parametrize(
    "rollout,agent_names,match",
    [
        (_rollout(mode="sync", agent_loop_config_path="x"), {_TOOL}, "multi_turn"),
        (_rollout(multi_turn=False, agent_loop_config_path="x"), {_SIM}, "multi_turn"),
        (_rollout(), {_SINGLE, _SIM}, "agent_loop_config_path"),
    ],
)
def test_invalid_agent_loop_config_raises(rollout, agent_names, match):
    with pytest.raises(ValueError, match=match):
        VerlGrpoTrainer._validate_agent_loop_config(rollout, agent_names)
