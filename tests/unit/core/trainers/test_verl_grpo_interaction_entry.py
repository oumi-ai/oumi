import json

import pytest
from omegaconf import OmegaConf

from oumi.core.trainers.verl_grpo_trainer import VerlGrpoTrainer


def _conv_json(messages, metadata=None):
    d = {"messages": messages}
    if metadata is not None:
        d["metadata"] = metadata
    return json.dumps(d)


def _interaction_example():
    return {
        "conversation_json": _conv_json(
            [
                {"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": "My order #4421 is late."},
            ],
            metadata={
                "interaction_kwargs": {
                    "user_persona": "You are Jane, a customer whose order #4421 "
                    "is late.",
                    "goal": "get a refund or delivery date",
                    "max_turns": 6,
                }
            },
        ),
    }


def test_interaction_branch_shape():
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _interaction_example(), idx=3, data_source="support", split="train"
    )
    assert len(entry["prompt"]) == 2
    assert entry["prompt"][0]["role"] == "system"
    assert entry["prompt"][-1]["role"] == "user"
    assert entry["reward_model"]["ground_truth"] == "get a refund or delivery date"
    ik = entry["extra_info"]["interaction_kwargs"]
    assert ik["max_turns"] == 6
    assert ik["goal"] == "get a refund or delivery date"
    assert ik["user_persona"].startswith("You are Jane")


def test_interaction_row_must_end_on_user():
    bad = {
        "conversation_json": _conv_json(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "resolved"},
            ],
            metadata={"interaction_kwargs": {"user_persona": "p", "goal": "g"}},
        )
    }
    with pytest.raises(ValueError):
        VerlGrpoTrainer._create_verl_data_entry_from_conversation(
            bad, idx=0, data_source="s", split="train"
        )


def test_non_interaction_row_uses_final_turn_path():
    example = {
        "conversation_json": _conv_json(
            [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
        )
    }
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        example, idx=0, data_source="math", split="train"
    )
    assert entry["reward_model"]["ground_truth"] == "4"
    assert "interaction_kwargs" not in entry["extra_info"]


def test_interaction_row_routes_to_agent_loop():
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _interaction_example(), idx=3, data_source="support", split="train"
    )
    assert entry["agent_name"] == VerlGrpoTrainer.USER_SIM_AGENT_LOOP_NAME
    assert entry["reward_model"]["style"] == "rule"
    assert entry["reward_model"]["ground_truth"] == "get a refund or delivery date"
    assert entry["extra_info"]["tools_kwargs"] == {}
    assert entry["extra_info"]["need_tools_kwargs"] is False
    kwargs = entry["extra_info"]["interaction_kwargs"]
    assert kwargs["user_persona"].startswith("You are Jane")
    assert kwargs["max_turns"] == 6
    assert "name" not in kwargs


def test_tool_only_row_routes_to_verl_tool_agent():
    example = {
        "conversation_json": _conv_json(
            [{"role": "user", "content": "How many orders?"}],
            metadata={
                "agent_name": "tool_agent",
                "ground_truth": "SELECT count(*) FROM orders",
                "tools_kwargs": {"run_sql": {"create_kwargs": {}}},
            },
        )
    }
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        example, idx=0, data_source="spider", split="train"
    )
    assert entry["agent_name"] == VerlGrpoTrainer.TOOL_AGENT_LOOP_NAME
    assert entry["extra_info"]["need_tools_kwargs"] is True
    assert "interaction_kwargs" not in entry["extra_info"]


def test_plain_row_carries_verl_default_agent_name():
    """Plain rows still need the key so a mixed dataset keeps a uniform schema."""
    example = {
        "conversation_json": _conv_json(
            [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        )
    }
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        example, idx=0, data_source="math", split="train"
    )
    assert entry["agent_name"] == VerlGrpoTrainer.SINGLE_TURN_AGENT_LOOP_NAME


def _mapped(rows):
    from datasets import Dataset

    return Dataset.from_list(rows).map(
        lambda ex, i: VerlGrpoTrainer._create_verl_data_entry_from_conversation(
            ex, i, "src", "train"
        ),
        with_indices=True,
    )


@pytest.mark.parametrize("plain_first", [True, False])
def test_mixed_rows_keep_agent_name_in_both_orders(plain_first):
    """`Dataset.map` fixes the schema from row 0, so every row must carry agent_name.

    Without it, plain-first silently drops the column and agent-loop rows fall back to
    single-turn generation with no error.
    """
    interaction = {
        "conversation_json": _conv_json(
            [{"role": "user", "content": "late order"}],
            metadata={"interaction_kwargs": {"user_persona": "Jane", "goal": "refund"}},
        )
    }
    plain = {
        "conversation_json": _conv_json(
            [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        )
    }
    rows = [plain, interaction] if plain_first else [interaction, plain]

    mapped = _mapped(rows)

    assert "agent_name" in mapped.column_names
    assert set(mapped["agent_name"]) == {
        VerlGrpoTrainer.USER_SIM_AGENT_LOOP_NAME,
        VerlGrpoTrainer.SINGLE_TURN_AGENT_LOOP_NAME,
    }


def test_interaction_only_dataset_writes_parquet(tmp_path):
    """Simulator-only rows carry `tools_kwargs={}`; the schema must still serialize."""
    rows = [
        {
            "conversation_json": _conv_json(
                [{"role": "user", "content": "late order"}],
                metadata={"interaction_kwargs": {"user_persona": "Jane"}},
            )
        }
    ]
    out = tmp_path / "train.parquet"

    _mapped(rows).to_parquet(str(out))

    assert out.stat().st_size > 0


def test_explicit_null_max_turns_falls_back_to_default():
    """A null `max_turns` falls back to DEFAULT_MAX_TURNS."""
    from oumi.core.rollout.user_sim import DEFAULT_MAX_TURNS

    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        {
            "conversation_json": _conv_json(
                [{"role": "user", "content": "hi"}],
                metadata={
                    "interaction_kwargs": {"user_persona": "Jane", "max_turns": None}
                },
            )
        },
        idx=0,
        data_source="src",
        split="train",
    )

    assert entry["extra_info"]["interaction_kwargs"]["max_turns"] == DEFAULT_MAX_TURNS


def _tools_and_sim_example():
    return {
        "conversation_json": _conv_json(
            [{"role": "user", "content": "where is order #4421?"}],
            metadata={
                "agent_name": "tool_agent",
                "ground_truth": "SELECT status FROM orders WHERE id=4421",
                "tools_kwargs": {"run_sql": {"create_kwargs": {}}},
                "interaction_kwargs": {
                    "user_persona": "You are Jane, chasing order #4421.",
                    "goal": "get a delivery date",
                    "max_turns": 4,
                },
            },
        )
    }


def test_row_can_carry_both_tools_and_simulated_user():
    """The loop runs tools then hands over to the simulator; the row must say so."""
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _tools_and_sim_example(), idx=0, data_source="support", split="train"
    )

    assert entry["agent_name"] == VerlGrpoTrainer.USER_SIM_AGENT_LOOP_NAME
    assert entry["extra_info"]["tools_kwargs"] == {"run_sql": {"create_kwargs": {}}}
    assert entry["extra_info"]["need_tools_kwargs"] is True
    assert entry["extra_info"]["interaction_kwargs"]["user_persona"].startswith(
        "You are Jane"
    )
    assert entry["extra_info"]["interaction_kwargs"]["max_turns"] == 4
    assert entry["reward_model"]["ground_truth"] == (
        "SELECT status FROM orders WHERE id=4421"
    )


def test_tools_only_row_has_no_simulator():
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        {
            "conversation_json": _conv_json(
                [{"role": "user", "content": "how many orders?"}],
                metadata={
                    "agent_name": "tool_agent",
                    "ground_truth": "SELECT count(*) FROM orders",
                    "tools_kwargs": {"run_sql": {}},
                },
            )
        },
        idx=0,
        data_source="spider",
        split="train",
    )

    assert "interaction_kwargs" not in entry["extra_info"]


def test_simulator_only_row_has_no_tools():
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _interaction_example(), idx=0, data_source="support", split="train"
    )

    assert entry["extra_info"]["tools_kwargs"] == {}
    assert entry["extra_info"]["need_tools_kwargs"] is False


def test_interaction_row_without_persona_raises():
    bad = {
        "conversation_json": _conv_json(
            [{"role": "user", "content": "hi"}],
            metadata={"interaction_kwargs": {"goal": "g"}},
        )
    }
    with pytest.raises(ValueError, match="user_persona"):
        VerlGrpoTrainer._create_verl_data_entry_from_conversation(
            bad, idx=0, data_source="s", split="train"
        )


def _rollout(mode="async", multi_turn=True, agent_loop_config_path=None):
    return OmegaConf.create(
        {
            "mode": mode,
            "multi_turn": {"enable": multi_turn},
            "agent": {"agent_loop_config_path": agent_loop_config_path},
        }
    )


_SINGLE = VerlGrpoTrainer.SINGLE_TURN_AGENT_LOOP_NAME
_TOOL = VerlGrpoTrainer.TOOL_AGENT_LOOP_NAME
_SIM = VerlGrpoTrainer.USER_SIM_AGENT_LOOP_NAME


@pytest.mark.parametrize("agent_names", [set(), {_SINGLE}])
def test_single_turn_dataset_keeps_default_sync_config(agent_names):
    VerlGrpoTrainer._validate_agent_loop_config(_rollout(mode="sync"), agent_names)


@pytest.mark.parametrize("agent_names", [{_TOOL}, {_SIM}, {_SINGLE, _SIM}])
@pytest.mark.parametrize("mode,multi_turn", [("sync", True), ("async", False)])
def test_multi_turn_rows_require_async_multi_turn(agent_names, mode, multi_turn):
    with pytest.raises(ValueError, match="multi_turn"):
        VerlGrpoTrainer._validate_agent_loop_config(
            _rollout(mode, multi_turn, "loops.yaml"), agent_names
        )


def test_simulated_user_rows_require_agent_loop_config_path():
    with pytest.raises(ValueError, match="agent_loop_config_path"):
        VerlGrpoTrainer._validate_agent_loop_config(_rollout(), {_SIM})


def test_tool_rows_need_no_agent_loop_config_path():
    VerlGrpoTrainer._validate_agent_loop_config(_rollout(), {_TOOL})


def test_valid_simulated_user_config_passes():
    VerlGrpoTrainer._validate_agent_loop_config(
        _rollout(agent_loop_config_path="loops.yaml"), {_SINGLE, _TOOL, _SIM}
    )


@pytest.mark.parametrize("max_turns", [0, -1, 2.5, True, "3"])
def test_invalid_max_turns_raises(max_turns):
    bad = {
        "conversation_json": _conv_json(
            [{"role": "user", "content": "hi"}],
            metadata={
                "interaction_kwargs": {"user_persona": "Jane", "max_turns": max_turns}
            },
        )
    }
    with pytest.raises(ValueError, match="max_turns"):
        VerlGrpoTrainer._create_verl_data_entry_from_conversation(
            bad, idx=0, data_source="s", split="train"
        )
