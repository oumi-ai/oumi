import json

import pytest

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


def test_missing_conversation_json_raises_with_context():
    with pytest.raises(ValueError, match="conversation_json"):
        VerlGrpoTrainer._create_verl_data_entry_from_conversation(
            {}, idx=0, data_source="s", split="train"
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


AGENT_NAME = "oumi_user_sim_tool_agent"


def test_interaction_row_routes_to_agent_loop():
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _interaction_example(), idx=3, data_source="support", split="train"
    )
    assert entry["agent_name"] == AGENT_NAME
    assert entry["reward_model"]["style"] == "rule"
    assert entry["reward_model"]["ground_truth"] == "get a refund or delivery date"
    assert entry["extra_info"]["tools_kwargs"] == {}
    assert entry["extra_info"]["need_tools_kwargs"] is False
    kwargs = entry["extra_info"]["interaction_kwargs"]
    assert kwargs["user_persona"].startswith("You are Jane")
    assert kwargs["max_turns"] == 6
    assert "name" not in kwargs


def test_tool_agent_row_routes_to_same_loop():
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
    assert entry["agent_name"] == AGENT_NAME
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
    assert entry["agent_name"] == VerlGrpoTrainer.VERL_DEFAULT_AGENT_LOOP_NAME


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
        VerlGrpoTrainer.OUMI_AGENT_LOOP_NAME,
        VerlGrpoTrainer.VERL_DEFAULT_AGENT_LOOP_NAME,
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
    """`.get(key, default)` returns None when the key is present but null."""
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

    assert entry["agent_name"] == AGENT_NAME
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
