import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from oumi.core.trainers.verl_grpo_trainer import VerlGrpoTrainer
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.core.types.tool_call import FunctionCall, ToolCall

try:
    verl_import_failed = False
    import verl  # pyright: ignore[reportMissingImports]  # noqa: F401
except ModuleNotFoundError:
    verl_import_failed = True


def _example(conversation: Conversation) -> dict:
    return {"conversation_json": conversation.to_json()}


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_init_without_verl():
    with patch("oumi.core.trainers.verl_grpo_trainer.verl", None):
        with pytest.raises(RuntimeError, match="verl is not installed"):
            VerlGrpoTrainer(
                processing_class=MagicMock(),
                config=MagicMock(),
                reward_funcs=[MagicMock()],
                train_dataset=MagicMock(),
                eval_dataset=MagicMock(),
            )


def test_create_verl_data_entry_single_turn():
    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="4"),
        ]
    )
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 7, "my_dataset", "train"
    )
    assert entry["data_source"] == "my_dataset"
    assert entry["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
    assert entry["images"] == []
    assert entry["reward_model"] == {"style": "rule", "ground_truth": "4"}
    assert entry["extra_info"]["split"] == "train"
    assert entry["extra_info"]["index"] == 7
    assert entry["extra_info"]["answer"] == "4"


def test_create_verl_data_entry_multi_turn():
    convo = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Hello!"),
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="4"),
        ]
    )
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 0, "my_dataset", "validation"
    )
    assert entry["prompt"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert entry["images"] == []
    assert entry["reward_model"]["ground_truth"] == "4"


def test_create_verl_data_entry_single_turn_image_prepends_marker():
    convo = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=b"imgbytes"),
                    ContentItem(type=Type.TEXT, content="Describe this."),
                ],
            ),
            Message(role=Role.ASSISTANT, content="A cat."),
        ]
    )
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 0, "my_dataset", "train"
    )
    assert entry["prompt"] == [{"role": "user", "content": "<image>Describe this."}]
    assert entry["images"] == [{"bytes": b"imgbytes"}]


def test_create_verl_data_entry_single_turn_image_with_system():
    convo = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=b"imgbytes"),
                    ContentItem(type=Type.TEXT, content="Describe this."),
                ],
            ),
            Message(role=Role.ASSISTANT, content="A cat."),
        ]
    )
    entry = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 0, "my_dataset", "train"
    )
    assert entry["prompt"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "<image>Describe this."},
    ]
    assert entry["images"] == [{"bytes": b"imgbytes"}]


def test_create_verl_data_entry_multi_turn_with_images_raises():
    convo = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=b"imgbytes"),
                    ContentItem(type=Type.TEXT, content="Describe this."),
                ],
            ),
            Message(role=Role.ASSISTANT, content="A cat."),
            Message(role=Role.USER, content="And this one?"),
            Message(role=Role.ASSISTANT, content="A dog."),
        ]
    )
    with pytest.raises(ValueError, match="multi-turn"):
        VerlGrpoTrainer._create_verl_data_entry_from_conversation(
            _example(convo), 0, "my_dataset", "train"
        )


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_init_with_multiple_reward_funcs():
    with pytest.raises(ValueError, match="We only support up to one reward function"):
        VerlGrpoTrainer(
            processing_class=MagicMock(),
            config=MagicMock(),
            reward_funcs=[MagicMock(), MagicMock()],
            train_dataset=MagicMock(),
            eval_dataset=MagicMock(),
        )


def test_create_verl_data_entry_tool_agent_carries_metadata():
    convo = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You can call run_sql."),
            Message(role=Role.USER, content="How many rows in t?"),
        ],
        metadata={
            "agent_name": "tool_agent",
            "ground_truth": "SELECT count(*) FROM t",
            "tools_kwargs": {
                "run_sql": {
                    "create_kwargs": {"schema_sql": "CREATE TABLE t(x INTEGER);"}
                }
            },
        },
    )
    row = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 0, "nl2sql", "train"
    )
    assert row["agent_name"] == "tool_agent"
    assert row["reward_model"]["ground_truth"] == "SELECT count(*) FROM t"
    assert row["extra_info"]["need_tools_kwargs"] is True
    assert "run_sql" in row["extra_info"]["tools_kwargs"]
    assert row["prompt"][-1]["role"] == "user"


def test_create_verl_data_entry_tool_agent_preserves_structured_history():
    tool_call = ToolCall(
        id="call_1",
        function=FunctionCall(
            name="run_sql", arguments='{"query":"SELECT count(*) FROM t"}'
        ),
    )
    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="How many rows in t?"),
            Message(role=Role.ASSISTANT, content=None, tool_calls=[tool_call]),
            Message(role=Role.TOOL, content='{"rows":[[2]]}', tool_call_id="call_1"),
        ],
        metadata={
            "agent_name": "tool_agent",
            "ground_truth": "SELECT count(*) FROM t",
            "tools_kwargs": {"run_sql": {}},
        },
    )

    row = VerlGrpoTrainer._create_verl_data_entry_from_conversation(
        _example(convo), 0, "nl2sql", "train"
    )

    assert row["prompt"] == [
        {"role": "user", "content": "How many rows in t?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_sql",
                        "arguments": '{"query":"SELECT count(*) FROM t"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": '{"rows":[[2]]}', "tool_call_id": "call_1"},
    ]
    assert json.loads(json.dumps(row["prompt"])) == row["prompt"]


def _make_trainer_for_config(save_steps: int, save_final_model: bool):
    """Builds a trainer with just enough state for `_create_config`.

    `_create_config` only reads `_oumi_config`, `_train_filepath`, `_val_filepath`,
    `_reward_funcs` and `_temp_output_dir`, so we bypass `__init__` entirely. That
    keeps this a unit test: no tokenizer download, no Ray, no GPU.
    """
    from oumi.core.configs import (
        DataParams,
        DatasetParams,
        DatasetSplitParams,
        ModelParams,
        TrainerType,
        TrainingConfig,
        TrainingParams,
    )

    trainer = object.__new__(VerlGrpoTrainer)
    trainer._oumi_config = TrainingConfig(
        model=ModelParams(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[DatasetParams(dataset_name="d1shs0ap/countdown")]
            ),
            # verl requires a validation split.
            validation=DatasetSplitParams(
                datasets=[DatasetParams(dataset_name="d1shs0ap/countdown")]
            ),
        ),
        training=TrainingParams(
            trainer_type=TrainerType.VERL_GRPO,
            max_steps=2,
            save_steps=save_steps,
            save_epoch=False,
            save_final_model=save_final_model,
            enable_wandb=False,
            output_dir="/tmp/oumi-verl-unit-test",
        ),
    )
    trainer._train_filepath = "/tmp/train.parquet"
    trainer._val_filepath = "/tmp/val.parquet"
    trainer._reward_funcs = []
    trainer._temp_output_dir = Path("/tmp/oumi-verl-unit-test/verl_output")
    return trainer


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
@pytest.mark.parametrize(
    "save_steps,save_final_model,expect_checkpoint",
    [
        # `save_steps: -1` is what configs/examples/grpo_verl_countdown ships.
        # verl only writes a checkpoint when save_freq > 0, so without the
        # override the run would train fine and export nothing.
        (-1, True, True),
        (0, True, True),
        # An explicit cadence is left alone.
        (5, True, True),
        # Nothing will be exported, so no checkpoint needs forcing.
        (-1, False, False),
    ],
)
def test_save_freq_guarantees_a_checkpoint_to_export(
    save_steps, save_final_model, expect_checkpoint
):
    """save_final_model requires a verl checkpoint to merge into HF format."""
    verl_config = _make_trainer_for_config(
        save_steps, save_final_model
    )._create_config()

    # verl saves when `save_freq > 0 and (is_last_step or step % save_freq == 0)`,
    # so any positive value guarantees the final step is checkpointed.
    if expect_checkpoint:
        assert verl_config.trainer.save_freq > 0
    else:
        assert verl_config.trainer.save_freq <= 0

    if save_steps > 0:
        # An explicit cadence must not be silently overridden.
        assert verl_config.trainer.save_freq == save_steps
    elif save_final_model:
        # Forced value must be large enough that only the last step matches,
        # otherwise we would checkpoint every step.
        assert verl_config.trainer.save_freq > verl_config.trainer.total_training_steps


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_verl_train_saves_final_model():
    """The verl path must reach save_model(); it returns before the shared one."""
    from oumi.train import _verl_train

    trainer = MagicMock()
    config = MagicMock()
    config.training.save_final_model = True

    import ray  # pyright: ignore[reportMissingImports]

    # Run the remote function inline so call order is observable.
    with (
        patch.object(ray, "is_initialized", return_value=True),
        patch.object(
            ray, "remote", lambda fn: type("_F", (), {"remote": staticmethod(fn)})
        ),
        patch.object(ray, "get", lambda x: x),
    ):
        _verl_train(lambda: trainer, config)

    trainer.train.assert_called_once()
    trainer.save_model.assert_called_once_with(config=config)
    assert trainer.method_calls.index(call.train()) < trainer.method_calls.index(
        call.save_model(config=config)
    )


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_verl_train_respects_save_final_model_false():
    """save_final_model=False must not export."""
    from oumi.train import _verl_train

    trainer = MagicMock()
    config = MagicMock()
    config.training.save_final_model = False

    import ray  # pyright: ignore[reportMissingImports]

    with (
        patch.object(ray, "is_initialized", return_value=True),
        patch.object(
            ray, "remote", lambda fn: type("_F", (), {"remote": staticmethod(fn)})
        ),
        patch.object(ray, "get", lambda x: x),
    ):
        _verl_train(lambda: trainer, config)

    trainer.train.assert_called_once()
    trainer.save_model.assert_not_called()
