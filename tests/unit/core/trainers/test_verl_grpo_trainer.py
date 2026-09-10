import json
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from oumi.core.constants import VERL_METRICS_FILENAME
from oumi.core.trainers import verl_grpo_trainer
from oumi.core.trainers.verl_grpo_trainer import VerlGrpoTrainer
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.core.types.tool_call import FunctionCall, ToolCall
from oumi.utils.verl_utils.grpo_metrics import REWARD_GROUP_LOW_STD_CONFIG_KEY

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
    assert json.loads(entry["extra_info"]["prompt_json"]) == entry["prompt"]


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
    assert json.loads(row["extra_info"]["prompt_json"]) == row["prompt"]
    assert json.loads(json.dumps(row["prompt"])) == row["prompt"]


def _make_trainer_for_config(
    save_steps: int,
    save_final_model: bool,
    save_epoch: bool = False,
    output_dir: str = "/tmp/oumi-verl-unit-test",
):
    """Builds a trainer with just enough state for `_create_config`.

    `_create_config` only reads `_oumi_config`, `_train_filepath`, `_val_filepath`,
    `_reward_funcs` and `_temp_output_dir`, so we bypass `__init__` entirely. That
    keeps this a unit test: no tokenizer download, no Ray, no GPU. `_export_hf_model`
    additionally reads `_final_output_dir`.
    """
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
            save_epoch=save_epoch,
            save_final_model=save_final_model,
            enable_wandb=False,
            output_dir=output_dir,
        ),
    )
    trainer._train_filepath = "/tmp/train.parquet"
    trainer._val_filepath = "/tmp/val.parquet"
    trainer._reward_funcs = []
    trainer._final_output_dir = Path(output_dir)
    trainer._temp_output_dir = Path(output_dir) / "verl_output"
    return trainer


def _make_trainer_for_setup(adv_estimator: str, low_std_threshold: float):
    trainer = object.__new__(VerlGrpoTrainer)
    verl_config = MagicMock()
    verl_config.algorithm.adv_estimator = adv_estimator
    verl_config.algorithm.get.side_effect = lambda key, default: (
        low_std_threshold if key == REWARD_GROUP_LOW_STD_CONFIG_KEY else default
    )
    verl_config.trainer.n_gpus_per_node = 1
    verl_config.trainer.nnodes = 1
    trainer._create_config = MagicMock(return_value=verl_config)
    trainer._processing_class = MagicMock()
    trainer._processor = None
    trainer._reward_funcs = []
    return trainer


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_create_config_preserves_reward_group_low_std_threshold_override():
    trainer = _make_trainer_for_config(save_steps=-1, save_final_model=False)
    trainer._oumi_config.training.verl_config_overrides = {
        "algorithm": {REWARD_GROUP_LOW_STD_CONFIG_KEY: 0.25}
    }

    verl_config = trainer._create_config()

    assert verl_config.algorithm.get(REWARD_GROUP_LOW_STD_CONFIG_KEY) == 0.25


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_setup_installs_grpo_reward_group_metrics_before_trainer_construction():
    trainer = _make_trainer_for_setup("grpo", 0.25)
    events = []

    with (
        patch(
            "oumi.core.trainers.verl_grpo_trainer."
            "install_verl_grpo_reward_group_metrics_patch",
            side_effect=lambda **_: events.append("install"),
        ) as install_patch,
        patch(
            "oumi.core.trainers.verl_grpo_trainer.RayPPOTrainer",
            side_effect=lambda **_: events.append("construct") or MagicMock(),
        ),
        patch(
            "oumi.core.trainers.verl_grpo_trainer.ResourcePoolManager",
            return_value=MagicMock(),
        ),
        patch(
            "oumi.core.trainers.verl_grpo_trainer.ray.remote", side_effect=lambda x: x
        ),
        patch(
            "oumi.core.trainers.verl_grpo_trainer.is_verl_v0_7_or_later",
            return_value=True,
        ),
    ):
        trainer._setup_verl_trainer()

    install_patch.assert_called_once_with(low_std_threshold=0.25)
    assert events == ["install", "construct"]


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_setup_does_not_install_reward_group_metrics_for_other_estimators():
    trainer = _make_trainer_for_setup("gae", 0.25)

    with (
        patch(
            "oumi.core.trainers.verl_grpo_trainer."
            "install_verl_grpo_reward_group_metrics_patch"
        ) as install_patch,
        patch(
            "oumi.core.trainers.verl_grpo_trainer.RayPPOTrainer",
            return_value=MagicMock(),
        ),
        patch(
            "oumi.core.trainers.verl_grpo_trainer.ResourcePoolManager",
            return_value=MagicMock(),
        ),
        patch(
            "oumi.core.trainers.verl_grpo_trainer.ray.remote", side_effect=lambda x: x
        ),
        patch(
            "oumi.core.trainers.verl_grpo_trainer.is_verl_v0_7_or_later",
            return_value=True,
        ),
    ):
        trainer._setup_verl_trainer()

    install_patch.assert_not_called()


class _TrackingWithFileBackend:
    supported_backend = ["console", "wandb", "file"]


class _TrackingWithoutFileBackend:
    supported_backend = ["console", "wandb"]


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_file_logger_mirrors_metrics_under_output_dir(monkeypatch, tmp_path):
    """Metrics are written to verl_metrics.jsonl in the output dir by default."""
    monkeypatch.delenv("VERL_FILE_LOGGER_PATH", raising=False)
    trainer = _make_trainer_for_config(
        save_steps=5, save_final_model=True, output_dir=str(tmp_path)
    )

    with patch.object(verl_grpo_trainer, "VerlTracking", _TrackingWithFileBackend):
        config = trainer._create_config()

    assert "file" in config.trainer.logger
    assert os.environ["VERL_FILE_LOGGER_PATH"] == str(tmp_path / VERL_METRICS_FILENAME)


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_file_logger_respects_preset_path(monkeypatch, tmp_path):
    """A caller-provided VERL_FILE_LOGGER_PATH wins over the default."""
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", "/elsewhere/metrics.jsonl")
    trainer = _make_trainer_for_config(
        save_steps=5, save_final_model=True, output_dir=str(tmp_path)
    )

    with patch.object(verl_grpo_trainer, "VerlTracking", _TrackingWithFileBackend):
        config = trainer._create_config()

    assert "file" in config.trainer.logger
    assert os.environ["VERL_FILE_LOGGER_PATH"] == "/elsewhere/metrics.jsonl"


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_file_logger_skipped_when_verl_lacks_backend(monkeypatch, tmp_path):
    """Older verl without the ``file`` backend keeps the plain logger list."""
    monkeypatch.delenv("VERL_FILE_LOGGER_PATH", raising=False)
    trainer = _make_trainer_for_config(
        save_steps=5, save_final_model=True, output_dir=str(tmp_path)
    )

    with patch.object(verl_grpo_trainer, "VerlTracking", _TrackingWithoutFileBackend):
        config = trainer._create_config()

    assert "file" not in config.trainer.logger
    assert "VERL_FILE_LOGGER_PATH" not in os.environ


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


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
@pytest.mark.parametrize("save_steps", [-1, 0, 100, 500])
@pytest.mark.parametrize("save_final_model", [True, False])
def test_save_epoch_is_effectively_dead_on_verl(save_steps, save_final_model):
    """BUG: `save_epoch=True` collapses every cadence to final-checkpoint-only.

    `_create_config` only copies `save_steps` into verl's `save_freq` when
    `save_epoch` is False, so turning `save_epoch` on leaves `save_freq` at
    verl's `-1` default. The `save_final_model` guard then rewrites that to
    `_SAVE_FINAL_STEP_ONLY_FREQ`, which is deliberately too large for
    `step % save_freq` to ever match. Two things go wrong:

    1. No per-epoch checkpoint is written -- exactly what `save_epoch` asked for.
    2. An explicit `save_steps` is silently dropped, inverting the precedence
       documented on `TrainingParams.save_epoch` ("If both `save_steps` and
       `save_epoch` are set, then `save_steps` takes precedence").
    """
    verl_config = _make_trainer_for_config(
        save_steps, save_final_model=save_final_model, save_epoch=True
    )._create_config()

    # An explicitly requested cadence never reaches verl. (`save_steps <= 0` is
    # not asserted here: it coincides with the `-1` default verl already has.)
    if save_steps > 0:
        assert verl_config.trainer.save_freq != save_steps

    if save_final_model:
        # Too large for `global_step % save_freq` to ever match: final step only.
        assert verl_config.trainer.save_freq > verl_config.trainer.total_training_steps
    else:
        # Fails verl's `save_freq > 0` check: no checkpoint at all.
        assert verl_config.trainer.save_freq <= 0


@pytest.mark.skipif(verl_import_failed, reason="verl not available")
def test_verl_checkpoint_and_export_paths(tmp_path):
    """Documents the on-disk layout verl checkpoints into.

        <output_dir>/                          <- `_final_output_dir`, HF export target
        └── verl_output/                   <- `_temp_output_dir` == verl's
            │                                 `trainer.default_local_dir`
            ├── global_step_5/
            │   └── actor/                 <- FSDP shards; merge source
            │       └── huggingface/       <- config/tokenizer, i.e.
            │                                 `hf_model_config_path`
            └── global_step_10/
                └── actor/
                    └── huggingface/

    `_export_hf_model` picks the `global_step_N` with the largest N and merges
    it up into `<output_dir>` itself, so the exported HF model sits alongside
    the raw verl checkpoints rather than inside them.
    """
    trainer = _make_trainer_for_config(
        save_steps=5, save_final_model=True, output_dir=str(tmp_path)
    )

    # verl is pointed at the nested directory, never at the export directory.
    verl_config = trainer._create_config()
    assert verl_config.trainer.default_local_dir == str(tmp_path / "verl_output")

    for step in (5, 10):
        (tmp_path / "verl_output" / f"global_step_{step}" / "actor").mkdir(parents=True)
    # A bare checkpoint with no `actor/` payload must be ignored, not crash the
    # export by looking like the newest step.
    (tmp_path / "verl_output" / "global_step_20").mkdir()

    with patch(
        "oumi.core.trainers.verl_grpo_trainer.FSDPModelMerger"
    ) as mock_merger_cls:
        assert trainer._export_hf_model() is True

    merger_config = mock_merger_cls.call_args.args[0]
    latest = tmp_path / "verl_output" / "global_step_10"
    assert merger_config.local_dir == str(latest / "actor")
    assert merger_config.hf_model_config_path == str(latest / "actor" / "huggingface")
    assert merger_config.target_dir == str(tmp_path)
    mock_merger_cls.return_value.merge_and_save.assert_called_once()
