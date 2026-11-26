import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml
from omegaconf import OmegaConf

from oumi.verl_grpo.pipeline import main
from oumi_verl_ext.tool_episode import TOOL_AGENT_LOOP_NAME


def _config(tmp_path: Path, multi_turn: bool = False) -> Path:
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        )
        + "\n"
    )
    split = {
        "datasets": [
            {
                "dataset_name": "text_sft",
                "dataset_path": str(data),
                "dataset_kwargs": {"return_conversations": True},
            }
        ]
    }
    overrides = {
        "model": {"external_lib": "my.lib", "lora_rank": 8},
        "rollout": {"multi_turn": {"enable": multi_turn}},
    }
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_name": "org/base"},
                "data": {"train": split, "validation": split},
                "training": {
                    "trainer_type": "VERL_GRPO",
                    "output_dir": str(tmp_path / "out"),
                    "verl_config_overrides": {"actor_rollout_ref": overrides},
                },
            }
        )
    )
    return path


def test_preprocess_writes_the_files_verl_needs(tmp_path):
    with patch("oumi.verl_grpo.pipeline.build_tokenizer", return_value=None):
        main(["preprocess", "-c", str(_config(tmp_path))])

    out = tmp_path / "out"
    config = OmegaConf.load(out / "verl_config.yaml")
    assert list(config.defaults) == ["ppo_trainer", "_self_"]
    assert list(config.data.train_files) == [str(out / "train.parquet")]
    assert list(config.data.val_files) == [str(out / "validation.parquet")]
    assert (out / "train.parquet").is_file() and (out / "validation.parquet").is_file()
    model = config.actor_rollout_ref.model
    assert model.path == "org/base"
    assert model.lora_rank == 8
    assert list(model.external_lib) == ["my.lib", "oumi_verl_ext.install"]
    assert config.trainer.default_local_dir == str(out / "checkpoints")
    assert (out / "ext/oumi_verl_ext/install.py").is_file()


def _preprocess(config_path: Path):
    with patch("oumi.verl_grpo.pipeline.build_tokenizer", return_value=None):
        main(["preprocess", "-c", str(config_path)])
    return OmegaConf.load(config_path.parent / "out" / "verl_config.yaml")


def test_preprocess_runs_multi_turn_rows_in_the_tool_loop(tmp_path):
    config = _preprocess(_config(tmp_path, multi_turn=True))

    path = Path(config.actor_rollout_ref.rollout.agent.agent_loop_config_path)
    assert path == tmp_path / "out/ext/oumi_verl_ext/agent_loops.yaml"
    registered = {loop["name"] for loop in yaml.safe_load(path.read_text())}
    assert TOOL_AGENT_LOOP_NAME in registered
    rows = pd.read_parquet(tmp_path / "out/train.parquet")
    assert set(rows["agent_name"]) == {TOOL_AGENT_LOOP_NAME}


def test_preprocess_leaves_single_turn_runs_unregistered(tmp_path):
    config = _preprocess(_config(tmp_path))

    assert config.actor_rollout_ref.rollout.agent.agent_loop_config_path is None


def test_postprocess_exports_the_latest_checkpoint(tmp_path):
    config_path = _config(tmp_path)
    with patch("oumi.verl_grpo.pipeline.build_tokenizer", return_value=None):
        main(["preprocess", "-c", str(config_path)])
    checkpoints = tmp_path / "out" / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "latest_checkpointed_iteration.txt").write_text("6")

    with patch("oumi.verl_grpo.pipeline.export_verl_checkpoint") as export:
        main(["postprocess", "-c", str(config_path), "--verl-python", "py"])

    export.assert_called_once_with(
        checkpoints / "global_step_6" / "actor",
        tmp_path / "out" / "export",
        base_model="org/base",
        verl_python="py",
        exclude_modules=None,
        text_only=False,
    )


def test_preprocess_trains_from_the_init_model_prepared_for_verl(tmp_path):
    base, init_model = tmp_path / "base", tmp_path / "parent"
    base.mkdir()
    init_model.mkdir()
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["model"]["model_name"] = str(base)
    config_path.write_text(yaml.safe_dump(config))

    with (
        patch("oumi.verl_grpo.pipeline.build_tokenizer", return_value=None),
        patch(
            "oumi.verl_grpo.pipeline.prepare_for_verl", return_value=tmp_path / "ready"
        ) as prepare,
    ):
        main(["preprocess", "-c", str(config_path), "--init-model", str(init_model)])

    prepare.assert_called_once_with(init_model, base, tmp_path / "out" / "init_model")
    verl_config = OmegaConf.load(tmp_path / "out" / "verl_config.yaml")
    assert verl_config.actor_rollout_ref.model.path == str(tmp_path / "ready")
