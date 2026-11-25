import json
from pathlib import Path
from unittest.mock import patch

import yaml
from omegaconf import OmegaConf

from oumi.verl_grpo.pipeline import main


def _config(tmp_path: Path) -> Path:
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
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_name": "org/base"},
                "data": {"train": split, "validation": split},
                "training": {
                    "trainer_type": "VERL_GRPO",
                    "output_dir": str(tmp_path / "out"),
                    "verl_config_overrides": {
                        "actor_rollout_ref": {
                            "model": {"external_lib": "my.lib", "lora_rank": 8}
                        }
                    },
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


def test_preprocess_trains_from_an_init_model_filled_from_the_base(tmp_path):
    base, init_model = tmp_path / "base", tmp_path / "parent"
    base.mkdir()
    init_model.mkdir()
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["model"]["model_name"] = str(base)
    config_path.write_text(yaml.safe_dump(config))

    with (
        patch("oumi.verl_grpo.pipeline.build_tokenizer", return_value=None),
        patch("oumi.verl_grpo.pipeline.fill_from_base") as fill,
    ):
        main(["preprocess", "-c", str(config_path), "--init-model", str(init_model)])

    fill.assert_called_once_with(init_model, base)
    verl_config = OmegaConf.load(tmp_path / "out" / "verl_config.yaml")
    assert verl_config.actor_rollout_ref.model.path == str(init_model)
