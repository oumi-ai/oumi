import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

from oumi.verl_grpo.pipeline import main


def _config(tmp_path: Path, agent_loop_config_path: str | None = None) -> Path:
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
    overrides: dict = {"model": {"external_lib": "my.lib", "lora_rank": 8}}
    if agent_loop_config_path:
        overrides["rollout"] = {
            "agent": {"agent_loop_config_path": agent_loop_config_path}
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


def test_preprocess_registers_the_tool_loop(tmp_path):
    config = _preprocess(_config(tmp_path))

    path = Path(config.actor_rollout_ref.rollout.agent.agent_loop_config_path)
    assert path == tmp_path / "out/ext/oumi_verl_ext/agent_loops.yaml"
    registered = {loop["name"] for loop in yaml.safe_load(path.read_text())}
    agent_names = set(pd.read_parquet(tmp_path / "out/train.parquet")["agent_name"])
    assert agent_names <= registered | {"single_turn_agent", "tool_agent"}


def _writes(columns: dict):
    """Stands in for write_verl_parquet, writing one row of `columns`."""

    def write_verl_parquet(data_params, *, split, tokenizer, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns).to_parquet(output_path)

    return patch("oumi.verl_grpo.pipeline.write_verl_parquet", write_verl_parquet)


def test_preprocess_refuses_a_user_agent_loop_config_path(tmp_path):
    config_path = _config(tmp_path, agent_loop_config_path="/my/loops.yaml")

    with (
        _writes({"agent_name": ["oumi_tool_agent"]}),
        pytest.raises(ValueError, match="agent_loop_config_path"),
    ):
        _preprocess(config_path)


@pytest.mark.parametrize(
    "columns", [{"agent_name": ["single_turn_agent"]}, {"data_source": ["verl"]}]
)
def test_preprocess_keeps_a_user_agent_loop_config_path_without_tool_rows(
    tmp_path, caplog, columns
):
    with _writes(columns):
        config = _preprocess(_config(tmp_path, agent_loop_config_path="/my/loops.yaml"))

    assert config.actor_rollout_ref.rollout.agent.agent_loop_config_path == (
        "/my/loops.yaml"
    )
    assert "Keeping the configured agent_loop_config_path" in caplog.text


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
    )
