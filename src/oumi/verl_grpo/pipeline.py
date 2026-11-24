# Copyright 2026 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The steps around a verl 0.9.1 GRPO run: preprocess before it, postprocess after.

Usage:
    python -m oumi.verl_grpo.pipeline preprocess -c train.yaml
    (then run the verl command that preprocess prints)
    python -m oumi.verl_grpo.pipeline postprocess -c train.yaml --verl-python PATH
"""

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from omegaconf import OmegaConf

import oumi_verl_ext
from oumi.builders.models import build_tokenizer
from oumi.core.configs import DatasetSplit, TrainingConfig
from oumi.utils.logging import logger
from oumi.verl_grpo.config import merge_verl_config
from oumi.verl_grpo.data import write_verl_parquet
from oumi.verl_grpo.export import export_verl_checkpoint, latest_checkpoint_actor_dir
from oumi_verl_ext.tool_episode import TOOL_AGENT_LOOP_NAME


def preprocess(config: TrainingConfig) -> Path:
    """Writes the parquet files, verl config and extension that verl's run needs.

    Returns:
        The verl config's path, `<training.output_dir>/verl_config.yaml`.

    Raises:
        ValueError: If the settings name an agent-loop file and a written row uses
            oumi's tool agent loop, which only oumi's agent-loop file registers, or
            if the train split keeps fewer rows than `data.train_batch_size`.
    """
    output_dir = Path(config.training.output_dir).resolve()
    parquet = {
        split: output_dir / f"{split.value}.parquet"
        for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION)
    }
    verl_config = merge_verl_config(
        config.training.verl_config_overrides,
        {
            "data": {
                "train_files": [str(parquet[DatasetSplit.TRAIN])],
                "val_files": [str(parquet[DatasetSplit.VALIDATION])],
            },
            "actor_rollout_ref": {"model": {"path": config.model.model_name}},
            "trainer": {"default_local_dir": str(output_dir / "checkpoints")},
        },
    )
    data = verl_config.data
    multi_turn = verl_config.actor_rollout_ref.rollout.multi_turn
    tool_schemas = (
        _run_tool_schemas(multi_turn.tool_config_path) if multi_turn.enable else None
    )
    tokenizer = build_tokenizer(config.model)
    for split, path in parquet.items():
        write_verl_parquet(
            config.data,
            split=split,
            tokenizer=tokenizer,
            output_path=path,
            multi_turn=multi_turn.enable,
            max_prompt_length=data.max_prompt_length,
            chat_template_kwargs=data.apply_chat_template_kwargs,
            tool_schemas=tool_schemas,
            # Fewer train rows than one batch give verl no steps.
            min_rows=data.train_batch_size if split == DatasetSplit.TRAIN else 1,
        )

    # verl imports these in its processes; oumi's adds the GRPO metrics.
    model = verl_config.actor_rollout_ref.model
    libs = model.external_lib
    model.external_lib = [
        *([] if libs is None else [libs] if isinstance(libs, str) else libs),
        "oumi_verl_ext.install",
    ]
    shutil.copytree(
        oumi_verl_ext.__path__[0],
        output_dir / "ext" / "oumi_verl_ext",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    # verl loads a single agent-loop file.
    agent = verl_config.actor_rollout_ref.rollout.agent
    if not agent.agent_loop_config_path:
        agent.agent_loop_config_path = str(
            output_dir / "ext" / "oumi_verl_ext" / "agent_loops.yaml"
        )
    elif any(map(_uses_tool_agent_loop, parquet.values())):
        raise ValueError(
            "actor_rollout_ref.rollout.agent.agent_loop_config_path is set, but rows "
            f"use the '{TOOL_AGENT_LOOP_NAME}' agent loop, which only oumi's "
            "agent-loop file registers. Leave the setting unset."
        )
    else:
        logger.info(
            "Keeping the configured agent_loop_config_path; oumi's agent loops are "
            "not registered."
        )

    path = output_dir / "verl_config.yaml"
    body = OmegaConf.to_container(verl_config, resolve=False)
    assert isinstance(body, dict)
    OmegaConf.save({"defaults": ["ppo_trainer", "_self_"], **body}, path)
    return path


def _run_tool_schemas(tool_config_path: str | None) -> list[dict[str, Any]] | None:
    """The tool schemas in a verl tool config, read as verl's tool registry does."""
    if not tool_config_path:
        return None
    try:
        tools = OmegaConf.load(tool_config_path).tools
    except (OSError, yaml.YAMLError) as error:
        logger.warning(
            "Measuring prompts without tools: can't read "
            f"actor_rollout_ref.rollout.multi_turn.tool_config_path ({error})."
        )
        return None
    # A tool without a tool_schema brings its own, which preprocess can't see.
    return [  # type: ignore[return-value]
        OmegaConf.to_container(tool.tool_schema, resolve=True)
        for tool in tools
        if tool.get("tool_schema") is not None
    ]


def _uses_tool_agent_loop(parquet_path: Path) -> bool:
    if "agent_name" not in pq.read_schema(parquet_path).names:
        return False
    agent_names = pq.read_table(parquet_path, columns=["agent_name"])["agent_name"]
    return TOOL_AGENT_LOOP_NAME in agent_names.to_pylist()


def postprocess(config: TrainingConfig, *, verl_python: str) -> Path:
    """Exports the run's latest checkpoint to `<training.output_dir>/export`."""
    output_dir = Path(config.training.output_dir).resolve()
    verl_config = OmegaConf.load(output_dir / "verl_config.yaml")
    model = verl_config.actor_rollout_ref.model
    return export_verl_checkpoint(
        latest_checkpoint_actor_dir(Path(verl_config.trainer.default_local_dir)),
        output_dir / "export",
        base_model=model.path,
        verl_python=verl_python,
        exclude_modules=model.exclude_modules,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Runs one step from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("preprocess").add_argument("-c", "--config", required=True)
    post = steps.add_parser("postprocess")
    post.add_argument("-c", "--config", required=True)
    post.add_argument("--verl-python", required=True)
    args = parser.parse_args(argv)

    config = TrainingConfig.from_yaml(args.config)
    if args.step == "preprocess":
        path = preprocess(config)
        print(
            f"PYTHONPATH={path.parent / 'ext'} <verl python> -m verl.trainer.main_ppo "
            f"--config-dir {path.parent} --config-name {path.stem}"
        )
    else:
        print(postprocess(config, verl_python=args.verl_python))


if __name__ == "__main__":
    main()
