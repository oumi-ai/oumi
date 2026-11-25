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
    python -m oumi.verl_grpo.pipeline preprocess -c train.yaml [--init-model DIR]
    (then run the verl command that preprocess prints)
    python -m oumi.verl_grpo.pipeline postprocess -c train.yaml --verl-python PATH
"""

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

import oumi_verl_ext
from oumi.builders.models import build_tokenizer
from oumi.core.configs import DatasetSplit, TrainingConfig
from oumi.verl_grpo.config import merge_verl_config
from oumi.verl_grpo.data import write_verl_parquet
from oumi.verl_grpo.export import export_verl_checkpoint, latest_checkpoint_actor_dir
from oumi.verl_grpo.layout import INDEX_FILENAME, prepare_for_verl


def preprocess(config: TrainingConfig, *, init_model: Path | None = None) -> Path:
    """Writes the parquet files, verl config and extension that verl's run needs.

    Args:
        config: The run's training config.
        init_model: A full fine-tuned model to train from in place of
            `model.model_name`. One lacking that base's tensors or layout trains from
            a filled-out copy under `<training.output_dir>/init_model`.

    Returns:
        The verl config's path, `<training.output_dir>/verl_config.yaml`.
    """
    output_dir = Path(config.training.output_dir).resolve()
    model_path = config.model.model_name
    if init_model is not None:
        model_path = str(
            prepare_for_verl(
                init_model.resolve(),
                _local_model_dir(config),
                output_dir / "init_model",
            )
        )
    tokenizer = build_tokenizer(config.model)
    parquet = {}
    for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION):
        parquet[split] = output_dir / f"{split.value}.parquet"
        write_verl_parquet(
            config.data, split=split, tokenizer=tokenizer, output_path=parquet[split]
        )

    verl_config = merge_verl_config(
        config.training.verl_config_overrides,
        {
            "data": {
                "train_files": [str(parquet[DatasetSplit.TRAIN])],
                "val_files": [str(parquet[DatasetSplit.VALIDATION])],
            },
            "actor_rollout_ref": {"model": {"path": model_path}},
            "trainer": {"default_local_dir": str(output_dir / "checkpoints")},
        },
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

    path = output_dir / "verl_config.yaml"
    body = OmegaConf.to_container(verl_config, resolve=False)
    assert isinstance(body, dict)
    OmegaConf.save({"defaults": ["ppo_trainer", "_self_"], **body}, path)
    return path


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
        text_only=config.model.text_only,
    )


def _local_model_dir(config: TrainingConfig) -> Path:
    """The base model's files: its directory, or its pinned Hub snapshot."""
    name = config.model.model_name
    if Path(name).is_dir():
        return Path(name)
    return Path(
        snapshot_download(
            name,
            revision=config.model.model_revision,
            allow_patterns=["*.json", "*.jinja", "model*.safetensors", INDEX_FILENAME],
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Runs one step from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)
    pre = steps.add_parser("preprocess")
    pre.add_argument("-c", "--config", required=True)
    pre.add_argument("--init-model", type=Path)
    post = steps.add_parser("postprocess")
    post.add_argument("-c", "--config", required=True)
    post.add_argument("--verl-python", required=True)
    args = parser.parse_args(argv)

    config = TrainingConfig.from_yaml(args.config)
    if args.step == "preprocess":
        path = preprocess(config, init_model=args.init_model)
        print(
            f"PYTHONPATH={path.parent / 'ext'} <verl python> -m verl.trainer.main_ppo "
            f"--config-dir {path.parent} --config-name {path.stem}"
        )
    else:
        print(postprocess(config, verl_python=args.verl_python))


if __name__ == "__main__":
    main()
