#!/usr/bin/env python3

# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from argparse import ArgumentParser
import dataclasses
import shutil


def main(model_path: str | Path, ckpt_path: str | Path):
    try:
        from gpt_oss_jax import model as gpt_jax
        from gpt_oss_jax import chkpt_utils as utils
    except ImportError:
        sys.path.append(str(Path(__file__).parents[1].absolute()))

        from gpt_oss_jax import model as gpt_jax
        from gpt_oss_jax import chkpt_utils as utils

    from transformers import AutoConfig
    from safetensors import safe_open
    from tqdm import tqdm

    model_path, ckpt_path = Path(model_path).expanduser(), Path(ckpt_path).expanduser()
    files = list(model_path.glob("*safetensors"))
    assert len(files) > 1
    config_file = model_path / "config.json"
    assert config_file.exists(), (
        "Must have only one `config.json` file in the model path"
    )
    config = AutoConfig.from_pretrained(config_file)
    cfg = gpt_jax.hf_to_jax_config(config)

    # we convert the model unquantized when reading a GPT OSS model checkpoint
    weights = gpt_jax.Weights.abstract(
        dataclasses.replace(cfg, quant_moe=False, quant_attn=False)
    )

    if not ckpt_path.exists():
        model = {}
        for file in tqdm(files):
            with safe_open(file, framework="torch") as f:
                for key in tqdm(f.keys(), leave=False):
                    model[key] = f.get_tensor(key)
        converted_weights = utils.convert_model_or_layer(
            weights, model, cfg, sequential=False
        )
        gpt_jax.save_pytree(converted_weights, ckpt_path)

    additional_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
    ]
    for additional_file in additional_files:
        full_path = model_path / f"{additional_file}"
        if not full_path.exists():
            print(f"Could not find {additional_file}, skipping...")
            continue
        full_path = full_path
        shutil.copyfile(full_path, ckpt_path / full_path.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source-path",
        default="~/gpt-oss-20b",
        required=True,
        help="HF model directory path",
    )
    parser.add_argument(
        "--dest-path",
        default="~/gpt_oss_jax/gpt-oss-20b",
        required=True,
        help="JAX model model directory (to be created).",
    )
    args = parser.parse_args()
    main(args.source_path, args.dest_path)
