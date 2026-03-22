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

import dataclasses
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path


def main(model_path: str | Path, ckpt_path: str | Path):
    try:
        from qwen3_jax import chkpt_utils as utils
        from qwen3_jax import model as q3jax
    except ImportError:
        sys.path.append(str(Path(__file__).parents[1].absolute()))

        from qwen3_jax import chkpt_utils as utils
        from qwen3_jax import model as q3jax

    from safetensors import safe_open
    from tqdm import tqdm
    from transformers import AutoConfig

    model_path, ckpt_path = Path(model_path).expanduser(), Path(ckpt_path).expanduser()
    files = list(model_path.glob("**/*safetensors"))
    assert len(files) >= 1
    config_files = list(model_path.glob("**/config.json"))
    assert len(config_files) == 1, (
        "Must have only one `config.json` file in the model path"
    )
    config = AutoConfig.from_pretrained(config_files[0])
    cfg = q3jax.hf_to_jax_config(config)

    # Qwen 3 model checkpoints are distributed unquantized
    weights = q3jax.Weights.abstract(
        dataclasses.replace(cfg, quant_moe=False, quant_mlp=False, quant_attn=False)
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
        q3jax.save_pytree(converted_weights, ckpt_path)

    additional_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for additional_file in additional_files:
        full_paths = list(model_path.glob(f"**/{additional_file}"))
        if len(full_paths) != 1:
            print(f"Found more than 1 file for {additional_file}")
        if len(full_paths) == 0:
            continue
        full_path = full_paths[0]
        shutil.copyfile(full_path, ckpt_path / full_path.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source-path",
        default="~/Qwen3-30B-A3B",
        required=True,
        help="HF model directory path",
    )
    parser.add_argument(
        "--dest-path",
        default="~/qwen3_jax/Qwen3-30B-A3B",
        required=True,
        help="JAX model model directory (to be created).",
    )
    args = parser.parse_args()
    main(args.source_path, args.dest_path)
