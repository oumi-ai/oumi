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
from pprint import pprint
from argparse import ArgumentParser
import dataclasses
import shutil


def main(model_path: str | Path, ckpt_path: str | Path):
    try:
        from nemotron3_jax import model as n3jax
        from nemotron3_jax import chkpt_utils as utils
    except ImportError:
        sys.path.append(str(Path(__file__).parents[1].absolute()))

        from nemotron3_jax import model as n3jax
        from nemotron3_jax import chkpt_utils as utils

    from transformers import AutoConfig
    from safetensors import safe_open
    from tqdm import tqdm

    model_path, ckpt_path = Path(model_path).expanduser(), Path(ckpt_path).expanduser()
    files = list(model_path.glob("**/*safetensors"))
    config_files = list(model_path.glob("**/config.json"))
    assert len(config_files) == 1, (
        "Must have only one `config.json` file in the model path"
    )
    config = AutoConfig.from_pretrained(config_files[0].parent, trust_remote_code=True)
    cfg = n3jax.hf_to_jax_config(config)

    weights = n3jax.Weights.abstract(
        dataclasses.replace(cfg, quant_moe=False, quant_mlp=False, quant_attn=False)
    )

    if not ckpt_path.exists():
        model = {}
        for file in tqdm(files):
            with safe_open(file, framework="torch") as f:
                for key in tqdm(f.keys(), leave=False):
                    model[key] = f.get_tensor(key)
        converted_weights = utils.convert_model_or_layer(
            weights, model, cfg, sequential=True
        )
        n3jax.save_pytree(converted_weights, ckpt_path)

    # < 100 MB or so
    for full_path in [
        path
        for path in model_path.glob("*")
        if path.stat().st_size < 100e6 and path.is_file()
    ]:
        shutil.copyfile(full_path, ckpt_path / full_path.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source-path",
        default="~/Nemotron3-30B-A3B-Nano",
        required=True,
        help="HF model directory path",
    )
    parser.add_argument(
        "--dest-path",
        default="~/nemotron3/nemotron3_jax_30b_a3b",
        required=True,
        help="JAX model model directory (to be created).",
    )
    args = parser.parse_args()
    main(args.source_path, args.dest_path)
