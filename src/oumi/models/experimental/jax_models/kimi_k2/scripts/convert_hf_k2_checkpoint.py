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
import json
import gzip
from pathlib import Path
from argparse import ArgumentParser

import jax
from jax.sharding import PartitionSpec as P


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--root-dir", required=True, help="Directory with *.safetensors files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Name of the output directory"
    )
    args = parser.parse_args()

    from kimi_k2_jax.model import ShardingRules, Config
    from kimi_k2_jax import chkpt_utils as utils

    root_path, dest_path = Path(args.root_dir), Path(args.output_dir)

    cfg = Config()
    cfg.quantize_mlp = False
    cfg.quantize_attn = True
    cfg.quantize_moe = True

    rules = ShardingRules(
        *(None for _ in dataclasses.fields(ShardingRules))
    )  # fully replicated
    cfg = dataclasses.replace(cfg, mesh=jax.make_mesh((1,), P("x")), rules=rules)
    params_map = json.loads(
        gzip.decompress(
            (
                Path(__file__).parent.absolute() / "k2_hf_ckpt_params_map.json.gz"
            ).read_bytes()
        )
    )
    utils.convert_hf_checkpoint(params_map, root_path, dest_path, cfg)


if __name__ == "__main__":
    main()
