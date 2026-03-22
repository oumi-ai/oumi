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


def main(path: str | Path, suffix: str):
    try:
        from llama3_jax import chkpt_utils as utils
    except ImportError:
        sys.path.append(str(Path(__file__).parents[1].absolute()))

        from llama3_jax import chkpt_utils as utils

    path = Path(path).expanduser().absolute()
    dest_path = path.parent / f"{path.name}{suffix}"
    utils.quantize_model(path, dest_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        default="~/meta-llama--Llama-3.1-8B-Instruct",
        required=True,
        help="Existing JAX model checkpoint path",
    )
    parser.add_argument(
        "--suffix",
        default="quant",
        help="Suffix for a new checkpoint directory, e.g., path=~/model, suffix=-quant -> ~/model-quant",
    )

    args = parser.parse_args()
    main(args.path, args.suffix if args.suffix.startswith("-") else f"-{args.suffix}")
