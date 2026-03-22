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

from argparse import ArgumentParser
from pathlib import Path

example_models = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B",
]


def main(model_id: str, dest_root_path: str | Path):
    from huggingface_hub import snapshot_download

    local_dir = Path(dest_root_path).expanduser().absolute() / str(model_id).replace(
        "/", "--"
    )
    snapshot_download(repo_id=model_id, local_dir=local_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-id",
        required=True,
        help=f"HuggingFace model / repo id. Examples include: {example_models}",
    )
    parser.add_argument(
        "--dest-root-path",
        required=True,
        default="~/",
        help="Destination root directory, the model will be saved into its own directory.",
    )
    args = parser.parse_args()
    main(args.model_id, args.dest_root_path)
