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

"""Convert HuggingFace Kimi K2-architecture models to JAX format
Based on jax-llm-examples/kimi_k2/scripts/convert_weights.py
"""

import dataclasses
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path


def main(model_path: str | Path, ckpt_path: str | Path):
    """Convert HuggingFace model to JAX format"""
    try:
        from kimi_k2_jax import chkpt_utils as utils
        from kimi_k2_jax import model as k2jax
    except ImportError:
        # Try relative import for our structure
        sys.path.append(str(Path(__file__).parent.parent.absolute()))
        from kimi_k2_jax import chkpt_utils as utils
        from kimi_k2_jax import model as k2jax

    import jax
    from jax.sharding import AxisType
    from safetensors import safe_open
    from tqdm import tqdm
    from transformers import AutoConfig

    model_path, ckpt_path = Path(model_path).expanduser(), Path(ckpt_path).expanduser()

    # Find safetensors files
    files = list(model_path.glob("**/*safetensors"))
    if not files:
        raise ValueError(f"No safetensors files found in {model_path}")

    print(f"📂 Found {len(files)} safetensors files")

    # Load config
    config_files = list(model_path.glob("**/config.json"))
    if len(config_files) != 1:
        raise ValueError("Must have exactly one `config.json` file in the model path")

    config = AutoConfig.from_pretrained(config_files[0].parent)
    cfg = k2jax.hf_to_jax_config(config.to_dict())

    print(f"📋 Config: {cfg.num_layers} layers, {cfg.vocab_size} vocab")

    # Create mesh
    mesh = jax.make_mesh(
        (1, 1, jax.device_count()),
        ("x", "y", "z"),
        devices=jax.devices(),
        axis_types=(AxisType.Explicit,) * 3,
    )
    cfg = dataclasses.replace(cfg, mesh=mesh, quant_layer=False)

    # Kimi K2 model checkpoints are distributed unquantized
    weights = k2jax.Weights.abstract(cfg)

    if not ckpt_path.exists():
        print("🔄 Converting model weights...")
        model = {}
        for file in tqdm(files, desc="Loading tensors"):
            with safe_open(file, framework="torch") as f:
                for key in tqdm(f.keys(), leave=False):
                    model[key] = f.get_tensor(key)

        print(f"📊 Loaded {len(model)} tensors")

        converted_weights = utils.convert_model_or_layer(
            weights, model, cfg, sequential=False
        )
        k2jax.save_pytree(converted_weights, ckpt_path)

        print(f"✅ Saved JAX weights to {ckpt_path}")

    # Copy additional files
    additional_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for additional_file in additional_files:
        full_paths = list(model_path.glob(f"**/{additional_file}"))
        if len(full_paths) == 1:
            full_path = full_paths[0]
            shutil.copyfile(full_path, ckpt_path / full_path.name)
            print(f"✅ Copied {additional_file}")
        elif len(full_paths) > 1:
            print(f"⚠️  Found multiple {additional_file} files, using first one")
            full_path = full_paths[0]
            shutil.copyfile(full_path, ckpt_path / full_path.name)
        else:
            print(f"⚠️  {additional_file} not found")

    print(f"🎉 Conversion complete! JAX model saved to: {ckpt_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source-path", required=True, help="HF model directory path")
    parser.add_argument(
        "--dest-path",
        required=True,
        help="JAX model directory (to be created).",
    )
    args = parser.parse_args()
    main(args.source_path, args.dest_path)
