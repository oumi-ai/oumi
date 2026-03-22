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
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import torch
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from . import model as l3jax


def quantize_model(ckpt_path: Path, quant_ckpt_path: Path):
    ckpt_path, quant_ckpt_path = (
        Path(ckpt_path).expanduser(),
        Path(quant_ckpt_path).expanduser(),
    )
    assert ckpt_path.is_dir()
    cfg = l3jax.load_config(ckpt_path / "config.json")
    mesh = jax.make_mesh((1, jax.device_count(), 1), P("x", "y", "z"))
    cfg = dataclasses.replace(cfg, mesh=mesh, quant_layer=True)

    additional_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    assert all((ckpt_path / file).exists() for file in additional_files)
    print("Loading weights...")
    weights = l3jax.load_pytree(
        ckpt_path, l3jax.Weights.shardings(dataclasses.replace(cfg, quant_layer=False))
    )

    print("Converting weights...")
    quant_layers = [
        l3jax.Layer.quantize(layer, cfg)
        for layer in tqdm(weights.layers, total=len(weights.layers))
    ]
    quant_weights = dataclasses.replace(weights, layers=quant_layers)

    print("Saving weights...")
    if quant_ckpt_path.exists():
        shutil.rmtree(quant_ckpt_path)
    quant_ckpt_path.parent.mkdir(exist_ok=True)
    l3jax.save_pytree(quant_weights, quant_ckpt_path)

    for file in additional_files:
        shutil.copyfile(ckpt_path / file, quant_ckpt_path / file)


# checkpoint conversion from Llama3

is_leaf = lambda x: isinstance(x, l3jax.ArrayInfo)
j2t = lambda x: torch.from_dlpack(x)


def t2j(x):
    try:
        prev_level, os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
            os.environ.get("TF_CPP_MIN_LOG_LEVEL", None),
            "9",
        )
        return jnp.from_dlpack(x.detach().contiguous())
    finally:
        if prev_level is not None:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = prev_level


def _index_to_str(x):
    """Convert objects from jax.tree.flatten_with_path to dot separated strings."""
    for field in ["name", "idx", "key"]:
        if hasattr(x, field):
            return str(getattr(x, field))
    raise ValueError


def convert_weight(key, value, cfg):
    value = value.detach()
    if re.search(r"q_proj", key) is not None:
        assert value.shape == (cfg.q_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.embed, cfg.q_heads, cfg.head_dim)))
    elif re.search(r"[kv]_proj", key) is not None:
        assert value.shape == (cfg.kv_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.embed, cfg.kv_heads, cfg.head_dim)))
    elif re.search(r"o_proj", key) is not None:
        assert value.shape == (cfg.q_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.q_heads, cfg.head_dim, cfg.embed)))
    elif re.search(r"(up|gate)_proj", key) is not None:
        assert value.shape == (cfg.ffw_size, cfg.embed)
        return t2j(value.T)
    elif re.search(r"down_proj", key) is not None:
        assert value.shape == (cfg.embed, cfg.ffw_size)
        return t2j(value.T)
    elif re.search(r"embed_tokens", key) is not None:
        assert value.shape == (cfg.vocab_size, cfg.embed)
        return t2j(value)
    elif re.search(r"lm_head", key) is not None:
        assert value.shape == (cfg.vocab_size, cfg.embed)
        return t2j(value.T)
    elif re.search(r"norm", key) is not None:
        assert value.shape == (cfg.embed,)
        return t2j(value)
    else:
        raise ValueError(f"Unknown weight key {key = }")


_MODEL_KEY_MAPPING = {
    r"model\.embed_tokens\.weight": "embedding",
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight": r"layers.\1.q",
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight": r"layers.\1.k",
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight": r"layers.\1.v",
    r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight": r"layers.\1.o",
    r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight": r"layers.\1.w_gate",
    r"model\.layers\.(\d+)\.mlp\.up_proj\.weight": r"layers.\1.w_up",
    r"model\.layers\.(\d+)\.mlp\.down_proj\.weight": r"layers.\1.w_down",
    r"model\.layers\.(\d+)\.input_layernorm\.weight": r"layers.\1.attn_pre_gamma",
    r"model\.layers\.(\d+)\.post_attention_layernorm\.weight": r"layers.\1.attn_post_gamma",
    r"model\.norm\.weight": "gamma_final",
    r"lm_head\.weight": "lm_head",
}

_LAYER_KEY_MAPPING = {
    r"self_attn\.q_proj\.weight": r"q",
    r"self_attn\.k_proj\.weight": r"k",
    r"self_attn\.v_proj\.weight": r"v",
    r"self_attn\.o_proj\.weight": r"o",
    r"mlp\.gate_proj\.weight": r"w_gate",
    r"mlp\.up_proj\.weight": r"w_up",
    r"mlp\.down_proj\.weight": r"w_down",
    r"input_layernorm\.weight": r"attn_pre_gamma",
    r"post_attention_layernorm\.weight": r"attn_post_gamma",
}


def _llama_key_to_jax_key(llama_key, custom_key_map: dict[str, str] | None = None):
    key_maps = [_MODEL_KEY_MAPPING, _LAYER_KEY_MAPPING] + (
        [] if custom_key_map is None else [custom_key_map]
    )
    for key_map in key_maps:
        for pat, repl in key_map.items():
            m = re.match(pat, llama_key)
            if m is None:
                continue
            return re.sub(pat, repl, llama_key)
    return None


def convert_model_or_layer(
    layer: l3jax.Weights | l3jax.Layer,
    llama_layer: torch.nn.Module,
    cfg: l3jax.Config,
    device: jax.Device | None = None,
    sequential: bool = False,
    custom_key_map: dict[str, str] | None = None,
    allow_unconverted_parameters: bool = False,
):
    device = device if device is not None else jax.devices("cpu")[0]
    torch_params = dict(
        llama_layer.named_parameters()
        if hasattr(llama_layer, "named_parameters")
        else llama_layer
    )

    layer_params = {
        ".".join(map(_index_to_str, k)): v
        for (k, v) in jax.tree.flatten_with_path(layer, is_leaf=is_leaf)[0]
    }
    new_params = {k: None for k in layer_params.keys()}

    def convert_weight_thread(tkey, tweight):
        with jax.default_device(device):
            jweight = convert_weight(tkey, tweight, cfg)
        jkey = _llama_key_to_jax_key(tkey, custom_key_map=custom_key_map)
        if jkey is None:
            raise ValueError(
                f"Could not find parameter mapping for torch paramter: `{tkey}`."
            )
        if jkey not in new_params:
            raise ValueError(
                f"The JAX model is not expecting `{jkey}`!  Expected keys are {list(new_params.keys())}"
            )
        if new_params[jkey] is not None:
            raise ValueError(f"Parameter `{jkey}` already set!")
        new_params[jkey] = jweight

    if sequential:
        for tkey, tweight in torch_params.items():
            convert_weight_thread(tkey, tweight)
    else:
        futures, executor = [], ThreadPoolExecutor(max_workers=16)
        for tkey, tweight in torch_params.items():
            futures.append(executor.submit(convert_weight_thread, tkey, tweight))
        for fut in tqdm(futures, desc="Converting weights"):
            fut.result()

    if not allow_unconverted_parameters:
        assert all(v is not None for v in new_params.values())
        return jax.tree.unflatten(
            jax.tree.structure(layer, is_leaf=is_leaf), new_params.values()
        )
    else:
        return jax.tree.unflatten(
            jax.tree.structure(layer, is_leaf=is_leaf),
            [
                new_param if new_param is not None else param
                for (new_param, param) in zip(
                    new_params.values(), layer_params.values()
                )
            ],
        )
