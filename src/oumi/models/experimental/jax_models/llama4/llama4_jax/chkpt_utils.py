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

from . import model as l4jax


def quantize_model(ckpt_path: Path, quant_ckpt_path: Path):
    ckpt_path, quant_ckpt_path = (
        Path(ckpt_path).expanduser(),
        Path(quant_ckpt_path).expanduser(),
    )
    assert ckpt_path.is_dir()
    cfg = l4jax.load_config(ckpt_path / "config.json")
    mesh = jax.make_mesh((1, jax.device_count(), 1), P("x", "y", "z"))
    cfg = dataclasses.replace(
        cfg, mesh=mesh, quant_moe=True, quant_mlp=True, quant_attn=True
    )

    additional_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    assert all((ckpt_path / file).exists() for file in additional_files)
    print("Loading weights...")
    weights = l4jax.load_pytree(
        ckpt_path,
        l4jax.Weights.shardings(
            dataclasses.replace(cfg, quant_moe=False, quant_mlp=False, quant_attn=False)
        ),
    )

    print("Converting weights...")
    quant_layers = [
        l4jax.Layer.quantize(layer, cfg)
        for layer in tqdm(weights.layers, total=len(weights.layers))
    ]
    quant_weights = dataclasses.replace(weights, layers=quant_layers)

    print("Saving weights...")
    if quant_ckpt_path.exists():
        shutil.rmtree(quant_ckpt_path)
    quant_ckpt_path.parent.mkdir(exist_ok=True)
    l4jax.save_pytree(quant_weights, quant_ckpt_path)

    for file in additional_files:
        shutil.copyfile(ckpt_path / file, quant_ckpt_path / file)


# checkpoint conversion from Llama 4

is_leaf = lambda x: isinstance(x, l4jax.ArrayInfo)
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


def convert_weight(key: str, value: torch.Tensor, cfg: l4jax.Config):
    value = value.detach()
    # HF checkpoint naming convention ------------------------------------------
    # attention ################################################################
    if re.search(r"q_proj\.weight", key) is not None:
        assert value.shape == (cfg.q_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.embed, cfg.q_heads, cfg.head_dim)))
    elif re.search(r"[kv]_proj\.weight", key) is not None:
        assert value.shape == (cfg.kv_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.embed, cfg.kv_heads, cfg.head_dim)))
    elif re.search(r"o_proj\.weight", key) is not None:
        assert value.shape == (cfg.q_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.q_heads, cfg.head_dim, cfg.embed)))
    # MLP ######################################################################
    elif re.search(r"feed_forward\.(gate|up)_proj\.weight", key):
        assert value.shape == (cfg.mlp_ffw_size, cfg.embed)
        return t2j(value.T)
    elif re.search(r"feed_forward\.down_proj\.weight", key):
        assert value.shape == (cfg.embed, cfg.mlp_ffw_size)
        return t2j(value.T)
    # MoE ######################################################################
    elif re.search(r"shared_expert\.(gate|up)_proj\.weight", key):
        assert value.shape == (cfg.moe_ffw_size, cfg.embed)
        return t2j(value.T)
    elif re.search(r"shared_expert\.down_proj\.weight", key):
        assert value.shape == (cfg.embed, cfg.moe_ffw_size)
        return t2j(value.T)
    elif re.search(r"router\.weight", key) is not None:
        assert value.shape == (cfg.moe_num_experts, cfg.embed)
        return t2j(value.T)
    elif re.search(r"experts\.down_proj", key) is not None:
        assert value.shape == (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed)
        return t2j(value)
    elif re.search(r"experts\.(gate|up)_proj", key) is not None:
        assert value.shape == (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size)
        return t2j(value)
    # Llama model below --------------------------------------------------------
    # attention ################################################################
    elif re.search(r"wq\.weight", key) is not None:
        assert value.shape == (cfg.q_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.embed, cfg.q_heads, cfg.head_dim)))
    elif re.search(r"w[kv]\.weight", key) is not None:
        assert value.shape == (cfg.kv_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.embed, cfg.kv_heads, cfg.head_dim)))
    elif re.search(r"wo\.weight", key) is not None:
        assert value.shape == (cfg.q_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.q_heads, cfg.head_dim, cfg.embed)))
    # MoE ######################################################################
    elif re.search(r"shared_expert\.w[1-3]\.weight", key):
        assert value.ndim == 2  # not checking the actual shape
        return t2j(value.T)
    elif re.search(r"experts\.w(1|3)", key):
        assert value.shape == (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size)
        return t2j(value)
    elif re.search(r"experts\.w2", key):
        assert value.shape == (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed)
        return t2j(value)
    elif re.search(r"router_DE", key) is not None:
        assert value.shape == (cfg.embed, cfg.moe_num_experts)
        return t2j(value)
    # MLP ######################################################################
    elif re.search(r"w(1|3)\.weight", key) is not None:
        assert value.shape == (cfg.mlp_ffw_size, cfg.embed)
        return t2j(value.T)
    elif re.search(r"w2\.weight", key) is not None:
        assert value.shape == (cfg.embed, cfg.mlp_ffw_size)
        return t2j(value.T)
    # shared misc weights ------------------------------------------------------
    # misc #####################################################################
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


_HF_KEY_MAPPING = {
    # Embedding
    r"language_model\.model\.embed_tokens\.weight": "embedding",
    # Attention Layers
    r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight": r"layers.\1.attn.q",
    r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight": r"layers.\1.attn.k",
    r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight": r"layers.\1.attn.v",
    r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight": r"layers.\1.attn.o",
    # Layer Normalization
    r"language_model\.model\.layers\.(\d+)\.input_layernorm\.weight": r"layers.\1.attn_pre_gamma",
    r"language_model\.model\.layers\.(\d+)\.post_attention_layernorm\.weight": r"layers.\1.attn_post_gamma",
    # MLP Layers
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.gate_proj\.weight": r"layers.\1.mlp.w_gate",
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.up_proj\.weight": r"layers.\1.mlp.w_up",
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.down_proj\.weight": r"layers.\1.mlp.w_down",
    # MoE MLP Layers (Shared Experts)
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.shared_expert\.gate_proj\.weight": r"layers.\1.mlp.ws_gate",
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.shared_expert\.up_proj\.weight": r"layers.\1.mlp.ws_up",
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.shared_expert\.down_proj\.weight": r"layers.\1.mlp.ws_down",
    # MoE MLP Layers (Router)
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.router\.weight": r"layers.\1.mlp.w_router",
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.gate_proj": r"layers.\1.mlp.we_gate",
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.up_proj": r"layers.\1.mlp.we_up",
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.down_proj": r"layers.\1.mlp.we_down",
    # Final Layer Normalization (Assuming source key exists like in the example)
    r"language_model\.model\.norm\.weight": "gamma_final",
    # LM Head
    r"language_model\.lm_head\.weight": "lm_head",
}


def split_gate_up_proj(key: str, weight: torch.Tensor, cfg: l4jax.Config):
    match = re.match(
        r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.gate_up_proj",
        key,
    )
    layer_idx = int(match.group(1))
    assert weight.shape == (cfg.moe_num_experts, cfg.embed, 2 * cfg.moe_ffw_size)
    return {
        f"language_model.model.layers.{layer_idx}.feed_forward.experts.gate_proj": weight[
            ..., : cfg.moe_ffw_size
        ],
        f"language_model.model.layers.{layer_idx}.feed_forward.experts.up_proj": weight[
            ..., cfg.moe_ffw_size :
        ],
    }


_SPECIAL_MAP = {
    r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.gate_up_proj": split_gate_up_proj,
}


def _llama_key_to_jax_key(llama_key, custom_key_map: dict[str, str] | None = None):
    key_maps = [_HF_KEY_MAPPING] + ([] if custom_key_map is None else [custom_key_map])
    for key_map in key_maps:
        for pat, repl in key_map.items():
            m = re.match(pat, llama_key)
            if m is None:
                continue
            return re.sub(pat, repl, llama_key)
    return None


def convert_model_or_layer(
    layer: l4jax.Weights | l4jax.Layer,
    llama_layer: torch.nn.Module,
    cfg: l4jax.Config,
    device: jax.Device | None = None,
    sequential: bool = False,
    custom_key_map: dict[str, str] | None = None,
    allow_unconverted_parameters: bool = False,
    prefix: str | None = None,
):
    device = device if device is not None else jax.devices("cpu")[0]
    torch_params = dict(
        llama_layer.named_parameters()
        if hasattr(llama_layer, "named_parameters")
        else llama_layer
    )
    torch_params = {
        k: v
        for (k, v) in torch_params.items()
        if prefix is None or k.startswith(prefix)
    }

    for tkey in list(torch_params.keys()):
        for pat, fn in _SPECIAL_MAP.items():
            m = re.match(pat, tkey)
            if m is not None:
                new_update = fn(tkey, torch_params[tkey], cfg)
                torch_params.update(new_update)
                del torch_params[tkey]
                print(f"updating {tkey} to {new_update.keys()}")

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
        assert all(v is not None for v in new_params.values()), str(
            {k: v for k, v in new_params.items() if v is None}
        )

    if isinstance(layer, l4jax.Weights):
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
