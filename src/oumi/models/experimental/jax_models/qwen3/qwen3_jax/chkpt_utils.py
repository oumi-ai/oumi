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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import torch
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from . import model as q3jax


def quantize_model(ckpt_path: Path, quant_ckpt_path: Path):
    ckpt_path, quant_ckpt_path = (
        Path(ckpt_path).expanduser(),
        Path(quant_ckpt_path).expanduser(),
    )
    assert ckpt_path.is_dir()
    cfg = q3jax.load_config(ckpt_path / "config.json")
    mesh = jax.make_mesh((1, jax.device_count(), 1), P("x", "y", "z"))
    cfg = dataclasses.replace(
        cfg, mesh=mesh, quant_moe=True, quant_mlp=True, quant_attn=True
    )

    additional_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    assert all((ckpt_path / file).exists() for file in additional_files)
    print("Loading weights...")
    weights = q3jax.load_pytree(
        ckpt_path,
        q3jax.Weights.shardings(
            dataclasses.replace(cfg, quant_moe=False, quant_mlp=False, quant_attn=False)
        ),
    )

    print("Converting weights...")
    quant_layers = [
        q3jax.Layer.quantize(layer, cfg)
        for layer in tqdm(weights.layers, total=len(weights.layers))
    ]
    quant_weights = dataclasses.replace(weights, layers=quant_layers)

    print("Saving weights...")
    if quant_ckpt_path.exists():
        shutil.rmtree(quant_ckpt_path)
    quant_ckpt_path.parent.mkdir(exist_ok=True)
    q3jax.save_pytree(quant_weights, quant_ckpt_path)

    for file in additional_files:
        shutil.copyfile(ckpt_path / file, quant_ckpt_path / file)


is_leaf = lambda x: isinstance(x, q3jax.ArrayInfo)
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


def _stack_experts(params: dict[str, torch.Tensor]):
    key_fn = lambda x: int(re.match(r"(.*?)experts\.([0-9]+)\..*", x).group(2))
    params = dict(params)
    new_params = dict(params).copy()
    for kw in ["gate", "up", "down"]:
        match = rf"(.*?)experts\.(.*?)\.{kw}_proj\.(.*)"
        match = match.format(kw)
        keys = [k for k in params.keys() if re.match(match, k)]
        prefix_groups = {re.match(match, k).group(1) for k in keys}
        for group_prefix in prefix_groups:
            keys_to_merge = list(
                sorted([k for k in keys if k.startswith(group_prefix)], key=key_fn)
            )
            for k in keys_to_merge:
                del new_params[k]
            suffix = re.match(match, keys_to_merge[0]).group(3)
            new_params[f"{group_prefix}experts.{kw}_proj.{suffix}"] = torch.stack(
                [params[k] for k in keys_to_merge], 0
            )
    return new_params


def convert_weight(key: str, value: torch.Tensor, cfg: q3jax.Config):
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
        assert value.shape == (cfg.embed, cfg.q_heads * cfg.head_dim)
        return t2j(value.T.reshape((cfg.q_heads, cfg.head_dim, cfg.embed)))
    # MoE ######################################################################
    elif re.search(r"gate\.weight", key) is not None:
        assert value.shape == (cfg.moe_num_experts, cfg.embed)
        return t2j(value.T)
    elif re.search(r"experts\.down_proj", key) is not None:
        assert value.shape == (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size)
        return t2j(value.permute((0, 2, 1)))
    elif re.search(r"experts\.(gate|up)_proj", key) is not None:
        assert value.shape == (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed)
        return t2j(value.permute((0, 2, 1)))
    # MLP ######################################################################
    elif re.search(r"down_proj", key) is not None:
        assert value.shape == (cfg.embed, cfg.mlp_ffw_size)
        return t2j(value.T)
    elif re.search(r"(gate|up)_proj", key) is not None:
        assert value.shape == (cfg.mlp_ffw_size, cfg.embed)
        return t2j(value.T)
    # shared misc weights ------------------------------------------------------
    # misc #####################################################################
    elif re.search(r"embed_tokens", key) is not None:
        assert value.shape == (cfg.vocab_size, cfg.embed)
        return t2j(value)
    elif re.search(r"lm_head", key) is not None:
        assert value.shape == (cfg.vocab_size, cfg.embed)
        return t2j(value.T)
    elif re.search(r"(q|k)_norm", key) is not None:
        assert value.shape == (cfg.head_dim,)
        return t2j(value)
    elif re.search(r"layernorm", key) is not None:
        assert value.shape == (cfg.embed,)
        return t2j(value)
    elif re.search(r"norm", key) is not None:
        assert value.shape == (cfg.embed,)
        return t2j(value)
    else:
        raise ValueError(f"Unknown weight {key = }")


_HF_KEY_MAPPING = {
    r"model\.embed_tokens\.weight": "embedding",
    # attention projection weights
    r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"layers.\1.attn.q",
    r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"layers.\1.attn.k",
    r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"layers.\1.attn.v",
    r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"layers.\1.attn.o",
    # norms
    r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": r"layers.\1.attn.q_gamma",
    r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": r"layers.\1.attn.k_gamma",
    # layer norms (pre/post attention)
    r"model\.layers\.([0-9]+)\.input_layernorm\.weight": r"layers.\1.attn_pre_gamma",
    r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"layers.\1.attn_post_gamma",
    # moe router
    r"model\.layers\.([0-9]+)\.mlp\.gate\.weight": r"layers.\1.ffw.w_router",
    # moe experts
    r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_proj\.weight": r"layers.\1.ffw.we_gate",
    r"model\.layers\.([0-9]+)\.mlp\.experts\.up_proj\.weight": r"layers.\1.ffw.we_up",
    r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj\.weight": r"layers.\1.ffw.we_down",
    # mlp
    r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"layers.\1.ffw.w_gate",
    r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"layers.\1.ffw.w_up",
    r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"layers.\1.ffw.w_down",
    r"model\.norm\.weight": "gamma_final",
    r"lm_head\.weight": "lm_head",
}


def _torch_key_to_jax_key(source_key, custom_key_map: dict[str, str] | None = None):
    key_maps = dict(
        _HF_KEY_MAPPING, **(dict() if custom_key_map is None else custom_key_map)
    )
    subs = [
        re.sub(pat, repl, source_key)
        for pat, repl in key_maps.items()
        if re.match(pat, source_key)
    ]
    if len(subs) > 1:
        raise ValueError(f"More than 1 key matched: {subs}")
    else:
        return None if len(subs) == 0 else subs[0]


def _map_weight(
    source_key,
    value: torch.Tensor,
    custom_transform_map: dict[str, Callable] | None = None,
):
    key_maps = dict(
        dict(), **(dict() if custom_transform_map is None else custom_transform_map)
    )
    fns = {pat: fn for pat, fn in key_maps.items() if re.match(pat, source_key)}
    if len(fns) > 1:
        raise ValueError(f"More than 1 key matched: {fns}")
    else:
        return value if len(fns) == 0 else list(fns.values())[0](value)


def convert_model_or_layer(
    layer: q3jax.Weights | q3jax.Layer,
    ref_layer: torch.nn.Module,
    cfg: q3jax.Config,
    device: jax.Device | None = None,
    sequential: bool = True,
    custom_key_map: dict[str, str] | None = None,
    custom_transform_map: dict[str, Callable] | None = None,
    allow_unconverted_parameters: bool = False,
    prefix: str | None = None,
):
    device = device if device is not None else jax.devices("cpu")[0]
    torch_params = dict(
        ref_layer.named_parameters()
        if hasattr(ref_layer, "named_parameters")
        else ref_layer
    )
    torch_params = {
        k: v
        for (k, v) in torch_params.items()
        if prefix is None or k.startswith(prefix)
    }
    torch_params = _stack_experts(torch_params)
    layer_params = {
        ".".join(map(_index_to_str, k)): v
        for (k, v) in jax.tree.flatten_with_path(layer, is_leaf=is_leaf)[0]
    }
    new_params = {k: None for k in layer_params.keys()}

    # some qwen3 checkpoints store both 'embed_tokens' and 'lm_head' even if the embeddings are tied
    # in this case, we check that the weights are identical and delete 'lm_head'
    if cfg.tie_embed and "lm_head.weight" in torch_params:
        torch.testing.assert_close(
            torch_params["lm_head.weight"], torch_params["model.embed_tokens.weight"]
        )
        del torch_params["lm_head.weight"]

    def convert_weight_thread(tkey, tweight):
        with jax.default_device(device):
            jweight = convert_weight(
                tkey,
                _map_weight(tkey, tweight, custom_transform_map=custom_transform_map),
                cfg,
            )
        jkey = _torch_key_to_jax_key(tkey, custom_key_map=custom_key_map)
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
    for (key, param), new_param in zip(layer_params.items(), new_params.values()):
        if param.shape != new_param.shape:
            raise ValueError(
                f"Shape of {key=} does not match, expected = {param.shape}, got {new_param.shape}"
            )

    if isinstance(layer, q3jax.Weights):
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
