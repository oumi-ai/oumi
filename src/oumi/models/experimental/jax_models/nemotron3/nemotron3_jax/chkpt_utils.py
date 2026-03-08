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

import os
import shutil
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import dataclasses
from collections.abc import Callable

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
import torch
from tqdm import tqdm

from . import model as n3jax


def quantize_model(ckpt_path: Path, quant_ckpt_path: Path):
    ckpt_path, quant_ckpt_path = (
        Path(ckpt_path).expanduser(),
        Path(quant_ckpt_path).expanduser(),
    )
    assert ckpt_path.is_dir()
    cfg = n3jax.load_config(ckpt_path / "config.json")
    mesh = jax.make_mesh((1, jax.device_count(), 1), P("x", "y", "z"))
    cfg = dataclasses.replace(
        cfg,
        mesh=mesh,
        quant_moe=True,
        quant_mlp=True,
        quant_attn=True,
        quant_mamba=True,
    )

    print("Loading weights...")
    weights = n3jax.load_pytree(
        ckpt_path,
        n3jax.Weights.shardings(
            dataclasses.replace(
                cfg,
                quant_moe=False,
                quant_mlp=False,
                quant_attn=False,
                quant_mamba=False,
            )
        ),
    )

    print("Converting weights...")
    quant_layers = [
        n3jax.Layer.quantize(layer, cfg)
        for layer in tqdm(weights.layers, total=len(weights.layers))
    ]
    quant_weights = dataclasses.replace(weights, layers=quant_layers)

    print("Saving weights...")
    if quant_ckpt_path.exists():
        shutil.rmtree(quant_ckpt_path)
    quant_ckpt_path.parent.mkdir(exist_ok=True)
    n3jax.save_pytree(quant_weights, quant_ckpt_path)

    # < 100 MB or so
    for full_path in [
        path
        for path in ckpt_path.glob("*")
        if path.stat().st_size < 100e6 and path.is_file()
    ]:
        if not (quant_ckpt_path / full_path.name).exists():
            shutil.copyfile(full_path, quant_ckpt_path / full_path.name)


is_leaf = lambda x: isinstance(x, n3jax.ArrayInfo)
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


def _format_mamba(params: dict[str, torch.Tensor], cfg: n3jax.Config):
    params = dict(params)
    new_params = dict(params).copy()  # shallow copy only
    x_size = cfg.mamba_num_heads * cfg.mamba_head_dim
    bc_size = cfg.mamba_n_groups * cfg.mamba_ssm_state_size
    dt_size = cfg.mamba_num_heads
    for key in [k for k in params.keys() if re.search(r"in_proj\.weight", k)]:
        w = params[key].mT
        assert w.shape == (cfg.embed, 2 * x_size + 2 * bc_size + dt_size)
        wg, wx, wb, wc, wdt = torch.split(
            w, [x_size, x_size, bc_size, bc_size, dt_size], dim=-1
        )
        wg = wg.reshape((cfg.embed, cfg.mamba_num_heads, cfg.mamba_head_dim))
        wx = wx.reshape((cfg.embed, cfg.mamba_num_heads, cfg.mamba_head_dim))
        wb = wb.reshape((cfg.embed, cfg.mamba_n_groups, cfg.mamba_ssm_state_size))
        wc = wc.reshape((cfg.embed, cfg.mamba_n_groups, cfg.mamba_ssm_state_size))
        wdt = wdt.reshape((cfg.embed, cfg.mamba_num_heads))
        del new_params[key]
        # key_root = ".".join(key.split(".")[:-2])
        key_root = re.match(r"(.*?)in_proj\.weight", key)[1]
        _join_key = lambda k, root=key_root: f"{root}{k}" if root else k
        new_params[_join_key("wg_in")] = wg
        new_params[_join_key("wx_in")] = wx
        new_params[_join_key("wb_in")] = wb
        new_params[_join_key("wc_in")] = wc
        new_params[_join_key("wdt_in")] = wdt
    pat = r"(.*?)(A_log|D|dt_bias)"
    A_log_dt_bias_D_keys = [k for k in params.keys() if re.match(pat, k)]
    prefix_groups = {re.match(pat, k)[1] for k in A_log_dt_bias_D_keys}
    for prefix_group in prefix_groups:
        keys = [k for k in A_log_dt_bias_D_keys if k.startswith(prefix_group)]
        weights = {re.match(pat, k)[2]: params[k] for k in keys}
        assert all(k in weights for k in ["A_log", "D", "dt_bias"]), (
            f"Some keys are missing, found: {weights.keys()}"
        )
        for k in keys:
            del new_params[k]
        new_params[_join_key("A_log_D_dt_bias", root=prefix_group)] = torch.cat(
            [weights[k] for k in ["A_log", "D", "dt_bias"]], -1
        )
    return new_params


def _stack_experts(params: dict[str, torch.Tensor]):
    key_fn = lambda x: int(re.match(r"(.*?)experts\.([0-9]+)\..*", x).group(2))
    params = dict(params)
    new_params = dict(params).copy()
    for kw in ["up", "down"]:
        match = fr"(.*?)experts\.(.*?)\.{kw}_proj\.(.*)"
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


def convert_weight(key: str, value: torch.Tensor, cfg: n3jax.Config):
    x_size = cfg.mamba_num_heads * cfg.mamba_head_dim
    bc_size = cfg.mamba_n_groups * cfg.mamba_ssm_state_size
    value = value.detach()
    # HF checkpoint naming convention ------------------------------------------
    # attention ################################################################
    if re.search(r"q_proj\.weight", key) is not None:
        assert value.shape == (cfg.q_heads * cfg.head_dim, cfg.embed)
        return t2j(
            value.T.reshape(
                (cfg.embed, cfg.kv_heads, cfg.q_heads // cfg.kv_heads, cfg.head_dim)
            )
        )
        # return t2j(value.T.reshape((cfg.embed, cfg.q_heads, cfg.head_dim)))
    elif re.search(r"[kv]_proj\.weight", key) is not None:
        assert value.shape == (cfg.kv_heads * cfg.head_dim, cfg.embed)
        return t2j(value.T.reshape((cfg.embed, cfg.kv_heads, cfg.head_dim)))
    elif re.search(r"o_proj\.weight", key) is not None:
        assert value.shape == (cfg.embed, cfg.q_heads * cfg.head_dim)
        return t2j(
            value.T.reshape(
                (cfg.kv_heads, cfg.q_heads // cfg.kv_heads, cfg.head_dim, cfg.embed)
            )
        )
        # return t2j(value.T.reshape((cfg.q_heads, cfg.head_dim, cfg.embed)))
    # MoE ######################################################################
    elif re.search(r"gate\.weight", key) is not None:
        assert value.shape == (cfg.moe_num_experts, cfg.embed)
        return t2j(value.T)
    elif re.search(r"gate\.e_score_correction_bias", key) is not None:
        assert value.shape == (cfg.moe_num_experts,)
        return t2j(value)
    elif re.search(r"(\.|^)experts\.down_proj", key) is not None:
        assert value.shape == (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size)
        return t2j(value.permute((0, 2, 1)))
    elif re.search(r"(\.|^)experts\.up_proj", key) is not None:
        assert value.shape == (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed)
        return t2j(value.permute((0, 2, 1)))
    elif re.search(r"(\.|^)shared_experts\.down_proj", key) is not None:
        assert value.shape == (cfg.embed, cfg.moe_shared_ffw_size)
        return t2j(value.permute((1, 0)))
    elif re.search(r"(\.|^)shared_experts\.up_proj", key) is not None:
        assert value.shape == (cfg.moe_shared_ffw_size, cfg.embed)
        return t2j(value.permute((1, 0)))
    # shared misc weights ------------------------------------------------------
    elif re.search(r"down_proj", key) is not None:
        assert value.shape == (cfg.embed, cfg.mlp_ffw_size)
        return t2j(value.T)
    elif re.search(r"up_proj", key) is not None:
        assert value.shape == (cfg.mlp_ffw_size, cfg.embed)
        return t2j(value.T)
    # MAMBA ####################################################################
    elif re.search(r"conv1d\.weight", key) is not None:
        assert value.shape == (x_size + 2 * bc_size, 1, cfg.mamba_conv_kernel_size)
        return t2j(value)
    elif re.search(r"conv1d\.bias", key) is not None:
        assert value.shape == (x_size + 2 * bc_size,)
        return t2j(value)
    elif re.search(r"out_proj\.weight", key) is not None:
        assert value.shape == (cfg.embed, cfg.mamba_num_heads * cfg.mamba_head_dim)
        return t2j(
            value.T.reshape((cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.embed))
        )
    elif re.search(r"wg_in", key) is not None:
        assert value.shape == (cfg.embed, cfg.mamba_num_heads, cfg.mamba_head_dim)
        return t2j(value)
    elif re.search(r"wx_in", key) is not None:
        assert value.shape == (cfg.embed, cfg.mamba_num_heads, cfg.mamba_head_dim)
        return t2j(value)
    elif re.search(r"w(b|c)_in", key) is not None:
        assert value.shape == (cfg.embed, cfg.mamba_n_groups, cfg.mamba_ssm_state_size)
        return t2j(value)
    elif re.search(r"wdt_in", key) is not None:
        assert value.shape == (cfg.embed, cfg.mamba_num_heads)
        return t2j(value)
    elif re.search(r"A_log_D_dt_bias", key) is not None:
        assert value.shape == (3 * cfg.mamba_num_heads,)
        return t2j(value)
    # misc #####################################################################
    elif re.search(r"embeddings", key) is not None:
        assert value.shape == (cfg.vocab_size, cfg.embed)
        return t2j(value)
    elif re.search(r"lm_head", key) is not None:
        assert value.shape == (cfg.vocab_size, cfg.embed)
        return t2j(value.T)
    elif re.search(r"norm", key) is not None:
        assert value.shape in [(cfg.embed,), (cfg.mamba_intermediate_size,)]
        return t2j(value)
    else:
        raise ValueError(f"Unknown weight {key = }")


_HF_KEY_MAPPING = {
    # Global Embeddings and Head
    r"backbone\.embeddings\.weight": r"embedding",
    r"backbone\.norm_f\.weight": r"gamma_final",
    r"lm_head\.weight": r"lm_head",
    # Main Layer Norm
    r"backbone\.layers\.(\d+)\.norm\.weight": r"layers.\1.gamma",
    # SSM / Mamba Block Components (e.g., Layers 0, 2, 4...)
    r"backbone\.layers\.(\d+)\.mixer\.conv1d\.weight": r"layers.\1.ffw.w_conv",
    r"backbone\.layers\.(\d+)\.mixer\.conv1d\.bias": r"layers.\1.ffw.b_conv",
    r"backbone\.layers\.(\d+)\.mixer\.norm\.weight": r"layers.\1.ffw.gamma",
    r"backbone\.layers\.(\d+)\.mixer\.out_proj\.weight": r"layers.\1.ffw.w_out",
    r"backbone\.layers\.(\d+)\.mixer\.w([gxbcdt]+)_in": r"layers.\1.ffw.w\2_in",
    r"backbone\.layers\.(\d+)\.mixer\.A_log_D_dt_bias": r"layers.\1.ffw.A_log_D_dt_bias",
    # MoE Block Components (e.g., Layers 1, 3, 6...)
    r"backbone\.layers\.(\d+)\.mixer\.gate\.weight": r"layers.\1.ffw.w_router",
    r"backbone\.layers\.(\d+)\.mixer\.gate\.e_score_correction_bias": r"layers.\1.ffw.b_router",
    r"backbone\.layers\.(\d+)\.mixer\.experts\.up_proj\.weight": r"layers.\1.ffw.we_up",
    r"backbone\.layers\.(\d+)\.mixer\.experts\.down_proj\.weight": r"layers.\1.ffw.we_down",
    r"backbone\.layers\.(\d+)\.mixer\.shared_experts\.up_proj\.weight": r"layers.\1.ffw.ws_up",
    r"backbone\.layers\.(\d+)\.mixer\.shared_experts\.down_proj\.weight": r"layers.\1.ffw.ws_down",
    # Attention Block Components (e.g., Layers 5, 12, 19...)
    r"backbone\.layers\.(\d+)\.mixer\.([qkvo])_proj\.weight": r"layers.\1.ffw.\2",
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
    layer: n3jax.Weights | n3jax.Layer,
    ref_layer: torch.nn.Module,
    cfg: n3jax.Config,
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
    print("Stacking experts")
    torch_params = _stack_experts(torch_params)
    print("Reformatting mamba weights")
    torch_params = _format_mamba(torch_params, cfg)
    layer_params = {
        ".".join(map(_index_to_str, k)): v
        for (k, v) in jax.tree.flatten_with_path(layer, is_leaf=is_leaf)[0]
    }
    new_params = {k: None for k in layer_params.keys()}

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

    if isinstance(layer, n3jax.Weights):
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
