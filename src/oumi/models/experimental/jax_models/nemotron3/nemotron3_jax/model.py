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

"""Minimal model definition."""

import dataclasses
import os
import json
from pathlib import Path
import math
from functools import partial, lru_cache
from typing import Any
from collections.abc import Callable
from collections import OrderedDict as odict

import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
from jax.sharding import PartitionSpec as P, auto_axes, reshard
from jax.experimental.array_serialization import pytree_serialization as ser
from jax.experimental.layout import Format, Layout

PAD_ID = 0  # thankfully, a sane value like 0

AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]

# Expected physical mesh axis names:
# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
EXPERT_AXIS_NAME = "z"
TENSOR_ONLY_AXIS_NAME = "y"
ATTN_HEADS_AXIS_NAME = "y"
TENSOR_AXIS_NAME = ("y", "z")


@dataclasses.dataclass
class ShardingRules:
    """Mapping from logical data axes to physical mesh axes.

    To manage the different shardings in the model, we define the "logical"
    dimensions of various arrays (each dimension for each layer's weights,
    etc.). Each of these logical axes may then be sharded over a physical mesh
    axis, i.e. over multiple devices. For example, any values with a batch
    dimension should always be sharded over the batch axis of the mesh.

    Defining the shardings this way allows us to easily try out new sharding
    strategies by just changing this mapping. The rest of the code handles
    taking this mapping and eventually turning it into the correct JAX shardings
    and sharding contraints.
    """

    batch: AxisName = BATCH_AXIS_NAME
    sequence: AxisName = None
    act_embed: AxisName = None
    act_heads: AxisName = None
    head_dim: AxisName = None
    # attention
    qkv_embed: AxisName = None
    q_heads: AxisName = TENSOR_AXIS_NAME
    kv_heads: AxisName = None  # too few attention heads in nemotron to shard them
    o_heads: AxisName = TENSOR_AXIS_NAME
    o_embed: AxisName = None
    # MLP layer
    mlp_up_embed: AxisName = None
    mlp_up_ffw: AxisName = TENSOR_AXIS_NAME
    mlp_down_ffw: AxisName = TENSOR_AXIS_NAME
    mlp_down_embed: AxisName = None
    # MoE layer
    moe_e_experts: AxisName = EXPERT_AXIS_NAME
    moe_e_up_embed: AxisName = None
    moe_e_up_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_e_down_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_e_down_embed: AxisName = None
    moe_e_tp: AxisName = (
        TENSOR_ONLY_AXIS_NAME  # moe forward function tensor parallelism
    )
    moe_e_ep: AxisName = EXPERT_AXIS_NAME  # moe forward function expert parallelism
    moe_s_up_embed: AxisName = None
    moe_s_up_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_s_down_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_s_down_embed: AxisName = None
    # Mamba layer
    mamba_in_embed: AxisName = None
    mamba_num_heads: AxisName = TENSOR_AXIS_NAME
    mamba_n_groups: AxisName = ATTN_HEADS_AXIS_NAME
    mamba_head_dim: AxisName = None
    mamba_ssm_state_dim: AxisName = None
    mamba_out_embed: AxisName = None
    # vocab
    vocab_in: AxisName = None
    vocab_out: AxisName = TENSOR_AXIS_NAME


def logical_to_physical(
    logical: Axes, rules: ShardingRules
) -> jax.sharding.PartitionSpec:
    """Returns how to physically shard a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    spec = [getattr(rules, axis) if axis is not None else None for axis in logical]
    # `spec` may contain tuples, flatten to check that `spec` maps each physical mesh axis to at most one logical axis.
    flat_axes = jax.tree.leaves(spec)
    if len(set(flat_axes)) != len(flat_axes):
        raise ValueError(
            f"Colliding physical axes from translating logical spec {logical} -> {spec}"
        )
    return P(*spec)


def logical_to_sharding(
    logical: Axes, mesh: jax.sharding.Mesh, rules: ShardingRules
) -> jax.sharding.Sharding:
    """Returns the sharding for a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    assert mesh is not None
    return jax.sharding.NamedSharding(mesh, logical_to_physical(logical, rules))


def jax_pytree_struct(cls, meta_fields: tuple = ()):
    """jax.tree_util.register_dataclass wrapper that automatically infers data_fields."""
    if not dataclasses.is_dataclass(cls):
        cls = dataclasses.dataclass(cls)
    all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)
    return tree_util.register_dataclass(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )


jax_static = lambda cls: tree_util.register_static(dataclasses.dataclass(cls))


@jax_static
class Config:
    embed: int
    q_heads: int
    kv_heads: int
    num_layers: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    layer_pattern: str
    # Attention
    causal: bool
    # MoE
    moe_ffw_size: int
    moe_experts_per_tok: int
    moe_num_experts: int
    moe_shared_ffw_size: int
    moe_router_n_groups: int
    moe_router_topk_groups: int
    moe_routed_scaling_factor: float
    ep_strategy: str
    moe_gate_dtype: "jnp.dtype"
    # mamba
    mamba_intermediate_size: int
    mamba_conv_dim: int
    mamba_num_heads: int
    mamba_conv_kernel_size: int
    mamba_n_groups: int
    mamba_ssm_state_size: int
    mamba_time_step_limit: tuple[int, int]
    mamba_head_dim: int
    mamba_chunk_size: int
    # kernel config
    use_prefill_attn_kernel: bool = False
    use_decode_attn_kernel: bool = False
    use_ragged_dot_kernel: bool = False
    dtype: "jnp.dtype" = jnp.bfloat16
    mamba_dtype: "jnp.dtype" = jnp.float32
    norm_eps: float = 1e-6
    # sharding
    rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
    mesh: jax.sharding.Mesh | None = None
    rope_theta: float = 500000.0
    quant_moe: bool = False
    quant_mamba: bool = False
    quant_mlp: bool = False
    quant_attn: bool = False
    quant_cache: bool = True
    quant_scale_dtype: "jnp.dtype" = jnp.bfloat16


def hf_to_jax_config(hf_config: Any | dict[str, Any]) -> "Config":
    _get = lambda x, k, default=None: (
        getattr(x, k, default)
        if not isinstance(hf_config, dict)
        else hf_config.get(k, default)
    )
    assert _get(hf_config, "n_shared_experts") == 1
    cfg = Config(
        embed=_get(hf_config, "hidden_size"),
        moe_ffw_size=_get(hf_config, "moe_intermediate_size", -1),
        layer_pattern=_get(hf_config, "hybrid_override_pattern", []),
        mamba_conv_dim=_get(hf_config, "conv_kernel"),
        mamba_num_heads=_get(hf_config, "mamba_num_heads"),
        mamba_conv_kernel_size=_get(hf_config, "conv_kernel"),
        mamba_n_groups=_get(hf_config, "n_groups"),
        mamba_ssm_state_size=_get(hf_config, "ssm_state_size"),
        mamba_time_step_limit=_get(hf_config, "time_step_limit"),
        mamba_head_dim=_get(hf_config, "mamba_head_dim"),
        mamba_chunk_size=_get(hf_config, "chunk_size"),
        q_heads=_get(hf_config, "num_attention_heads"),
        kv_heads=_get(hf_config, "num_key_value_heads"),
        num_layers=_get(hf_config, "num_hidden_layers"),
        head_dim=_get(hf_config, "head_dim"),
        vocab_size=_get(hf_config, "vocab_size"),
        norm_eps=_get(hf_config, "layer_norm_epsilon"),
        moe_experts_per_tok=_get(hf_config, "num_experts_per_tok"),
        moe_num_experts=_get(hf_config, "n_routed_experts"),
        moe_router_n_groups=_get(hf_config, "n_group"),
        moe_router_topk_groups=_get(hf_config, "topk_group"),
        moe_routed_scaling_factor=_get(hf_config, "routed_scaling_factor"),
        moe_shared_ffw_size=_get(hf_config, "moe_shared_expert_intermediate_size"),
        max_seq_len=128,
        dtype=jnp.bfloat16,
        causal=True,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
        rope_theta=_get(hf_config, "rope_theta"),
        moe_gate_dtype=jnp.float32,
        ep_strategy="decode",
        # to be derived
        mamba_intermediate_size=-1,
    )
    cfg.mamba_intermediate_size = cfg.mamba_num_heads * cfg.mamba_head_dim
    return cfg


def load_config(config_path: str | os.PathLike[str] | Path) -> "Config":
    return hf_to_jax_config(json.loads(Path(config_path).read_text()))


def load_tokenizer(path: str | os.PathLike[str] | Path) -> "PreTrainedTokenizerFast":  # noqa: F821
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(Path(path).absolute(), local_files_only=True)


@partial(jax_pytree_struct, meta_fields=("shape", "logical_axes", "initializer"))
@dataclasses.dataclass(frozen=True)
class ArrayInfo:
    shape: tuple[int, ...]
    dtype: "jnp.dtype"
    logical_axes: tuple
    initializer: Callable | None = None


# module reload friendly isinstance check
is_type = lambda x, cls: (
    (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
)
is_param = lambda x: is_type(x, ArrayInfo)
_count_left_padding = lambda ids, pad_id=PAD_ID: auto_axes(
    lambda ids: jnp.sum(jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1),
    out_sharding=P(None),
)(ids)
########################################################################################################################
_count_right_padding = lambda ids, pad_id=PAD_ID: auto_axes(
    lambda ids: jnp.sum(
        jnp.cumsum(jnp.flip(ids, axis=-1) != pad_id, axis=-1) == 0, axis=-1
    ),
    out_sharding=P(None),
)(ids)
_length_minus_right_padding = lambda segment_ids, pad_id=PAD_ID: auto_axes(
    lambda segment_ids: jnp.sum(
        jnp.cumsum(jnp.flip(segment_ids != pad_id, -1), axis=-1) > 0, -1
    ),
    out_sharding=P(None),
)(segment_ids)
which_platform = lambda cfg: cfg.mesh.devices.reshape(-1)[0].platform
_he_normal = lru_cache(jax.nn.initializers.he_normal)
_const_init = lru_cache(jax.nn.initializers.constant)


@partial(jax.jit, static_argnames=("abstract", "shardings"))
def _init_leaves(key, abstract, shardings):
    @partial(jax.jit, out_shardings=shardings)
    def _init_fn(key):
        num_leaves = len(
            jax.tree.leaves(abstract, is_leaf=is_param)
        )  # one new RNG key per tensor
        key_iter = iter(random.split(key, num_leaves))
        return jax.tree.map(
            lambda info: info.initializer(next(key_iter), info.shape, info.dtype),
            abstract,
            is_leaf=is_param,
        )

    return _init_fn(key)


class _Init:
    @classmethod
    def abstract(cls, cfg: Config, *args, **kw):
        raise NotImplementedError

    @classmethod
    def shardings(cls, cfg: Config, *args, **kw):
        abstract = cls.abstract(cfg, *args, **kw)
        return jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )

    @classmethod
    def init(cls, key: random.PRNGKey, cfg: Config, *args, **kw):
        """Returns a pytree of randomly-initialized jax.Arrays corresponding to abstract()."""
        abstract = cls.abstract(cfg, *args, **kw)
        shardings = jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract,
            is_leaf=is_param,
        )
        abstract_leaves, abstract_struct = jax.tree.flatten(abstract, is_leaf=is_param)
        shardings_leaves = jax.tree.leaves(shardings, is_leaf=is_param)
        ret = jax.tree.unflatten(
            abstract_struct,
            _init_leaves(key, tuple(abstract_leaves), tuple(shardings_leaves)),
        )
        return jax.device_put(ret, shardings)


@partial(jax_pytree_struct, meta_fields=("out_scaling", "scale_expand_dims"))
class QuantArray:
    quant: jax.Array | ArrayInfo
    scale: jax.Array | ArrayInfo
    out_scaling: bool = False
    scale_expand_dims: int | tuple[int, ...] = ()
    shape = property(lambda self: self.quant.shape)
    ndim = property(lambda self: self.quant.ndim)


def einsum(
    subscripts: str,
    lhs: jax.Array,
    rhs: jax.Array | QuantArray,
    out_sharding: P | None = None,
):
    """jnp.einsum wrapper that handles regular arrays and QuantArrays"""
    if is_type(rhs, QuantArray):
        scale = jnp.expand_dims(rhs.scale, rhs.scale_expand_dims)
        if rhs.out_scaling:
            return (
                jnp.einsum(subscripts, lhs, rhs.quant, out_sharding=out_sharding)
                * scale
            )
        else:
            return jnp.einsum(
                subscripts, lhs * scale, rhs.quant, out_sharding=out_sharding
            )
    else:
        return jnp.einsum(subscripts, lhs, rhs, out_sharding=out_sharding)


_int8_quant_init = lambda key, shape, dtype=jnp.int8: random.randint(
    key, shape, -128, 128, dtype=dtype
)
_int8_scale_init = lambda key, shape, dtype: (
    random.normal(key, shape, dtype=dtype) / math.sqrt(math.prod(shape)) / 127
)


def quantize(
    x: jax.Array | ArrayInfo,
    axis: int | tuple[int, ...],
    scale_dtype=jnp.bfloat16,
    zero_init: bool = False,
):
    if is_type(x, QuantArray):
        raise ValueError("Attempting to quantize an already quantized QuantArray.")
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)
    axis = tuple(z % len(x.shape) for z in axis)

    if isinstance(x, jax.Array):
        axis = tuple(z % x.ndim for z in axis)
        amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
        scale = (amax / 127.0 + jnp.finfo(scale_dtype).tiny).astype(scale_dtype)
        quant = jnp.round(x / scale).astype(jnp.int8)
        scale = scale.reshape([z for i, z in enumerate(scale.shape) if i not in axis])
        return quant, scale

    if is_type(x, ArrayInfo):
        new_shape = tuple(ax for i, ax in enumerate(x.shape) if i not in axis)
        new_logical_axes = tuple(
            ax for i, ax in enumerate(x.logical_axes) if i not in axis
        )
        if zero_init:
            quant_init, scale_init = jax.nn.initializers.zeros, jax.nn.initializers.ones
        else:
            quant_init, scale_init = _int8_quant_init, _int8_scale_init
        quant = dataclasses.replace(
            x, shape=x.shape, dtype=jnp.int8, initializer=quant_init
        )
        scale = ArrayInfo(new_shape, scale_dtype, new_logical_axes, scale_init)
        return quant, scale
    raise ValueError(f"quantize got unexpected type: {type(x)}")


def update_slice(
    x: jax.Array | QuantArray,
    y: jax.Array,
    pos: int,
    update_axis: int,
    quant_axis: int = -1,
):
    """dynamic_update_slice wrapper that handles regular arrays and QuantArrays"""
    if is_type(x, QuantArray):
        assert x.quant.ndim == y.ndim
        quant_axis, update_axis = (
            quant_axis % x.quant.ndim,
            update_axis % x.quant.ndim,
        )  # normalize axis numbers
        y_quant, y_scale = quantize(
            y, axis=quant_axis, scale_dtype=x.scale.dtype
        )  # quantize rhs
        y_quant = reshard(
            y_quant.astype(x.quant.dtype), jax.typeof(x.quant).sharding.spec
        )
        y_scale = reshard(
            y_scale.astype(x.scale.dtype), jax.typeof(x.scale).sharding.spec
        )
        new_quant = jax.lax.dynamic_update_slice_in_dim(
            x.quant, y_quant, pos, axis=update_axis
        )
        scale_update_axis = [ax for ax in range(x.quant.ndim) if ax != quant_axis][
            update_axis
        ]
        new_scale = jax.lax.dynamic_update_slice_in_dim(
            x.scale, y_scale, pos, axis=scale_update_axis
        )  # update axis in `scale`
        return dataclasses.replace(x, quant=new_quant, scale=new_scale)
    else:
        assert x.ndim == y.ndim
        y = reshard(y.astype(x.dtype), jax.typeof(x).sharding.spec)
        return jax.lax.dynamic_update_slice_in_dim(x, y, pos, axis=update_axis)


@jax_pytree_struct
class AttentionLayer(_Init):
    q: jax.Array | ArrayInfo | QuantArray
    k: jax.Array | ArrayInfo | QuantArray
    v: jax.Array | ArrayInfo | QuantArray
    o: jax.Array | ArrayInfo | QuantArray

    ########################################################################################################################
    @classmethod
    def abstract(cls, cfg: Config) -> "AttentionLayer":
        _init = _he_normal
        layer = AttentionLayer(
            q=ArrayInfo(
                (cfg.embed, cfg.kv_heads, cfg.q_heads // cfg.kv_heads, cfg.head_dim),
                cfg.dtype,
                ("qkv_embed", "kv_heads", "q_heads", "head_dim"),
                _init(0, (1, 2, 3)),
            ),
            k=ArrayInfo(
                (cfg.embed, cfg.kv_heads, cfg.head_dim),
                cfg.dtype,
                ("qkv_embed", "kv_heads", "head_dim"),
                _init(0, (1, 2)),
            ),
            v=ArrayInfo(
                (cfg.embed, cfg.kv_heads, cfg.head_dim),
                cfg.dtype,
                ("qkv_embed", "kv_heads", "head_dim"),
                _init(0, (1, 2)),
            ),
            o=ArrayInfo(
                (cfg.kv_heads, cfg.q_heads // cfg.kv_heads, cfg.head_dim, cfg.embed),
                cfg.dtype,
                ("kv_heads", "o_heads", "head_dim", "o_embed"),
                _init((0, 1), (2, 3)),
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "AttentionLayer", cfg: Config):
        if not cfg.quant_attn:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            q=QuantArray(
                *quantize(layer.q, 0, scale_dtype),
                out_scaling=True,
                scale_expand_dims=-2,
            ),
            k=QuantArray(
                *quantize(layer.k, 0, scale_dtype),
                out_scaling=True,
                scale_expand_dims=-2,
            ),
            v=QuantArray(
                *quantize(layer.v, 0, scale_dtype),
                out_scaling=True,
                scale_expand_dims=-2,
            ),
            o=QuantArray(*quantize(layer.o, (0, 1, 2), scale_dtype), out_scaling=True),
        )


@jax_pytree_struct
class MLPLayer(_Init):
    w_gate: jax.Array | ArrayInfo | QuantArray
    w_up: jax.Array | ArrayInfo | QuantArray
    w_down: jax.Array | ArrayInfo | QuantArray

    ########################################################################################################################
    @classmethod
    def abstract(cls, cfg: Config) -> "MLPLayer":
        _init = _he_normal
        layer = MLPLayer(
            w_gate=ArrayInfo(
                (cfg.embed, cfg.mlp_ffw_size),
                cfg.dtype,
                ("mlp_up_embed", "mlp_up_ffw"),
                _init(0, 1),
            ),
            w_up=ArrayInfo(
                (cfg.embed, cfg.mlp_ffw_size),
                cfg.dtype,
                ("mlp_up_embed", "mlp_up_ffw"),
                _init(0, 1),
            ),
            w_down=ArrayInfo(
                (cfg.mlp_ffw_size, cfg.embed),
                cfg.dtype,
                ("mlp_down_ffw", "mlp_down_embed"),
                _init(0, 1),
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "MLPLayer", cfg: Config):
        if not cfg.quant_mlp:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            w_gate=QuantArray(
                *quantize(layer.w_gate, 0, scale_dtype), out_scaling=True
            ),
            w_up=QuantArray(*quantize(layer.w_up, 0, scale_dtype), out_scaling=True),
            w_down=QuantArray(
                *quantize(layer.w_down, 0, scale_dtype), out_scaling=True
            ),
        )


@jax_pytree_struct
class MambaLayer(_Init):
    wg_in: jax.Array | ArrayInfo | QuantArray
    wx_in: jax.Array | ArrayInfo | QuantArray
    wb_in: jax.Array | ArrayInfo | QuantArray
    wc_in: jax.Array | ArrayInfo | QuantArray
    wdt_in: jax.Array | ArrayInfo | QuantArray
    A_log_D_dt_bias: jax.Array | ArrayInfo | QuantArray
    w_conv: jax.Array | ArrayInfo | QuantArray
    b_conv: jax.Array | ArrayInfo | QuantArray
    w_out: jax.Array | ArrayInfo | QuantArray
    gamma: jax.Array | ArrayInfo | QuantArray

    @classmethod
    def abstract(cls, cfg: Config):
        _init2d = _he_normal(in_axis=0, out_axis=(1,))
        _init3d = _he_normal(in_axis=0, out_axis=(1, 2))
        in_spec_x = ("mamba_in_embed", "mamba_num_heads", "mamba_head_dim")
        in_spec_bc = ("mamba_in_embed", "mamba_n_groups", "mamba_ssm_state_dim")
        out_spec_x = ("mamba_num_heads", "mamba_head_dim", "mamba_out_embed")
        conv_size = (
            cfg.mamba_num_heads * cfg.mamba_head_dim
            + 2 * cfg.mamba_n_groups * cfg.mamba_ssm_state_size
        )
        layer = MambaLayer(
            wg_in=ArrayInfo(
                (cfg.embed, cfg.mamba_num_heads, cfg.mamba_head_dim),
                cfg.dtype,
                in_spec_x,
                _init3d,
            ),
            wx_in=ArrayInfo(
                (cfg.embed, cfg.mamba_num_heads, cfg.mamba_head_dim),
                cfg.dtype,
                in_spec_x,
                _init3d,
            ),
            wb_in=ArrayInfo(
                (cfg.embed, cfg.mamba_n_groups, cfg.mamba_ssm_state_size),
                cfg.dtype,
                in_spec_bc,
                _init3d,
            ),
            wc_in=ArrayInfo(
                (cfg.embed, cfg.mamba_n_groups, cfg.mamba_ssm_state_size),
                cfg.dtype,
                in_spec_bc,
                _init3d,
            ),
            wdt_in=ArrayInfo(
                (cfg.embed, cfg.mamba_num_heads), cfg.dtype, (None, None), _init2d
            ),  # negligable size
            A_log_D_dt_bias=ArrayInfo(
                (3 * cfg.mamba_num_heads,),
                cfg.dtype,
                (None,),
                jax.nn.initializers.zeros,
            ),
            w_conv=ArrayInfo(
                (conv_size, 1, cfg.mamba_conv_kernel_size),
                cfg.dtype,
                (None, None),
                _init2d,
            ),
            b_conv=ArrayInfo(
                (conv_size,), cfg.dtype, (None,), jax.nn.initializers.zeros
            ),
            w_out=ArrayInfo(
                (cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.embed),
                cfg.dtype,
                out_spec_x,
                _init3d,
            ),
            gamma=ArrayInfo(
                (cfg.mamba_intermediate_size,),
                cfg.dtype,
                ("act_embed",),
                _const_init(1.0),
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "MambaLayer", cfg: Config):
        if not cfg.quant_mamba:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            wg_in=QuantArray(*quantize(layer.wg_in, 0, scale_dtype), out_scaling=True),
            wx_in=QuantArray(*quantize(layer.wx_in, 0, scale_dtype), out_scaling=True),
            wb_in=QuantArray(*quantize(layer.wb_in, 0, scale_dtype), out_scaling=True),
            wc_in=QuantArray(*quantize(layer.wc_in, 0, scale_dtype), out_scaling=True),
        )


@jax_pytree_struct
class MoELayer(_Init):
    # router
    w_router: jax.Array | ArrayInfo | QuantArray
    b_router: jax.Array | ArrayInfo | QuantArray
    # experts
    we_up: jax.Array | ArrayInfo | QuantArray
    we_down: jax.Array | ArrayInfo | QuantArray
    # shared experts
    ws_up: jax.Array | ArrayInfo | QuantArray
    ws_down: jax.Array | ArrayInfo | QuantArray

    @classmethod
    def abstract(cls, cfg: Config):
        _einit = _he_normal(in_axis=0, out_axis=(1, 2))
        _sinit = _he_normal(in_axis=0, out_axis=1)
        dtype = cfg.dtype
        layer = MoELayer(
            w_router=ArrayInfo(
                (cfg.embed, cfg.moe_num_experts),
                cfg.moe_gate_dtype,
                ("moe_e_up_embed", None),
                _sinit,
            ),
            b_router=ArrayInfo(
                (cfg.moe_num_experts,),
                cfg.moe_gate_dtype,
                (None,),
                jax.nn.initializers.zeros,
            ),
            we_up=ArrayInfo(
                (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size),
                dtype,
                ("moe_e_experts", "moe_e_up_embed", "moe_e_up_ffw"),
                _einit,
            ),
            we_down=ArrayInfo(
                (cfg.moe_num_experts, cfg.moe_ffw_size, cfg.embed),
                dtype,
                ("moe_e_experts", "moe_e_down_ffw", "moe_e_down_embed"),
                _einit,
            ),
            ws_up=ArrayInfo(
                (cfg.embed, cfg.moe_shared_ffw_size),
                dtype,
                ("moe_s_up_embed", "moe_s_up_ffw"),
                _sinit,
            ),
            ws_down=ArrayInfo(
                (cfg.moe_shared_ffw_size, cfg.embed),
                dtype,
                ("moe_s_down_ffw", "moe_s_down_embed"),
                _sinit,
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "MoELayer", cfg: Config):
        if not cfg.quant_moe:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            we_up=QuantArray(*quantize(layer.we_up, 1, scale_dtype), out_scaling=True),
            we_down=QuantArray(
                *quantize(layer.we_down, 1, scale_dtype), out_scaling=True
            ),
            ws_up=QuantArray(*quantize(layer.ws_up, 0, scale_dtype), out_scaling=True),
            ws_down=QuantArray(
                *quantize(layer.ws_down, 0, scale_dtype), out_scaling=True
            ),
        )


@jax_pytree_struct
class Layer(_Init):
    ffw: MoELayer | MLPLayer | AttentionLayer | MambaLayer
    gamma: jax.Array | ArrayInfo

    ########################################################################################################################
    @classmethod
    def abstract(cls, cfg: Config, layer_idx: int) -> "Layer":
        if cfg.layer_pattern[layer_idx] == "M":
            ffw = MambaLayer.abstract(cfg)
        elif cfg.layer_pattern[layer_idx] == "E":
            ffw = MoELayer.abstract(cfg)
        elif cfg.layer_pattern[layer_idx] == "*":
            ffw = AttentionLayer.abstract(cfg)
        else:
            raise NotImplementedError
        return Layer(
            ffw=ffw,
            gamma=ArrayInfo((cfg.embed,), cfg.dtype, ("act_embed",), _const_init(1.0)),
        )

    @staticmethod
    def quantize(layer: "Layer", cfg: Config):
        return dataclasses.replace(layer, ffw=layer.ffw.quantize(layer.ffw, cfg))


@jax_pytree_struct
class Weights(_Init):
    layers: list[Layer]
    embedding: jax.Array | ArrayInfo
    gamma_final: jax.Array | ArrayInfo
    lm_head: jax.Array | ArrayInfo

    @classmethod
    def abstract(cls, cfg: Config):
        layers = [Layer.abstract(cfg, layer_idx) for layer_idx in range(cfg.num_layers)]
        init = _he_normal
        return Weights(
            layers=layers,
            embedding=ArrayInfo(
                (cfg.vocab_size, cfg.embed),
                cfg.dtype,
                ("vocab_in", "vocab_in"),
                init(0, 1),
            ),
            gamma_final=ArrayInfo(
                (cfg.embed,), cfg.dtype, ("act_embed",), _const_init(1.0)
            ),
            lm_head=ArrayInfo(
                (cfg.embed, cfg.vocab_size),
                cfg.dtype,
                ("vocab_in", "vocab_out"),
                init(1, 0),
            ),
        )


@jax_pytree_struct
class MambaCacheEntry(_Init):
    ssm_states: jax.Array  # (batch_size, key_heads, max_seq_len, head_dim)
    conv_state: tuple[jax.Array, ...]  # (batch_size, key_heads, max_seq_len, head_dim)

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int):
        ssm_states_val = ArrayInfo(
            (
                batch_size,
                cfg.mamba_num_heads,
                cfg.mamba_head_dim,
                cfg.mamba_ssm_state_size,
            ),
            cfg.mamba_dtype,
            ("batch", "mamba_num_heads", "mamba_head_dim", "mamba_ssm_state_dim"),
            jax.nn.initializers.zeros,
        )
        hidden_states_conv_state = ArrayInfo(
            (
                batch_size,
                cfg.mamba_conv_kernel_size,
                cfg.mamba_num_heads,
                cfg.mamba_head_dim,
            ),
            cfg.mamba_dtype,
            ("batch", None, "mamba_num_heads", "mamba_head_dim"),
            jax.nn.initializers.zeros,
        )
        bc_conv_state = ArrayInfo(
            (
                batch_size,
                cfg.mamba_conv_kernel_size,
                cfg.mamba_n_groups,
                cfg.mamba_ssm_state_size,
            ),
            cfg.mamba_dtype,
            ("batch", None, "mamba_n_groups", "mamba_ssm_state_dim"),
            jax.nn.initializers.zeros,
        )
        cache = MambaCacheEntry(
            ssm_states=ssm_states_val,
            conv_state=(hidden_states_conv_state, bc_conv_state, bc_conv_state),
        )
        return cache

    @property
    def buffers(self) -> tuple[jax.Array | QuantArray, ...]:
        raise NotImplementedError


@jax_pytree_struct
class KVCacheEntry(_Init):
    k: jax.Array  # (batch_size, key_heads, max_seq_len, head_dim)
    v: jax.Array  # (batch_size, key_heads, max_seq_len, head_dim)

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int, max_seq_len: int):
        val_info = ArrayInfo(
            (batch_size, cfg.kv_heads, max_seq_len, cfg.head_dim),
            cfg.dtype,
            ("batch", "kv_heads", "sequence", "head_dim"),
            jax.nn.initializers.zeros,
        )
        cache = KVCacheEntry(k=val_info, v=val_info)
        if cfg.quant_cache:
            _quantize = partial(
                quantize, axis=-1, scale_dtype=cfg.quant_scale_dtype, zero_init=True
            )
            cache = dataclasses.replace(
                cache,
                k=QuantArray(
                    *_quantize(cache.k), out_scaling=True, scale_expand_dims=(-2, -3)
                ),
                v=QuantArray(
                    *_quantize(cache.v), out_scaling=False, scale_expand_dims=(-2, -3)
                ),
            )
        return cache

    def fill_len(self) -> jax.Array:
        return jnp.where(self.iter >= 0, (self.iter - self.starts) % self.size, 0)

    @property
    def buffers(self) -> tuple[jax.Array | QuantArray, ...]:
        # return (self.k, self.v)
        raise NotImplementedError


@partial(jax_pytree_struct, meta_fields=["time_axis", "size"])
class KVCache(_Init):
    entries: list[KVCacheEntry | MambaCacheEntry | None]
    iter: jax.Array  # []  # sequences are right-aligned for slice udpate performance
    starts: (
        jax.Array
    )  # [batch_size]  # sequences are right-aligned, we need start indices
    time_axis: int = 2
    size: int = -1

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int, max_seq_len: int):
        entries = []
        for idx, pat in enumerate(cfg.layer_pattern):
            del idx
            if pat == "M":
                entries.append(MambaCacheEntry.abstract(cfg, batch_size))
            elif pat == "*":
                entries.append(
                    KVCacheEntry.abstract(cfg, batch_size, max_seq_len=max_seq_len)
                )
            else:
                entries.append(None)
        return KVCache(
            entries=entries,
            iter=ArrayInfo((), jnp.int32, (), _const_init(-1)),
            starts=ArrayInfo(
                (batch_size,), jnp.int32, ("batch",), jax.nn.initializers.zeros
            ),
            size=max_seq_len,
        )

    def fill_len(self) -> jax.Array:
        return jnp.where(self.iter >= 0, (self.iter - self.starts) % self.size, 0)


def segment_ids_to_positions(segment_ids):
    """Counts positions for segment ids."""

    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

    vals = (jnp.zeros_like(segment_ids), segment_ids)
    return jnp.array(
        jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32"
    )


def make_attention_mask(
    q_len, k_len, q_segment_ids, kv_segment_ids, q_offset, kv_offset, causal: bool
):
    segment_mask = (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :])[
        :, None, :, :
    ]  # [B, 1, t, T]
    if causal:
        qk = (1, 1, q_len, k_len)  # [b, h, t, T]
        q_positions = (
            jax.lax.broadcasted_iota(jnp.int32, qk, 2) + q_offset[:, None, None, None]
        )
        kv_positions = (
            jax.lax.broadcasted_iota(jnp.int32, qk, 3) + kv_offset[:, None, None, None]
        ) % k_len
        causal_mask = q_positions >= kv_positions
        return segment_mask & causal_mask
    return segment_mask


def attention(
    q: jax.Array,
    k: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    k_segment_ids: jax.Array,
    q_offset: jax.Array,
    kv_offset: jax.Array,
    starts: jax.Array,
    lengths: jax.Array,
    cfg: Config,
) -> jax.Array:
    """
    Compute attention.

    Args:
    q: Query tensor of shape (batch_size, num_heads, q_len, head_dim)
    k: Key tensor of shape (batch_size, num_heads, k_len, head_dim)
    v: Value tensor of shape (batch_size, num_heads, k_len, head_dim)
    q_segment_ids: Query segment IDs of shape (batch_size, q_len)
    k_segment_ids: Key segment IDs of shape (batch_size, k_len)
    q_offset: Query offset of shape (batch_size,)
    cfg: Configuration object

    Returns:
    Attention output of shape (batch_size, num_heads, q_len, head_dim)
    """
    del starts, lengths
    # grouped-query attention
    (_, _, _, t, _), (_, _, T, _) = q.shape, k.shape
    scale = cfg.head_dim**-0.5

    qk = einsum("bhgtd,bhTd->bhgtT", q, k) * scale
    mask = make_attention_mask(
        t, T, q_segment_ids, k_segment_ids, q_offset, kv_offset, cfg.causal
    )
    qk = jnp.where(
        mask[:, :, None, ...], qk, -1e30
    )  # Apply the combined mask and add axis for q-groups
    # jax softmax impl includes max subtraction for numerical stability, no need to do it outside.
    attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)
    return einsum("bhgtT,bhTd->bhgtd", attn, v).astype(cfg.dtype)


def attention_kernel(
    q: jax.Array,
    k: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_offset: jax.Array,
    kv_offset: jax.Array,
    starts: jax.Array,
    lengths: jax.Array,
    *,
    cfg: Config,
) -> jax.Array:
    """Flash attention kernel!"""

    # On TPUv3, pallas seems to only work with float32.
    # q, k, v = jnp.float32(q), jnp.float32(k), jnp.float32(v)

    k, k_scale = (k.quant, k.scale) if is_type(k, QuantArray) else (k, None)
    v, v_scale = (v.quant, v.scale) if is_type(v, QuantArray) else (v, None)

    # handle grouped query attention
    assert q.shape[-3] % k.shape[-3] == 0
    scale = q.shape[-1] ** -0.5

    l2p = lambda *logical: logical_to_physical(logical, cfg.rules)
    q_shape, kv_repeats = q.shape, q.shape[-3] // k.shape[-3]
    kv_repeats_spec = tuple(set(*l2p("q_heads")) - set(*l2p("kv_heads")))
    kv_repeats_spec = kv_repeats_spec if len(kv_repeats_spec) > 0 else (None,)
    q_spec = P(
        *(l2p("batch", "kv_heads") + kv_repeats_spec + l2p("sequence", "head_dim"))
    )
    q = jax.lax.reshape(
        q,
        (q.shape[:-3] + (k.shape[-3], kv_repeats, q.shape[-2], q.shape[-1])),
        out_sharding=q_spec,
    )

    # shard_map
    in_specs = (
        q_spec,  # q
        l2p("batch", "kv_heads", "sequence", "head_dim"),  # k
        l2p("batch", "kv_heads", "sequence", "head_dim"),  # v
        l2p("batch", "sequence"),  # q_segment_ids
        l2p("batch", "sequence"),  # kv_segment_ids
        l2p("batch"),  # q_offset
        l2p("batch"),  # kv_offset
    )
    in_specs += (
        None if k_scale is None else l2p("batch", "kv_heads", "sequence"),
    )  # k_scales
    in_specs += (
        None if v_scale is None else l2p("batch", "kv_heads", "sequence"),
    )  # v_scales

    @partial(
        jax.shard_map,
        mesh=cfg.mesh,
        in_specs=in_specs,
        out_specs=q_spec,
        check_vma=False,
    )
    def _f(
        q, k, v, q_segment_ids, kv_segment_ids, q_offset, kv_offset, k_scale, v_scale
    ) -> jax.Array:
        assert which_platform(cfg) == "tpu", (
            "Currently only TPU supports prefill attention, feel free to send a PR."
        )
        q_seq, kv_seq, kv_heads = q.shape[-2], v.shape[-2], v.shape[-3]
        block_q, block_kv = min(q_seq, 512), min(kv_seq, 1024)
        block_sizes = splash.BlockSizes(
            block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv
        )

        # for prefill with an empty cache
        mask = mask_lib.MultiHeadMask(
            [mask_lib.CausalMask((q_seq, kv_seq)) for _ in range(q.shape[-3])]
        )

        def attn_static_fn(q, k, v, segment_ids):
            attn_fn = splash.make_splash_mqa_single_device(
                mask=mask, block_sizes=block_sizes
            )
            attn_fn = jax.vmap(attn_fn, (0, 0, 0, None))  # map over kv heads for mqa
            attn_fn = jax.vmap(attn_fn, (0, 0, 0, 0))  # map over batch
            return attn_fn(q, k, v, segment_ids)

        # when the offsets are different (chunked prefill)
        def attn_dynamic_fn(
            q, k, v, segment_ids
        ):  # when the offsets are different (chunked prefill)
            q_segment_ids, kv_segment_ids = segment_ids.q, segment_ids.kv
            mask = make_attention_mask(
                q_seq,
                kv_seq,
                q_segment_ids,
                kv_segment_ids,
                q_offset,
                kv_offset,
                causal=True,
            )
            attn_fn = lambda q, k, v, segment_ids, mask: (
                splash.make_splash_mqa_single_device(
                    mask=mask, block_sizes=block_sizes
                )(q, k, v, segment_ids)
            )
            attn_fn = jax.vmap(
                attn_fn, (0, 0, 0, None, None)
            )  # map over kv heads for mqa
            attn_fn = jax.vmap(attn_fn, (0, 0, 0, 0, 0))  # map over batch
            return attn_fn(q, k, v, segment_ids, mask)

        segment_ids = splash.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        if k_scale is not None:
            k = (k * k_scale[..., None]).astype(jnp.bfloat16)
        if v_scale is not None:
            v = (v * v_scale[..., None]).astype(jnp.bfloat16)
        return jax.lax.cond(
            jnp.all(q_offset == kv_offset),
            attn_static_fn,
            attn_dynamic_fn,
            q * scale,
            k,
            v,
            segment_ids,
        )

    out = _f(
        q, k, v, q_segment_ids, kv_segment_ids, q_offset, kv_offset, k_scale, v_scale
    ).astype(jnp.bfloat16)
    return jax.lax.reshape(
        out, q_shape, out_sharding=l2p("batch", "q_heads", "sequence", "head_dim")
    )


def rms_norm(
    x: jax.Array, gamma: jax.Array | None, eps: jax.Array | float, axis=-1
) -> jax.Array:
    """Apply RMS normalization."""
    x_dtype = x.dtype
    rms = jnp.sqrt(
        jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=axis, keepdims=True) + eps
    )
    return jnp.astype((gamma if gamma is not None else 1) * x / rms, x_dtype)


def attention_block(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: AttentionLayer,
    *,
    cache: KVCache | None = None,
    idx: int,
    cfg: Config,
):
    l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
    x = x.astype(cfg.dtype)

    # Multi-head attention
    with jax.named_scope("qkv_matmul"):
        q = einsum("btd,dhgq->bhgtq", x, layer.q).astype(cfg.dtype)
        k = einsum("btd,dhq->bhtq", x, layer.k).astype(cfg.dtype)
        v = einsum("btd,dhq->bhtq", x, layer.v).astype(cfg.dtype)

    with jax.named_scope("cache_update"):
        if is_type(cache, KVCache):
            it = jnp.maximum(cache.iter, 0)
            k = update_slice(
                cache.entries[idx].k, k, it, update_axis=cache.time_axis, quant_axis=-1
            )
            v = update_slice(
                cache.entries[idx].v, v, it, update_axis=cache.time_axis, quant_axis=-1
            )

            # create position embeddings
            additional_tokens = jnp.max(_length_minus_right_padding(segment_ids))
            time_indices = (
                jnp.arange(0, v.shape[-2])[None, :] - cache.starts[:, None]
            ) % cache.size
            q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
            kv_segment_ids = (time_indices >= 0) & (
                time_indices < cache.fill_len()[:, None] + additional_tokens
            )
            q_offset = cache.fill_len() - _count_left_padding(
                q_segment_ids, pad_id=0
            )  # pad_id=0 for segment_ids
            kv_offset = -cache.starts
            starts, lengths = cache.starts, cache.fill_len() + additional_tokens
        else:
            q_segment_ids, kv_segment_ids = segment_ids, segment_ids
            starts = _count_left_padding(kv_segment_ids, 0)  # pad_id=0 for segment_ids
            lengths = _length_minus_right_padding(kv_segment_ids)
            q_offset, kv_offset = -starts, -starts
        cache_updates = KVCacheEntry(k, v)

    # Compute attention
    with jax.named_scope("attention"):
        attn_args = (
            q,
            k,
            v,
            q_segment_ids,
            kv_segment_ids,
            q_offset,
            kv_offset,
            starts,
            lengths,
        )
        if cfg.use_prefill_attn_kernel and q.shape[-2] != 1:
            attn_out = attention_kernel(*attn_args, cfg=cfg)
        else:
            attn_out = attention(*attn_args, cfg)

    # Project attention output
    with jax.named_scope("projection"):
        attn_out = einsum(
            "bhgtq,hgqd->btd",
            attn_out,
            layer.o,
            out_sharding=l2p("batch", "sequence", "act_embed"),
        ).astype(cfg.dtype)
    return attn_out, cache_updates


def _reshape_to_chunks(x, chunk_size):
    assert x.shape[1] % chunk_size == 0
    return x.reshape(x.shape[:1] + (x.shape[1] // chunk_size, chunk_size) + x.shape[2:])


def _segment_sum(x: jax.Array):
    c = x.shape[-1]
    iota = jnp.arange(c)
    z = jnp.cumsum(jnp.where(iota[:, None] > iota[None, :], x[..., None], 0), -2)
    return jnp.where(iota[:, None] >= iota[None, :], z, -float("inf"))


def mamba_block(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: MambaLayer,
    cache: KVCache,
    idx: int,
    cfg: Config,
):
    # adapted from https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/modeling_nemotron_h.py
    # with explicit sharding
    l2p = lambda *args: logical_to_physical(args, cfg.rules)

    if cache is not None:
        prev_ssm_state = cache.entries[idx].ssm_states
    else:
        out_sharding = l2p(
            "batch", "mamba_num_heads", "mamba_head_dim", "mamba_ssm_state_dim"
        )
        prev_ssm_state = jnp.zeros(
            (
                x.shape[0],
                cfg.mamba_num_heads,
                cfg.mamba_head_dim,
                cfg.mamba_ssm_state_size,
            ),
            dtype=x.dtype,
            out_sharding=out_sharding,
        )

    right_padding = jnp.max(_count_right_padding(segment_ids))
    with jax.named_scope("roll_act_right"):
        # if we're doing prefill that's not right aligned, we roll the state to the right
        if x.shape[1] > 1:

            def shift_right(x, segment_ids):
                x, segment_ids = (
                    jnp.roll(x, right_padding, axis=1),
                    jnp.roll(segment_ids, right_padding, axis=1),
                )
                segment_ids = jnp.where(
                    jnp.arange(segment_ids.shape[1]) > right_padding, segment_ids, 0
                )
                return x, segment_ids

            x, segment_ids = jax.lax.cond(
                right_padding > 0, shift_right, lambda *args: args, x, segment_ids
            )

    x = x.astype(cfg.mamba_dtype)
    mask = segment_ids != 0
    x = jnp.where(mask[..., None], x, 0.0)

    with jax.named_scope("in_project"):
        gate = einsum("bte,ehd->bthd", x, layer.wg_in).astype(cfg.mamba_dtype)
        hidden_states = einsum("bte,ehd->bthd", x, layer.wx_in).astype(cfg.mamba_dtype)
        dt = einsum("bte,eh->bth", x, layer.wdt_in).astype(cfg.mamba_dtype)
        B = einsum("bte,egs->btgs", x, layer.wb_in).astype(cfg.mamba_dtype)
        C = einsum("bte,egs->btgs", x, layer.wc_in).astype(cfg.mamba_dtype)

    kernel_size = cfg.mamba_conv_kernel_size

    with jax.named_scope("prepare_conv_weights"):
        splt_sz = [
            cfg.mamba_intermediate_size,
            cfg.mamba_intermediate_size + cfg.mamba_n_groups * cfg.mamba_ssm_state_size,
        ]
        (wx_conv, wb_conv, wc_conv), (bx_conv, bb_conv, bc_conv) = (
            jnp.split(w, splt_sz, axis=0) for w in [layer.w_conv, layer.b_conv]
        )
        wx_conv_spec = jax.typeof(hidden_states).sharding.spec[-2:]
        wbc_conv_spec = jax.typeof(B).sharding.spec[-2:]

        wx_conv = wx_conv.reshape(
            (*hidden_states.shape[-2:], 1, kernel_size),
            out_sharding=P(*wx_conv_spec, None, None),
        )
        bx_conv = bx_conv.reshape(
            hidden_states.shape[-2:], out_sharding=P(*wx_conv_spec)
        )
        wb_conv = wb_conv.reshape(
            (*B.shape[-2:], 1, kernel_size), out_sharding=P(*wbc_conv_spec, None, None)
        )
        bb_conv = bb_conv.reshape(B.shape[-2:], out_sharding=P(*wbc_conv_spec))
        wc_conv = wc_conv.reshape(
            (*B.shape[-2:], 1, kernel_size), out_sharding=P(*wbc_conv_spec, None, None)
        )
        bc_conv = bc_conv.reshape(B.shape[-2:], out_sharding=P(*wbc_conv_spec))

        seq_len = hidden_states.shape[1]
        if cache is not None and seq_len < kernel_size:
            prev_hidden_states, prev_B, prev_C = cache.entries[idx].conv_state
            hs_spec, BC_spec = (
                jax.typeof(prev_hidden_states).sharding.spec,
                jax.typeof(prev_B).sharding.spec,
            )
            hidden_states = jnp.concatenate(
                [prev_hidden_states, reshard(hidden_states, hs_spec)], 1
            )
            B, C = (
                jnp.concatenate([prev_B, reshard(B, BC_spec)], 1),
                jnp.concatenate([prev_C, reshard(C, BC_spec)], 1),
            )
            prev_mask = (
                jnp.arange(kernel_size)[::-1][None, :] < cache.fill_len()[:, None]
            )
            mask = jnp.concat(
                [reshard(prev_mask, jax.typeof(mask).sharding.spec), mask], axis=1
            )
            hidden_states, B, C = jax.tree.map(
                lambda z: jnp.where(mask[..., None, None], z, 0), (hidden_states, B, C)
            )
        elif cache is None and seq_len < kernel_size:
            raise NotImplementedError(
                f"Prefill without a cache and for {seq_len=} < {kernel_size=} isn't implemented."
            )
        else:
            hidden_states, B, C = jax.tree.map(
                lambda z: jnp.where(mask[..., None, None], z, 0), (hidden_states, B, C)
            )
        conv_states = (
            hidden_states[:, -kernel_size:, ...],
            B[:, -kernel_size:, ...],
            C[:, -kernel_size:, ...],
        )

    with jax.named_scope("convolution"):
        conv_fn = partial(
            jax.lax.conv, padding=[(kernel_size - 1, 0)], window_strides=(1,)
        )
        conv_fn = jax.vmap(conv_fn, in_axes=(3, 0), out_axes=3)
        conv_fn = jax.vmap(conv_fn, in_axes=(3, 0), out_axes=3)
        act_fn = jax.nn.silu

        hidden_states_spec, BC_spec = (
            jax.typeof(hidden_states).sharding.spec,
            jax.typeof(B).sharding.spec,
        )

        @partial(jax.shard_map, out_specs=(hidden_states_spec, BC_spec, BC_spec))
        def conv_block(
            hidden_states, B, C, wx_conv, wb_conv, wc_conv, bx_conv, bb_conv, bc_conv
        ):
            hidden_states = act_fn(
                jnp.squeeze(
                    conv_fn(
                        hidden_states[:, None, ...],
                        wx_conv[:, :, None, ...].astype(cfg.mamba_dtype),
                    ),
                    1,
                )
                + bx_conv
            )
            B = act_fn(
                jnp.squeeze(
                    conv_fn(
                        B[:, None, ...],
                        wb_conv[:, :, None, ...].astype(cfg.mamba_dtype),
                    ),
                    1,
                )
                + bb_conv
            )
            C = act_fn(
                jnp.squeeze(
                    conv_fn(
                        C[:, None, ...],
                        wc_conv[:, :, None, ...].astype(cfg.mamba_dtype),
                    ),
                    1,
                )
                + bc_conv
            )
            return hidden_states, B, C

        hidden_states, B, C = conv_block(
            hidden_states, B, C, wx_conv, wb_conv, wc_conv, bx_conv, bb_conv, bc_conv
        )

        if cache is not None and seq_len < kernel_size:
            hidden_states, B, C = jax.tree.map(
                lambda x: x[:, -seq_len:, ...], (hidden_states, B, C)
            )

    A_log, D, dt_bias = jnp.split(layer.A_log_D_dt_bias, 3, axis=-1)
    A = -jnp.exp(A_log.astype(jnp.float32))
    dt = jax.nn.softplus(dt + dt_bias)  # [b, 1, h]
    dt = jnp.clip(dt, *cfg.mamba_time_step_limit)
    A, D, dt = (
        A.astype(cfg.mamba_dtype),
        D.astype(cfg.mamba_dtype),
        dt.astype(cfg.mamba_dtype),
    )

    (bs, t), r = x.shape[:2], cfg.mamba_num_heads // cfg.mamba_n_groups
    if t == 1:  # a single token (decode)
        assert x.shape[1] == 1, (
            f"This routine supports a sequence length of 1 only, but {x.shape[1]=}"
        )
        hidden_states, B, C, dt = jax.tree.map(
            partial(jnp.squeeze, axis=1), (hidden_states, B, C, dt)
        )
        dA = jnp.exp(jnp.einsum("bh,h->bh", dt, A))

        hidden_states = hidden_states.reshape(
            (bs, cfg.mamba_num_heads, cfg.mamba_head_dim)
        )
        BC_spec = l2p("batch", "mamba_num_heads", "mamba_ssm_state_dim")
        B = jnp.repeat(
            B.reshape((bs, cfg.mamba_n_groups, cfg.mamba_ssm_state_size)),
            r,
            axis=1,
            out_sharding=BC_spec,
        )
        dBx = jnp.einsum(
            "bhs,bhd->bhds",
            jnp.einsum("bh,bhs->bhs", dt, B, out_sharding=BC_spec),
            hidden_states,
        )
        ssm_state_spec = jax.typeof(prev_ssm_state).sharding.spec
        ssm_state = (
            jnp.einsum("bh,bhds->bhds", dA, prev_ssm_state, out_sharding=ssm_state_spec)
            + dBx
        )
        C = jnp.repeat(
            C.reshape((bs, cfg.mamba_n_groups, cfg.mamba_ssm_state_size)),
            r,
            axis=1,
            out_sharding=BC_spec,
        )
        y = jnp.einsum(
            "bhs,bhds->bhd", C, ssm_state, out_sharding=P(*ssm_state_spec[:-1])
        )

        y = y + (D[..., None] * hidden_states)
        y = y.reshape((bs, 1, cfg.mamba_num_heads, cfg.mamba_head_dim))
    else:
        hidden_states = hidden_states.reshape((bs, t, -1, cfg.mamba_head_dim))
        B, C = (
            B.reshape(bs, t, -1, cfg.mamba_ssm_state_size),
            C.reshape(bs, t, -1, cfg.mamba_ssm_state_size),
        )

        BC_spec = l2p("batch", "sequence", "mamba_num_heads", "mamba_ssm_state_dim")
        B, C = (
            jnp.repeat(B, r, axis=2, out_sharding=BC_spec),
            jnp.repeat(C, r, axis=2, out_sharding=BC_spec),
        )
        # D_residual = jnp.einsum("h,bthd->bthd", layer.D, hidden_states)
        D_residual = jnp.einsum("h,bthd->bthd", D, hidden_states)

        hidden_states = jnp.einsum("bthd,bth->bthd", hidden_states, dt)
        A = A.astype(hidden_states.dtype) * dt

        chunk_size = t if t % cfg.mamba_chunk_size != 0 else cfg.mamba_chunk_size
        hidden_states, A, B, C = (
            _reshape_to_chunks(z, chunk_size) for z in [hidden_states, A, B, C]
        )

        A = A.transpose(0, 3, 1, 2)
        A_cumsum = jnp.cumsum(A, axis=-1)  # cumulative sum across chunks

        L = jnp.exp(_segment_sum(A))
        # Contraction of C and B to get G (attention-weights like)
        G = jnp.einsum("bTchs,bTChs->bTcCh", C, B)
        # Compute M, equivalent to applying attention mask to weights
        M = jnp.einsum("bTcCh,bhTcC->bTcCh", G, L)
        # Compute Y_diag (apply to values)
        Y_diag = jnp.einsum("bTCch,bTchd->bTChd", M, hidden_states)

        # 2. Compute the state for each intra-chunk (right term of low-rank factorization of off-diagonal blocks; B terms)
        states = jnp.einsum(
            "bTchs,bTchd,bhTc->bThds",
            B,
            hidden_states,
            jnp.exp(A_cumsum[:, :, :, -1:] - A_cumsum),
        )

        # previous_states = jnp.zeros_like(states[:, :1])
        # states: [b, T, h, d, s]
        # previous_states: [b, 1, h, d, s]
        states = jnp.concat(
            [
                reshard(prev_ssm_state[:, None, ...], jax.typeof(states).sharding.spec),
                states,
            ],
            axis=1,
        )
        # states: [b, (1 + T), h, d, s]
        decay_chunk = jnp.exp(
            _segment_sum(jnp.pad(A_cumsum[:, :, :, -1], ((0, 0),) * 2 + ((1, 0),)))
        )
        # decay_chunk: [b, h, (1 + T), (1 + T)'] from a segment_sum on [b, h, (1 + T)]
        decay_chunk = decay_chunk.swapaxes(1, 3)
        # decay_chunk: [b, (1 + T)', (1 + T), h]
        new_states = jnp.einsum("bTyh,bThds->byhds", decay_chunk, states)
        states, ssm_state = new_states[:, :-1, ...], new_states[:, -1, ...]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        Y_off = (
            jnp.einsum("bTchs,bThds->bTchd", C, states)
            * jnp.einsum("bhTc->bTch", jnp.exp(A_cumsum))[..., None]
        )

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        y = Y_diag + Y_off
        # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
        y = y.reshape((bs, -1, cfg.mamba_num_heads, cfg.mamba_head_dim))

        y = y + D_residual
        y = y.reshape((bs, t, cfg.mamba_num_heads, cfg.mamba_head_dim))

    with jax.named_scope("out_project"):
        group_size = (cfg.mamba_num_heads * cfg.mamba_head_dim) // cfg.mamba_n_groups
        norm_shape = y.shape[:-2] + (-1, group_size)
        # grouped layer norm
        rms_norm_in = (y * jax.nn.silu(gate)).reshape(
            norm_shape, out_sharding=l2p("batch", "sequence", "mamba_n_groups", None)
        )
        out = rms_norm(rms_norm_in, layer.gamma.reshape(norm_shape[-2:]), cfg.norm_eps)
        out = out.reshape(
            y.shape,
            out_sharding=l2p("batch", "sequence", "mamba_num_heads", "mamba_head_dim"),
        )
        out = jnp.einsum(
            "bthd,hde->bte",
            out,
            layer.w_out,
            out_sharding=l2p("batch", "sequence", "act_embed"),
        ).astype(cfg.dtype)

    if ssm_state.dtype != cfg.mamba_dtype:
        raise ValueError(
            f"ssm_state={jax.typeof(ssm_state)} doesn't have {cfg.mamba_dtype}, something went wrong."
        )
    if not all(z.dtype == cfg.mamba_dtype for z in jax.tree.leaves(conv_states)):
        raise ValueError(
            f"conv_states={jax.tree.map(jax.typeof, conv_states)} doesn't have {cfg.mamba_dtype}, something went wrong."
        )

    with jax.named_scope("roll_act_left"):
        if x.shape[1] > 1:
            out = jax.lax.cond(
                right_padding > 0,
                partial(jnp.roll, shift=-right_padding, axis=1),
                lambda x: x,
                out,
            )

    return out, MambaCacheEntry(ssm_state, conv_states)


def _route_tokens_to_moe_experts(
    x: jax.Array,
    w_router: jax.Array,
    b_router: jax.Array,
    replicated_routing: bool,
    cfg: Config,
):
    lsc = lambda x, spec: reshard(x, logical_to_physical(spec, cfg.rules))
    x_shape = x.shape
    x = x.reshape((-1, x.shape[-1]))
    # not distributing the routing work avoids communication for small batches
    x = (
        lsc(x, (None, None))
        if replicated_routing
        else reshard(x, P(TENSOR_AXIS_NAME, None))
    )
    w_router = lsc(w_router, (None, None))

    scores = jax.nn.sigmoid(
        jnp.einsum("Sk,kj->Sj", x, w_router).astype(cfg.moe_gate_dtype)
    )
    scores_for_choice = scores + b_router
    scores_for_choice = scores_for_choice.reshape(
        scores_for_choice.shape[:-1] + (cfg.moe_router_n_groups, -1)
    )
    group_scores = jnp.sum(jax.lax.top_k(scores_for_choice, k=2, axis=-1)[0], -1)
    group_choices = jax.lax.top_k(group_scores, k=cfg.moe_router_topk_groups)[1]
    group_choice_mask = jnp.any(
        jnp.arange(cfg.moe_router_n_groups)[:, None] == group_choices[..., None, :],
        axis=-1,
    )
    scores_for_choice = jnp.where(
        group_choice_mask[:, None], scores_for_choice, -jnp.inf
    )
    scores_for_choice = scores_for_choice.reshape(scores_for_choice.shape[:-2] + (-1,))
    topk_idx = jax.lax.top_k(scores_for_choice, k=cfg.moe_experts_per_tok)[1]
    topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)

    topk_weights = topk_weights / (
        jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
    )
    topk_weights *= cfg.moe_routed_scaling_factor

    topk_weights = lsc(topk_weights, (None, None)).reshape(
        x_shape[:-1] + (cfg.moe_experts_per_tok,)
    )
    topk_idx = lsc(topk_idx, (None, None)).reshape(
        x_shape[:-1] + (cfg.moe_experts_per_tok,)
    )

    return topk_weights, topk_idx


def _moe_gmm(lhs, rhs, group_sizes, topk_idx, cfg: Config):
    assert lhs.ndim == 2 and rhs.ndim == 3, f"{lhs.ndim=} != 2 and {rhs.ndim=} != 3"
    group_sizes = group_sizes.astype(jnp.int32)
    with jax.named_scope("jax.lax.ragged_dot"):
        if is_type(rhs, QuantArray):
            assert rhs.scale.ndim == 2 and rhs.scale.shape == (
                rhs.quant.shape[0],
                rhs.quant.shape[2],
            )
            scale = jnp.take_along_axis(rhs.scale, topk_idx[:, None], axis=-2)
            ret = jax.lax.ragged_dot(lhs, rhs.quant, group_sizes) * scale
        else:
            ret = jax.lax.ragged_dot(lhs, rhs, group_sizes)
    return ret.astype(cfg.dtype)


def moe_block(x: jax.Array, layer: MoELayer, cfg: Config):
    assert x.ndim == 3
    l2p = lambda *axes: logical_to_physical(axes, cfg.rules)
    _psc = lambda z, spec: reshard(z, P(*spec))
    _qpsc = lambda z, spec: dataclasses.replace(
        z, quant=_psc(z.quant, spec.quant), scale=_psc(z.scale, spec.scale)
    )
    psc = lambda z, spec: _qpsc(z, spec) if is_type(z, QuantArray) else _psc(z, spec)

    # we're decoding or device count does not divide total token count
    replicated_routing = (
        x.shape[-2] == 1 or (x.shape[-2] * x.shape[-3]) % jax.device_count() != 0
    )
    topk_weights, topk_idx = _route_tokens_to_moe_experts(
        x, layer.w_router, layer.b_router, replicated_routing, cfg
    )
    tensor_axname, expert_axname = l2p("moe_e_tp")[0], l2p("moe_e_ep")[0]

    x_spec = l2p("batch", "sequence", None)
    topk_weights_spec, topk_idx_spec = (
        l2p("batch", "sequence", None),
        l2p("batch", "sequence", None),
    )
    out_spec = l2p("batch", "sequence", None)

    we_up_spec = l2p("moe_e_ep", None, "moe_e_tp")
    we_down_spec = l2p("moe_e_ep", "moe_e_tp", None)
    if all(is_type(z, QuantArray) for z in [layer.we_up, layer.we_down]):
        we_up_spec = dataclasses.replace(
            layer.we_up, quant=we_up_spec, scale=P(we_up_spec[0], we_up_spec[2])
        )
        we_down_spec = dataclasses.replace(
            layer.we_down, quant=we_down_spec, scale=P(we_down_spec[0], we_down_spec[2])
        )
    we_up = psc(layer.we_up, we_up_spec)
    we_down = psc(layer.we_down, we_down_spec)

    in_specs = (x_spec, we_up_spec, we_down_spec, topk_weights_spec, topk_idx_spec)

    is_embedding_sharded = l2p("act_embed")[0] is not None
    if is_embedding_sharded:  # activations are sharded
        out_spec = P(*(out_spec[:-1] + (tensor_axname,)))  # override last axis name
    if cfg.ep_strategy == "prefill":
        out_spec = P(*(out_spec[:-1] + (tensor_axname,)))  # override last axis name

    expert_count = (
        cfg.mesh.axis_sizes[cfg.mesh.axis_names.index(expert_axname)]
        if expert_axname is not None
        else 1
    )
    tensor_count = (
        cfg.mesh.axis_sizes[cfg.mesh.axis_names.index(tensor_axname)]
        if tensor_axname is not None
        else 1
    )
    assert cfg.moe_num_experts % expert_count == 0
    expert_size = cfg.moe_num_experts // expert_count

    @partial(
        jax.shard_map,
        mesh=cfg.mesh,
        in_specs=in_specs,
        out_specs=out_spec,
        check_vma=False,
    )
    def _expert_fn(x, we_up, we_down, topk_weights, topk_idx):
        (b, s, d), e = x.shape, cfg.moe_experts_per_tok
        expert_idx = (
            jax.lax.axis_index(expert_axname) if expert_axname is not None else 0
        )
        tensor_idx = (
            jax.lax.axis_index(tensor_axname) if tensor_axname is not None else 0
        )
        del tensor_idx
        topk_idx_ = topk_idx.reshape(-1)
        valid_group_mask_ = (topk_idx_ >= expert_size * expert_idx) & (
            topk_idx_ < expert_size * (expert_idx + 1)
        )
        expert_mapped_topk_idx_ = jnp.where(
            valid_group_mask_, topk_idx_ - expert_idx * expert_size, 2**30
        )

        sort_idx_ = jnp.argsort(expert_mapped_topk_idx_, axis=-1)  # [b * s * e]
        isort_idx_ = jnp.argsort(sort_idx_)

        if cfg.ep_strategy == "prefill":
            truncate_size = round(2 * sort_idx_.size / expert_count)
            sort_idx_, isort_idx_ = (
                sort_idx_[:truncate_size],
                isort_idx_[:truncate_size],
            )

        topk_idx_sort_ = topk_idx_[sort_idx_]  # [b * s * e]
        expert_mapped_topk_idx_sort_ = expert_mapped_topk_idx_[sort_idx_]
        valid_group_mask_sort_ = expert_mapped_topk_idx_sort_ < 2**30
        expert_mapped_topk_idx_sort_ = jnp.where(
            expert_mapped_topk_idx_sort_ < 2**30, expert_mapped_topk_idx_sort_, 0
        )

        # equivalent to:
        # ```
        # x_repeat_ = jnp.repeat(x.reshape((-1, x.shape[-1])), e, axis=0)
        # x_repeat_sort_ = jnp.take_along_axis(x_repeat_, sort_idx_[:, None], axis=-2)  # [b * s, d]
        # ```
        x_repeat_sort_ = jnp.take_along_axis(
            x.reshape((-1, x.shape[-1])),
            sort_idx_[:, None] // e,
            axis=-2,  # index trick to avoid jnp.repeat
        )  # [b * s * e, d]

        group_sizes = jnp.bincount(topk_idx_sort_, length=cfg.moe_num_experts)
        group_sizes_shard = jax.lax.dynamic_slice_in_dim(
            group_sizes, expert_idx * expert_size, expert_size, 0
        )

        with jax.named_scope("we_up"):
            ff_up = _moe_gmm(
                x_repeat_sort_,
                we_up,
                group_sizes_shard,
                expert_mapped_topk_idx_sort_,
                cfg,
            )
            ff_up = jnp.where(valid_group_mask_sort_[..., None], ff_up, 0)
            ff_up = jax.nn.relu(ff_up) ** 2
        with jax.named_scope("we_down"):
            ff_out = _moe_gmm(
                ff_up, we_down, group_sizes_shard, expert_mapped_topk_idx_sort_, cfg
            )
            ff_out = jnp.where(
                valid_group_mask_sort_[..., None], ff_out, 0
            )  # expensive

        if cfg.ep_strategy == "prefill":
            rs_shape = (
                math.ceil((ff_out.shape[-1] // tensor_count) / 256) * 256 * tensor_count
            )
            pad_size = rs_shape - ff_out.shape[-1]
            ff_out = jnp.pad(ff_out, ((0, 0), (0, pad_size)))
            ff_out = jax.lax.psum_scatter(
                ff_out, axis_name=tensor_axname, scatter_dimension=1, tiled=True
            )
        ff_out = ff_out * topk_weights.reshape(-1)[sort_idx_][:, None]

        if cfg.ep_strategy == "prefill":
            with jax.named_scope("unpermute"):
                # unpermute tokens
                dtype = jnp.bfloat16
                dim_nums = jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1,),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                )
                ff_out_expert = jax.lax.scatter_add(
                    jnp.zeros((b * s, ff_out.shape[-1]), dtype=dtype),
                    sort_idx_[..., None] // e,
                    ff_out.astype(dtype),
                    dim_nums,
                ).astype(dtype)
                ff_out_expert = ff_out_expert.astype(cfg.dtype)
        else:
            with jax.named_scope("unpermute"):
                ff_out = jnp.take_along_axis(ff_out, isort_idx_[..., None], axis=-2)
            with jax.named_scope("expert_summing"):
                ff_out_expert = jnp.sum(ff_out.reshape((b * s, e, d)), -2)
                ff_out_expert = ff_out_expert.astype(cfg.dtype)

        with jax.named_scope("experts_collective"):
            if cfg.ep_strategy == "prefill":
                if expert_axname is not None:
                    ff_out_expert = jax.lax.psum(ff_out_expert, expert_axname)
            else:
                # collectives
                if (
                    is_embedding_sharded
                ):  # activations are supposed to be sharded on out
                    with jax.named_scope("tp_e_psum_scatter"):
                        ff_out_expert = jax.lax.psum_scatter(
                            ff_out_expert,
                            tensor_axname,
                            scatter_dimension=1,
                            tiled=True,
                        )
                    with jax.named_scope("ep_e_psum"):
                        if expert_axname is not None:
                            ff_out_expert = jax.lax.psum(ff_out_expert, expert_axname)
                else:
                    psum_axes = (
                        tensor_axname
                        if expert_axname is None
                        else (expert_axname, tensor_axname)
                    )
                    ff_out_expert = jax.lax.psum(ff_out_expert, psum_axes)
            ff_out_expert = ff_out_expert.reshape((b, s, ff_out_expert.shape[-1]))
            return ff_out_expert

    with jax.named_scope("moe_routed_expert"):
        x_ = psc(x, x_spec)
        ff_out_expert = _expert_fn(x_, we_up, we_down, topk_weights, topk_idx)[
            ..., : x.shape[-1]
        ]
        ff_out_expert = psc(ff_out_expert, l2p("batch", "sequence", "act_embed"))
    with jax.named_scope("moe_shared_expert"):
        x_ = psc(x, x_spec)
        ff_out_shared = mlp_block(x_, layer.ws_up, layer.ws_down, cfg=cfg)[
            ..., : x.shape[-1]
        ]
        ff_out_shared = psc(ff_out_shared, l2p("batch", "sequence", "act_embed"))
    return ff_out_expert + ff_out_shared


def mlp_block(x: jax.Array, w_up: jax.Array, w_down: jax.Array, *, cfg: Config):
    l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
    dtype = cfg.dtype
    with jax.named_scope("up_proj"):
        ff_up = einsum("btd,df->btf", x, w_up).astype(dtype)
        ff_up = jax.nn.relu(ff_up) ** 2
    with jax.named_scope("down_proj"):
        ff_out = einsum(
            "btf,fd->btd",
            ff_up,
            w_down,
            out_sharding=l2p("batch", "sequence", "act_embed"),
        )
        ff_out = ff_out.astype(dtype)
    return ff_out


def forward_layer(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    idx: int,
    cfg: Config,
    cache: KVCache | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = x.astype(cfg.dtype)

    # Attention block
    with jax.named_scope("attn_pre_norm"):
        layer_in = rms_norm(x, layer.gamma, cfg.norm_eps)
    with jax.named_scope(f"ffn-{cfg.layer_pattern[idx]}"):
        if cfg.layer_pattern[idx] == "M":
            out, cache_updates = mamba_block(
                layer_in, segment_ids, layer.ffw, cache=cache, idx=idx, cfg=cfg
            )
        elif cfg.layer_pattern[idx] == "E":
            out, cache_updates = moe_block(layer_in, layer.ffw, cfg=cfg), None
        elif cfg.layer_pattern[idx] == "*":
            out, cache_updates = attention_block(
                layer_in, segment_ids, layer.ffw, cache=cache, idx=idx, cfg=cfg
            )
        else:
            raise NotImplementedError
    with jax.named_scope("residual"):
        x = x + out.astype(cfg.dtype)
    return x, cache_updates


def forward(
    x: jax.Array,
    segment_ids: jax.Array,
    weights: Weights,
    cfg: Config,
    cache: KVCache | None = None,
):
    l2p = lambda *args: logical_to_physical(args, cfg.rules)
    x = weights.embedding.at[x, :].get(
        out_sharding=l2p("batch", "sequence", "act_embed")
    )  # Embed input tokens [B, T] -> [B, T D]

    positions = segment_ids_to_positions(segment_ids)
    if is_type(cache, KVCache):
        positions = positions + cache.fill_len()[:, None]

    all_cache_updates = []
    for idx, layer in enumerate(weights.layers):
        x, cache_updates = forward_layer(x, segment_ids, layer, idx, cfg, cache)
        all_cache_updates.append(cache_updates)

    x = rms_norm(x, weights.gamma_final, cfg.norm_eps)  # Final layer norm.
    logits = einsum("btd,dv->btv", x, weights.lm_head)  # Project to vocabulary size

    if is_type(cache, KVCache):
        additional_tokens = jnp.max(_length_minus_right_padding(segment_ids))
        iter = (jnp.maximum(0, cache.iter) + additional_tokens) % cache.size
        new_cache = dataclasses.replace(cache, iter=iter, entries=all_cache_updates)
        return logits, new_cache
    else:
        return logits, all_cache_updates


def optimal_formats(cfg: Config):
    SDS, tree_map, bs = (
        jax.ShapeDtypeStruct,
        partial(jax.tree.map, is_leaf=is_param),
        16,
    )
    weights_abstract, cache_abstract = (
        Weights.abstract(cfg),
        KVCache.abstract(cfg, bs, cfg.max_seq_len),
    )
    weights_shardings, cache_shardings = (
        Weights.shardings(cfg),
        KVCache.shardings(cfg, bs, cfg.max_seq_len),
    )
    weights_shapes = tree_map(
        lambda x, s: SDS(x.shape, x.dtype, sharding=s),
        weights_abstract,
        weights_shardings,
    )
    cache_shapes = tree_map(
        lambda x, s: SDS(x.shape, x.dtype, sharding=s), cache_abstract, cache_shardings
    )
    _forward = lambda weights, cache: forward(
        *([jnp.ones((bs, 1), jnp.int32)] * 2), weights, cfg, cache=cache
    )
    with jax.sharding.set_mesh(cfg.mesh):
        fn = jax.jit(
            _forward,
            in_shardings=Format(Layout.AUTO),
            out_shardings=Format(Layout.AUTO),
            donate_argnames=("cache",),
        )
        weights_formats, cache_formats = (
            fn.trace(weights_shapes, cache_shapes).lower().compile().input_formats[0]
        )
    weights = tree_map(
        lambda x, f: SDS(x.shape, x.dtype, sharding=f),
        weights_abstract,
        weights_formats,
    )
    cache = tree_map(
        lambda x, f: SDS(x.shape, x.dtype, sharding=f), cache_abstract, cache_formats
    )
    return weights, cache


# serialization
def save_pytree(weights, path):
    flat_data = odict(
        ("weights" + "".join(map(str, k)), v)
        for k, v in jax.tree.flatten_with_path(weights)[0]
    )
    ser.save(flat_data, path)  # save a flatten-with-path to avoid custom nodes


def load_pytree(path, sharding=None):
    flat_sharding = odict(
        ("weights" + "".join(map(str, k)), v)
        for k, v in jax.tree.flatten_with_path(sharding)[0]
    )
    data = jax.tree.unflatten(
        jax.tree.structure(sharding), jax.tree.leaves(ser.load(path, flat_sharding))
    )
    return data


# Inference.
@partial(jax.jit, static_argnums=(1, 2))
def prepare_chunk(chunk, pad_to: int, pad_id: int):
    # [bs, length] -> [bs, padded]
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    chunk = jnp.pad(
        chunk,
        [(0, 0), (0, pad_to - chunk.shape[-1])],
        mode="constant",
        constant_values=pad_id,
    )
    # chunk = jnp.pad(chunk, [(0, 0), (pad_to - chunk.shape[-1], 0)], mode="constant", constant_values=pad_id)
    segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)
    return chunk, segment_ids


def prefill(
    tokens: jax.Array,
    weights: Weights,
    cache: KVCache,
    cfg: Config,
    pad_id: int = PAD_ID,
) -> tuple[jax.Array, jax.Array, KVCache]:
    """Samples from a prompt."""
    # Calculate the next power of 2 for padding, up to cfg.max_seq.
    assert tokens.shape[-1] <= cfg.max_seq_len
    pad_to = 2 ** math.ceil(math.log2(tokens.shape[-1]))
    prompt, prompt_segment_ids = prepare_chunk(tokens, pad_to=pad_to, pad_id=pad_id)
    assert prompt.ndim == 2

    # cache_shardings = KVCache.shardings(cfg, prompt.shape[0], cfg.max_seq_len)
    if is_type(cache, KVCache):
        uninitialized_iter = -jnp.ones_like(cache.iter)
        cache = dataclasses.replace(
            cache,
            starts=_count_left_padding(prompt, pad_id=pad_id),
            iter=uninitialized_iter,
        )
        cache_shardings = KVCache.shardings(cfg, prompt.shape[0], cache.size)
    else:
        cache_shardings = tuple(
            [z[idx] for idx in range(cfg.num_layers)] for z in cache_shardings
        )
    logits_shardings = jax.sharding.NamedSharding(
        cfg.mesh, P(BATCH_AXIS_NAME, None, TENSOR_AXIS_NAME)
    )
    logits, cache = jax.jit(
        forward, donate_argnums=(4,), out_shardings=(logits_shardings, cache_shardings)
    )(prompt, prompt_segment_ids, weights, cfg, cache)
    next_tokens = jax.jit(partial(jnp.argmax, axis=-1))(logits)
    return next_tokens, logits, cache


@partial(jax.jit, donate_argnames=("cache",))
def decode_step(last_tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config):
    assert last_tokens.ndim == 2
    segment_ids = jnp.ones(last_tokens.shape, dtype=jnp.int32)
    next_logits, cache = forward(last_tokens, segment_ids, weights, cfg, cache)
    next_tokens = jnp.argmax(next_logits, -1)
    return next_tokens, cache
