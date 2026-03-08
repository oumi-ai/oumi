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
import json
import math
import os
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from etils import epath
from jax import random, tree_util
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
from jax.sharding import PartitionSpec as P
from jax.sharding import auto_axes, reshard

from . import ragged_attention
from .decode_ragged_dot import decode_ragged_dot

PAD_ID = 201134

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
    q_heads: AxisName = ATTN_HEADS_AXIS_NAME
    kv_heads: AxisName = ATTN_HEADS_AXIS_NAME
    o_heads: AxisName = ATTN_HEADS_AXIS_NAME
    o_embed: AxisName = None
    # MLP
    embed_up: AxisName = None
    ffw_up: AxisName = TENSOR_AXIS_NAME
    ffw_down: AxisName = TENSOR_AXIS_NAME
    embed_down: AxisName = None
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
    moe_s_up_embed: AxisName = None
    moe_s_up_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_s_down_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_s_down_embed: AxisName = None
    moe_e_tp: AxisName = (
        TENSOR_ONLY_AXIS_NAME  # moe forward function tensor parallelism
    )
    moe_e_ep: AxisName = EXPERT_AXIS_NAME  # moe forward function expert parallelism
    # vocab
    vocab_in: AxisName = None
    vocab_out: AxisName = TENSOR_AXIS_NAME


def logical_to_physical(
    logical: Axes, rules: ShardingRules
) -> jax.sharding.PartitionSpec:
    """Returns how to physically shard a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    spec = [getattr(rules, axis) if axis is not None else None for axis in logical]
    # `spec` may contain tuples, flatten to check that `spec` maps each physical mesh axis to at most one logical array
    # axis.
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
    # Attention
    causal: bool
    nope_layer_interval: int
    use_qk_norm: bool
    attn_chunk_size: int
    # MLP
    mlp_ffw_size: int
    # MoE
    moe_ffw_size: int
    moe_layer_interval: int
    moe_experts_per_tok: int
    moe_num_experts: int
    moe_num_shared_experts: int = 1
    moe_gate_dtype: "jnp.dtype" = jnp.float32
    ep_strategy: str = "decode"
    # kernel config
    use_prefill_attn_kernel: bool = False
    use_decode_attn_kernel: bool = False
    use_ragged_dot_kernel: bool = True
    dtype: "jnp.dtype" = jnp.bfloat16
    norm_eps: float = 1e-5
    # sharding
    rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
    mesh: jax.sharding.Mesh | None = None
    # Llama 3 specific frequency computation
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 8.0
    rope_scaling_low_freq_factor: float = 1.0
    rope_scaling_high_freq_factor: float = 4.0
    rope_scaling_original_max_position_embeddings: int = 8192
    quant_mlp: bool = False
    quant_moe: bool = False
    quant_attn: bool = False
    quant_cache: bool = True
    quant_scale_dtype: "jnp.dtype" = jnp.bfloat16


def llama_to_jax_config(llama_config: Any | dict[str, Any]) -> "Config":
    _get = lambda x, k, default=None: (
        getattr(x, k, default) if hasattr(x, k) else dict(x).get(k, default)
    )
    nope_layer_interval = 4
    return Config(
        embed=_get(llama_config, "dim"),
        mlp_ffw_size={1.2: 16384}[
            _get(llama_config, "ffn_dim_multiplier")
        ],  # odd Llama calculation of hidden dim
        moe_ffw_size={4: 8192}[
            _get(llama_config, "ffn_exp")
        ],  # odd Llama calculation of hidden dim
        q_heads=_get(llama_config, "n_heads"),
        kv_heads=_get(llama_config, "n_kv_heads"),
        num_layers=_get(llama_config, "n_layers"),
        head_dim=_get(llama_config, "head_dim", None)
        if _get(llama_config, "head_dim", None) is not None
        else 128,
        vocab_size=_get(llama_config, "vocab_size"),
        norm_eps=_get(llama_config, "norm_eps"),
        use_qk_norm=_get(llama_config, "use_qk_norm"),
        attn_chunk_size=_get(llama_config, "attention_chunk_size"),
        nope_layer_interval=nope_layer_interval,
        moe_layer_interval=_get(llama_config, "moe_args")["interleave_moe_layer_step"],
        moe_experts_per_tok=_get(llama_config, "moe_args")["top_k"],
        moe_num_experts=_get(llama_config, "moe_args")["num_experts"],
        max_seq_len=128,
        dtype=jnp.bfloat16,
        causal=True,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
        rope_theta=_get(llama_config, "rope_theta"),
    )


def hf_to_jax_config(llama_config: Any | dict[str, Any]) -> "Config":
    _get = lambda x, k, default=None: (
        getattr(x, k, default) if hasattr(x, k) else dict(x).get(k, default)
    )
    # num_layers = _get(llama_config, "num_hidden_layers")
    # nope_layer_interval = round(num_layers / (num_layers - sum(_get(llama_config, "no_rope_layers"))))
    nope_layer_interval = 4
    return Config(
        embed=_get(llama_config, "hidden_size"),
        mlp_ffw_size=_get(llama_config, "intermediate_size_mlp"),
        moe_ffw_size=_get(llama_config, "intermediate_size"),
        q_heads=_get(llama_config, "num_attention_heads"),
        kv_heads=_get(llama_config, "num_key_value_heads"),
        num_layers=_get(llama_config, "num_hidden_layers"),
        head_dim=_get(llama_config, "head_dim"),
        vocab_size=_get(llama_config, "vocab_size"),
        norm_eps=_get(llama_config, "rms_norm_eps"),
        use_qk_norm=_get(llama_config, "use_qk_norm"),
        attn_chunk_size=_get(llama_config, "attention_chunk_size"),
        nope_layer_interval=nope_layer_interval,
        moe_layer_interval=_get(llama_config, "interleave_moe_layer_step"),
        moe_experts_per_tok=_get(llama_config, "num_experts_per_tok"),
        moe_num_experts=_get(llama_config, "num_local_experts"),
        max_seq_len=128,
        dtype=jnp.bfloat16,
        causal=True,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
        rope_theta=_get(llama_config, "rope_theta"),
    )


def load_config(config_path: str | os.PathLike[str] | Path) -> "Config":
    # return llama_to_jax_config(json.loads(Path(config_path).read_text()))
    return hf_to_jax_config(json.loads(Path(config_path).read_text())["text_config"])


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path,
    tokenizer_config_path: str | os.PathLike[str] | Path,
) -> "PreTrainedTokenizerFast":  # noqa: F821
    from transformers import AddedToken, PreTrainedTokenizerFast

    config = json.loads(Path(tokenizer_config_path).read_text())
    config = {
        k: AddedToken(**v) if isinstance(v, dict) and str(k).endswith("token") else v
        for (k, v) in config.items()
    }
    config["added_tokens_decoder"] = {
        int(k): AddedToken(**v)
        for (k, v) in config.get("added_tokens_decoder", dict()).items()
    }
    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)


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
_count_left_padding = lambda ids, pad_id=0: auto_axes(
    lambda ids: jnp.sum(jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1),
    out_sharding=P(None),
)(ids)
_length_minus_right_padding = lambda segment_ids: auto_axes(
    lambda segment_ids: jnp.sum(
        jnp.cumsum(jnp.flip(segment_ids != 0, -1), axis=-1) > 0, -1
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
        return jax.tree.unflatten(
            abstract_struct,
            _init_leaves(key, tuple(abstract_leaves), tuple(shardings_leaves)),
        )


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
                (cfg.embed, cfg.q_heads, cfg.head_dim),
                cfg.dtype,
                ("qkv_embed", "q_heads", "head_dim"),
                _init(0, (1, 2)),
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
                (cfg.q_heads, cfg.head_dim, cfg.embed),
                cfg.dtype,
                ("o_heads", "head_dim", "o_embed"),
                _init(0, (1, 2)),
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
            q=QuantArray(*quantize(layer.q, (1, 2), scale_dtype)),
            k=QuantArray(*quantize(layer.k, (1, 2), scale_dtype)),
            v=QuantArray(*quantize(layer.v, (1, 2), scale_dtype)),
            o=QuantArray(*quantize(layer.o, (0, 1), scale_dtype), out_scaling=True),
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
class MoELayer(_Init):
    # router
    w_router: jax.Array | ArrayInfo | QuantArray
    # experts
    we_gate: jax.Array | ArrayInfo | QuantArray
    we_up: jax.Array | ArrayInfo | QuantArray
    we_down: jax.Array | ArrayInfo | QuantArray
    # shared experts
    ws_gate: jax.Array | ArrayInfo | QuantArray
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
            we_gate=ArrayInfo(
                (cfg.moe_num_experts, cfg.embed, cfg.moe_ffw_size),
                dtype,
                ("moe_e_experts", "moe_e_up_embed", "moe_e_up_ffw"),
                _einit,
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
            ws_gate=ArrayInfo(
                (cfg.embed, cfg.moe_num_shared_experts * cfg.moe_ffw_size),
                dtype,
                ("moe_s_up_embed", "moe_s_up_ffw"),
                _sinit,
            ),
            ws_up=ArrayInfo(
                (cfg.embed, cfg.moe_num_shared_experts * cfg.moe_ffw_size),
                dtype,
                ("moe_s_up_embed", "moe_s_up_ffw"),
                _sinit,
            ),
            ws_down=ArrayInfo(
                (cfg.moe_ffw_size, cfg.moe_num_shared_experts * cfg.embed),
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
            we_gate=QuantArray(
                *quantize(layer.we_gate, 1, scale_dtype), out_scaling=True
            ),
            we_up=QuantArray(*quantize(layer.we_up, 1, scale_dtype), out_scaling=True),
            we_down=QuantArray(
                *quantize(layer.we_down, 1, scale_dtype), out_scaling=True
            ),
            ws_gate=QuantArray(
                *quantize(layer.ws_gate, 0, scale_dtype), out_scaling=True
            ),
            ws_up=QuantArray(*quantize(layer.ws_up, 0, scale_dtype), out_scaling=True),
            ws_down=QuantArray(
                *quantize(layer.ws_down, 0, scale_dtype), out_scaling=True
            ),
        )


@jax_pytree_struct
class Layer(_Init):
    mlp: MLPLayer | MoELayer
    attn: AttentionLayer
    attn_pre_gamma: jax.Array | ArrayInfo
    attn_post_gamma: jax.Array | ArrayInfo

    ########################################################################################################################
    @classmethod
    def abstract(cls, cfg: Config, layer_idx: int) -> "Layer":
        use_moe = (layer_idx + 1) % cfg.moe_layer_interval == 0
        layer = Layer(
            mlp=MoELayer.abstract(cfg) if use_moe else MLPLayer.abstract(cfg),
            attn=AttentionLayer.abstract(cfg),
            attn_pre_gamma=ArrayInfo(
                (cfg.embed,), cfg.dtype, ("act_embed",), _const_init(1.0)
            ),
            attn_post_gamma=ArrayInfo(
                (cfg.embed,), cfg.dtype, ("act_embed",), _const_init(1.0)
            ),
        )
        # layer = cls.quantize(layer, cfg)  # abstract already quantized
        return layer

    @staticmethod
    def quantize(layer: "Layer", cfg: Config):
        return dataclasses.replace(
            layer,
            mlp=layer.mlp.quantize(layer.mlp, cfg),
            attn=layer.attn.quantize(layer.attn, cfg),
        )


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
                (cfg.vocab_size, cfg.embed), cfg.dtype, (None, "vocab_in"), init(0, 1)
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


@partial(jax_pytree_struct, meta_fields=["time_axis", "size"])
class KVCache(_Init):
    k: list[jax.Array]  # (batch_size, key_heads, max_seq_len, head_dim)
    v: list[jax.Array]  # (batch_size, key_heads, max_seq_len, head_dim)
    iter: jax.Array  # []  # sequences are right-aligned for slice udpate performance
    starts: (
        jax.Array
    )  # [batch_size]  # sequences are right-aligned, we need start indices
    time_axis: int = 2
    size: int = -1

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int, max_seq_len: int):
        val_info = ArrayInfo(
            (batch_size, cfg.kv_heads, max_seq_len, cfg.head_dim),
            cfg.dtype,
            ("batch", "kv_heads", "sequence", "head_dim"),
            jax.nn.initializers.zeros,
        )
        cache = KVCache(
            k=[val_info for _ in range(cfg.num_layers)],
            v=[val_info for _ in range(cfg.num_layers)],
            iter=ArrayInfo((), jnp.int32, (), _const_init(-1)),
            starts=ArrayInfo(
                (batch_size,), jnp.int32, ("batch",), jax.nn.initializers.zeros
            ),
            size=max_seq_len,
        )
        if cfg.quant_cache:
            _quantize = partial(
                quantize, axis=-1, scale_dtype=cfg.quant_scale_dtype, zero_init=True
            )
            cache = dataclasses.replace(
                cache,
                k=[
                    QuantArray(
                        *_quantize(cache.k[idx]),
                        out_scaling=True,
                        scale_expand_dims=(-2, -3),
                    )
                    for idx in range(len(cache.k))
                ],
                v=[
                    QuantArray(
                        *_quantize(cache.v[idx]),
                        out_scaling=False,
                        scale_expand_dims=(-2, -3),
                    )
                    for idx in range(len(cache.v))
                ],
            )
        return cache

    def fill_len(self) -> jax.Array:
        return jnp.where(self.iter >= 0, (self.iter - self.starts) % self.size, 0)

    @property
    def buffers(self) -> tuple[jax.Array | QuantArray, ...]:
        return (self.k, self.v)


def segment_ids_to_positions(segment_ids):
    """Counts positions for segment ids."""

    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

    vals = (jnp.zeros_like(segment_ids), segment_ids)
    return jnp.array(
        jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32"
    )


def _llama4_rope_freq_correction(rotational_frequency: jax.Array, cfg: Config):
    factor = cfg.rope_scaling_factor  # `8` in the original implementation
    low_freq_factor = (
        cfg.rope_scaling_low_freq_factor
    )  # `1` in the original implementation
    high_freq_factor = (
        cfg.rope_scaling_high_freq_factor
    )  # `4` in the original implementation
    old_context_len = (
        cfg.rope_scaling_original_max_position_embeddings
    )  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / rotational_frequency
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = jnp.where(
        wavelen > low_freq_wavelen, rotational_frequency / factor, rotational_frequency
    )
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


def _generate_pos_embeddings(
    # positions: jax.Array, features: int, min_timescale=1.0, max_timescale=16384.0
    positions: jax.Array,
    features: int,
    cfg: Config,
) -> tuple[jax.Array, jax.Array]:
    """Generate Sin/Cos for Rotary Embeddings.

    Generates sinusoids at (features//2) different timescales, where the
    timescales form a geometric series from min_timescale to max_timescale
    (max_timescale is not included, but would be the next element in the series).

    Sinusoids are evaluated at integer positions i in [0, length).

    The outputs are computed as:


    sin[b, t, j] = sin(rope_pos[b, t] / timescale[j])
    cos[b, t, j] = cos(rope_pos[b, t] / timescale[j])

    Args:
        postions: [batch, time]
        features: d_head.
        min_timescale: an optional float
        max_timescale: an optional float

    Returns:
        output_sin: a float32 Tensor with shape [length, features // 2]
        output_cos: a float32 Tensor with shape [length, features // 2]
    """
    # Forked from: flaxformer/components/embedding.py;l=592
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = cfg.rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    rotational_frequency = _llama4_rope_freq_correction(rotational_frequency, cfg)
    # Must use high precision einsum here, since rounding off to a bfloat16 is catastrophic. bfloat16 rounds 257 to 256,
    # but sin(257) is very different from sin(256).
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
        out_sharding=P(None, None, None),
    )
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rotary_embedding(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x_ = x.reshape(x.shape[:-1] + (-1, 2))
    x1, x2 = x_[..., 0], x_[..., 1]
    # [B, T, head_dim] -> [B, h, T, head_dim]
    sin, cos = sin[:, None, :, :], cos[:, None, :, :]
    return jnp.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).reshape(
        x.shape
    )


def make_attention_mask(
    q_len,
    k_len,
    q_segment_ids,
    kv_segment_ids,
    q_offset,
    kv_offset,
    causal: bool,
    chunk_attn_size: int | None = None,
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
        if chunk_attn_size is not None:
            chunk_attn_mask = (
                q_positions // chunk_attn_size == kv_positions // chunk_attn_size
            )
            return segment_mask & causal_mask & chunk_attn_mask
        return segment_mask & causal_mask
    return segment_mask


@partial(auto_axes, out_sharding=P(BATCH_AXIS_NAME, ATTN_HEADS_AXIS_NAME, None, None))
def attention(
    q: jax.Array,
    k: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_offset: jax.Array,
    kv_offset: jax.Array,
    starts: jax.Array,
    lengths: jax.Array,
    cfg: Config,
    attn_chunk_size: int | None = None,
) -> jax.Array:
    """Compute attention.

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
    scale = cfg.head_dim**-0.5

    # grouped-query attention
    b, qh, t, d = q.shape
    _, kh, T, _ = k.shape

    q_ = q.reshape((b, kh, qh // kh, t, d))
    qk = einsum("bhgtd,bhTd->bhgtT", q_, k) * scale
    qk = qk.reshape((b, qh, t, T))

    del lengths
    mask = make_attention_mask(
        t,
        T,
        q_segment_ids,
        kv_segment_ids,
        q_offset,
        kv_offset,
        cfg.causal,
        attn_chunk_size,
    )

    # Apply the combined mask
    qk = jnp.where(mask, qk, -1e30)
    # jax softmax impl includes max subtraction for numerical stability, no need to do it outside.
    attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)

    # grouped-query attention
    attn_ = attn.reshape((b, kh, qh // kh, t, T))
    qkv = einsum("bhgtT,bhTd->bhgtd", attn_, v).astype(cfg.dtype)
    return qkv.reshape((b, qh, t, d))


def attention_kernel(
    q,
    k,
    v,
    q_segment_ids,
    kv_segment_ids,
    q_offset,
    kv_offset,
    starts,
    lengths,
    cfg: Config,
    attn_chunk_size: int | None = None,
):
    """Flash attention kernel!"""
    # On TPUv3, pallas seems to only work with float32.
    # q, k, v = jnp.float32(q), jnp.float32(k), jnp.float32(v)

    k, k_scale = (k.quant, k.scale) if is_type(k, QuantArray) else (k, None)
    v, v_scale = (v.quant, v.scale) if is_type(v, QuantArray) else (v, None)

    # handle grouped query attention
    assert q.shape[-3] % k.shape[-3] == 0
    scale = q.shape[-1] ** -0.5

    kv_repeats = q.shape[-3] // k.shape[-3]
    l2p = lambda *logical: logical_to_physical(logical, cfg.rules)
    q_spec = P(
        *(
            l2p("batch", "kv_heads")
            + tuple(set(*l2p("q_heads")) - set(*l2p("kv_heads")))
            + l2p("sequence", "head_dim")
        )
    )
    q_shape__ = q.shape
    q = jax.lax.reshape(
        q,
        (q.shape[:-3] + (k.shape[-3], kv_repeats, q.shape[-2], q.shape[-1])),
        out_sharding=q_spec,
    )

    l2p = lambda *logical: logical_to_physical(logical, cfg.rules)

    # shard_map
    in_specs = (
        q_spec,
        l2p("batch", "kv_heads", "sequence", "head_dim"),
        l2p("batch", "kv_heads", "sequence", "head_dim"),
        l2p("batch", "sequence"),
        l2p("batch", "sequence"),
        l2p("batch"),
        l2p("batch"),
    )
    in_specs += (None if k_scale is None else l2p("batch", "kv_heads", "sequence"),)
    in_specs += (None if v_scale is None else l2p("batch", "kv_heads", "sequence"),)
    out_specs = q_spec

    @partial(
        jax.shard_map,
        mesh=cfg.mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    def _f(q, k, v, q_segment_ids, kv_segment_ids, starts, lengths, k_scale, v_scale):
        q_org_shape = q.shape
        # kv_repeats = q.shape[-3] // k.shape[-3]
        # q = q.reshape(q.shape[:-3] + (k.shape[-3], kv_repeats, q.shape[-2], q.shape[-1]))

        if q.shape[-2] != 1:
            mask = mask_lib.CausalMask((q.shape[-2], k.shape[-2]))
            if attn_chunk_size is not None:
                # map segment ids to enforce chunk local attention
                # this is incompatible with sequence packing, we're assuming 1 segment per row
                local_q_segment_ids = (
                    (jnp.arange(q_segment_ids.shape[-1]) - starts[:, None])
                    // attn_chunk_size
                ) + 1
                q_segment_ids = jnp.where(
                    (q_segment_ids != 0) & (local_q_segment_ids > 0), q_segment_ids, 0
                )
                local_kv_segment_ids = (
                    (jnp.arange(kv_segment_ids.shape[-1]) - starts[:, None])
                    // attn_chunk_size
                ) + 1
                kv_segment_ids = jnp.where(
                    (kv_segment_ids != 0) & (local_kv_segment_ids > 0),
                    local_kv_segment_ids,
                    0,
                )
                # add sliding window attention mask to lower bound the chunk local attention sparsity
                local_window_mask = mask_lib.LocalMask(
                    (q.shape[-2], k.shape[-2]), (attn_chunk_size, None), 0
                )
                mask = mask_lib.LogicalAnd(mask, local_window_mask)
            mask = mask_lib.MultiHeadMask([mask for _ in range(q.shape[-3])])
            segment_ids = splash.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
            block_q, block_kv = min(q.shape[-2], 512), min(k.shape[-2], 1024)
            block_sizes = splash.BlockSizes(
                block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv
            )
            attn_fn = splash.make_splash_mqa_single_device(
                mask=mask, block_sizes=block_sizes
            )
            attn_fn = jax.vmap(
                jax.vmap(attn_fn, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)
            )
            if k_scale is not None:
                k = (k * k_scale[..., None]).astype(jnp.bfloat16)
            if v_scale is not None:
                v = (v * v_scale[..., None]).astype(jnp.bfloat16)
            ret = attn_fn(q * scale, k, v, segment_ids)
        else:
            assert q.shape[-2] == 1, "This is a decode kernel, q.shape[-2] must be 1"
            q = q[..., 0, :]
            in_axes = (1, 1, 1, None, None)
            in_axes += ((None if k_scale is None else 1),)
            in_axes += ((None if v_scale is None else 1),)
            hyperparams = dict(scale=scale, block_kv=512, block_bs=32)
            if attn_chunk_size:
                # bring starts up to the beginning of the current block
                starts = jnp.maximum(
                    starts,
                    ((lengths - starts) // attn_chunk_size) * attn_chunk_size + starts,
                )
            ret = jax.vmap(
                partial(ragged_attention.ragged_decode_fwd, **hyperparams),
                in_axes=in_axes,
                out_axes=1,
            )(q, k, v, starts, lengths, k_scale, v_scale)
        return ret.reshape(q_org_shape)

    lengths = jnp.broadcast_to(lengths, starts.shape)
    ret = _f(
        q, k, v, q_segment_ids, kv_segment_ids, starts, lengths, k_scale, v_scale
    ).astype(jnp.bfloat16)
    return jax.lax.reshape(
        ret, q_shape__, out_sharding=l2p("batch", "q_heads", "sequence", "head_dim")
    )


def rms_norm(
    x: jax.Array, gamma: jax.Array | None, eps: jax.Array | float
) -> jax.Array:
    """Apply RMS normalization."""
    rms = jnp.sqrt(
        jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + eps
    )
    return jnp.astype((gamma if gamma is not None else 1) * x / rms, jnp.bfloat16)


def attention_block(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    sin: jax.Array,
    cos: jax.Array,
    cfg: Config,
    cache: KVCache | None = None,
    idx: int | None = None,
):
    assert idx is not None
    l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
    x = x.astype(cfg.dtype)

    # Multi-head attention
    with jax.named_scope("qkv_matmul"):
        q = einsum("btd,dhq->bhtq", x, layer.q).astype(cfg.dtype)
        k = einsum("btd,dhq->bhtq", x, layer.k).astype(cfg.dtype)
        v = einsum("btd,dhq->bhtq", x, layer.v).astype(cfg.dtype)

    # Apply rotary embeddings
    with jax.named_scope("rope"):
        is_nope = (idx + 1) % cfg.nope_layer_interval == 0
        if not is_nope:
            q, k = (
                apply_rotary_embedding(q, sin, cos),
                apply_rotary_embedding(k, sin, cos),
            )
            if cfg.use_qk_norm:
                q, k = rms_norm(q, None, cfg.norm_eps), rms_norm(k, None, cfg.norm_eps)

    with jax.named_scope("cache_update"):
        if is_type(cache, KVCache):
            it = jnp.maximum(cache.iter, 0)
            k = update_slice(
                cache.k[idx], k, it, update_axis=cache.time_axis, quant_axis=-1
            )
            v = update_slice(
                cache.v[idx], v, it, update_axis=cache.time_axis, quant_axis=-1
            )
            cache_updates = (k, v)

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
            cache_updates = (k, v)

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
        attn_chunk_size = None if not is_nope else cfg.attn_chunk_size
        if (cfg.use_prefill_attn_kernel and q.shape[-2] != 1) or (
            cfg.use_decode_attn_kernel and q.shape[-2] == 1
        ):
            attn_out = attention_kernel(
                *attn_args, cfg=cfg, attn_chunk_size=attn_chunk_size
            )
        else:
            attn_out = attention(*attn_args, cfg, attn_chunk_size=attn_chunk_size)

    # Project attention output
    with jax.named_scope("projection"):
        attn_out = einsum(
            "bhtq,hqd->btd",
            attn_out,
            layer.o,
            out_sharding=l2p("batch", "sequence", "act_embed"),
        ).astype(cfg.dtype)
    return attn_out, cache_updates


@partial(jax.jit, static_argnames=("replicated_routing",))
def _route_tokens_to_moe_experts(
    x: jax.Array, weight: jax.Array, replicated_routing: bool, cfg: Config
):
    lsc = lambda x, spec: reshard(x, logical_to_physical(spec, cfg.rules))
    x_shape = x.shape
    x = x.reshape((-1, x.shape[-1]))
    if (
        replicated_routing
    ):  # not distributing the routing work avoids communication for small batches
        x = lsc(x, (None, None))
    else:
        x = reshard(x, P(TENSOR_AXIS_NAME, None))
    weight = lsc(weight, (None, None))

    scores = jnp.einsum("Sk,kj->Sj", x, weight).astype(cfg.moe_gate_dtype)
    topk_weights, topk_idx = jax.lax.top_k(scores, cfg.moe_experts_per_tok)
    topk_weights = jax.nn.sigmoid(topk_weights)
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
    if (
        cfg.use_ragged_dot_kernel
        and lhs.shape[0] <= 1024
        and which_platform(cfg) == "tpu"
    ):
        with jax.named_scope("decode_ragged_dot"):
            block_g, block_n = min(8, rhs.shape[0]), min(128, lhs.shape[0])
            if is_type(rhs, QuantArray):
                assert rhs.scale.ndim == 2 and rhs.scale.shape == (
                    rhs.quant.shape[0],
                    rhs.quant.shape[2],
                )
                scale = jnp.take_along_axis(rhs.scale, topk_idx[:, None], axis=-2)
                ret = decode_ragged_dot(
                    lhs,
                    rhs.quant,
                    group_sizes,
                    block_g=block_g,
                    block_n=block_n,
                    interpret=False,
                )
                ret = ret * scale
            else:
                ret = decode_ragged_dot(
                    lhs,
                    rhs,
                    group_sizes,
                    block_g=block_g,
                    block_n=block_n,
                    interpret=False,
                )
    else:
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


def moe_block_ep(x: jax.Array, layer: MoELayer, cfg: Config):
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
        x, layer.w_router, replicated_routing, cfg
    )
    tensor_axname, expert_axname = l2p("moe_e_tp")[0], l2p("moe_e_ep")[0]

    x_spec = l2p("batch", "sequence", None)
    topk_weights_spec, topk_idx_spec = (
        l2p("batch", "sequence", None),
        l2p("batch", "sequence", None),
    )
    out_spec = l2p("batch", "sequence", None)

    we_gate_spec = l2p("moe_e_ep", None, "moe_e_tp")
    we_up_spec = l2p("moe_e_ep", None, "moe_e_tp")
    we_down_spec = l2p("moe_e_ep", "moe_e_tp", None)
    if all(is_type(z, QuantArray) for z in [layer.we_gate, layer.we_up, layer.we_down]):
        we_gate_spec = dataclasses.replace(
            layer.we_gate, quant=we_gate_spec, scale=P(we_gate_spec[0], we_gate_spec[2])
        )
        we_up_spec = dataclasses.replace(
            layer.we_up, quant=we_up_spec, scale=P(we_up_spec[0], we_up_spec[2])
        )
        we_down_spec = dataclasses.replace(
            layer.we_down, quant=we_down_spec, scale=P(we_down_spec[0], we_down_spec[2])
        )
    we_gate = psc(layer.we_gate, we_gate_spec)
    we_up = psc(layer.we_up, we_up_spec)
    we_down = psc(layer.we_down, we_down_spec)

    in_specs = (
        x_spec,
        we_gate_spec,
        we_up_spec,
        we_down_spec,
        topk_weights_spec,
        topk_idx_spec,
    )

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
    def _expert_fn(x, we_gate, we_up, we_down, topk_weights, topk_idx):
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
        x_repeat_sort_ = x_repeat_sort_ * topk_weights.reshape(-1)[sort_idx_][..., None]

        group_sizes = jnp.bincount(topk_idx_sort_, length=cfg.moe_num_experts)
        group_sizes_shard = jax.lax.dynamic_slice_in_dim(
            group_sizes, expert_idx * expert_size, expert_size, 0
        )

        with jax.named_scope("we_gate"):
            ff_gate = _moe_gmm(
                x_repeat_sort_,
                we_gate,
                group_sizes_shard,
                expert_mapped_topk_idx_sort_,
                cfg,
            )
            ff_gate = jax.nn.silu(ff_gate)
            ff_gate = jnp.where(valid_group_mask_sort_[..., None], ff_gate, 0)
        with jax.named_scope("we_up"):
            ff_up = _moe_gmm(
                x_repeat_sort_,
                we_up,
                group_sizes_shard,
                expert_mapped_topk_idx_sort_,
                cfg,
            )
        ff_gate_up = jnp.where(valid_group_mask_sort_[..., None], ff_gate * ff_up, 0)
        with jax.named_scope("we_down"):
            ff_out = _moe_gmm(
                ff_gate_up,
                we_down,
                group_sizes_shard,
                expert_mapped_topk_idx_sort_,
                cfg,
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
        ff_out_expert = _expert_fn(x_, we_gate, we_up, we_down, topk_weights, topk_idx)[
            ..., : x.shape[-1]
        ]
    with jax.named_scope("moe_shared_expert"):
        ff_out_shared = mlp_block(
            x, MLPLayer(layer.ws_gate, layer.ws_up, layer.ws_down), cfg
        )[..., : x.shape[-1]]
    return psc(ff_out_expert + ff_out_shared, l2p("batch", "sequence", "act_embed"))


def mlp_block(x: jax.Array, layer: Layer, cfg: Config):
    l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
    dtype = cfg.dtype
    with jax.named_scope("gate"):
        ff_gate = jax.nn.silu(einsum("btd,df->btf", x, layer.w_gate)).astype(dtype)
    with jax.named_scope("up_proj"):
        ff_up = einsum("btd,df->btf", x, layer.w_up).astype(dtype)
    with jax.named_scope("down_proj"):
        ff_out = einsum(
            "btf,fd->btd",
            ff_gate * ff_up,
            layer.w_down,
            out_sharding=l2p("batch", "sequence", "act_embed"),
        ).astype(dtype)
    return ff_out


def forward_layer(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    sin: jax.Array,
    cos: jax.Array,
    idx: int,
    cfg: Config,
    cache: KVCache | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = x.astype(cfg.dtype)

    # Attention block
    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.attn_pre_gamma, cfg.norm_eps)
    attn_out, cache_updates = attention_block(
        attn_in, segment_ids, layer.attn, sin, cos, cfg, cache, idx
    )
    with jax.named_scope("residual"):
        x = x + attn_out.astype(cfg.dtype)

    # FFN block
    with jax.named_scope("attn_post_norm"):
        ff_in = rms_norm(x, layer.attn_post_gamma, cfg.norm_eps)
    with jax.named_scope("ffn"):
        ff_out = (mlp_block if is_type(layer.mlp, MLPLayer) else moe_block_ep)(
            ff_in, layer.mlp, cfg
        )
    with jax.named_scope("residual"):
        x = x + ff_out.astype(cfg.dtype)

    return x, cache_updates


def forward(
    x: jax.Array,
    segment_ids: jax.Array,
    weights: Weights,
    cfg: Config,
    cache: KVCache | None = None,
):
    l2p = lambda *args: logical_to_physical(args, cfg.rules)
    # Embed input tokens [B, T] -> [B, T D]
    x = weights.embedding.at[x, :].get(
        out_sharding=l2p("batch", "sequence", "act_embed")
    )
    positions = segment_ids_to_positions(segment_ids)
    if is_type(cache, KVCache):
        positions = positions + cache.fill_len()[:, None]
    sin, cos = _generate_pos_embeddings(
        positions, cfg.head_dim, cfg
    )  # [B, T, head_dim]
    sin, cos = sin.astype(cfg.dtype), cos.astype(cfg.dtype)

    all_cache_updates = []
    for idx, layer in enumerate(weights.layers):
        x, cache_updates = forward_layer(
            x, segment_ids, layer, sin, cos, idx, cfg, cache
        )
        all_cache_updates.append(cache_updates)

    x = rms_norm(x, weights.gamma_final, cfg.norm_eps)  # Final layer norm.
    logits = einsum("btd,dv->btv", x, weights.lm_head)  # Project to vocabulary size
    if is_type(cache, KVCache):
        cache.k, cache.v = ([z[i] for z in all_cache_updates] for i in range(2))
        additional_tokens = jnp.max(_length_minus_right_padding(segment_ids))
        return logits, dataclasses.replace(
            cache, iter=(jnp.maximum(0, cache.iter) + additional_tokens) % cache.size
        )
    else:
        return logits, all_cache_updates


# serialization
def save_pytree(data, path):
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(
            epath.Path(path),
            data,
            ocp.args.PyTreeSave(data, ocdbt_target_data_file_size=1024 * 1024 * 500),
        )


def load_pytree(path, sharding=None):
    import orbax.checkpoint as ocp

    item, transforms = sharding, None
    restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            epath.Path(path),
            args=ocp.args.PyTreeRestore(
                item=item, transforms=transforms, restore_args=restore_args
            ),
        )


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

    cache_shardings = KVCache.shardings(cfg, prompt.shape[0], cfg.max_seq_len)
    if is_type(cache, KVCache):
        uninitialized_iter = -jnp.ones_like(cache.iter)
        cache = dataclasses.replace(
            cache,
            starts=_count_left_padding(prompt, pad_id=pad_id),
            iter=uninitialized_iter,
        )
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
    next_tokens = reshard(next_tokens, P())
    return next_tokens, cache
