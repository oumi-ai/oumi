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
from collections import OrderedDict as odict
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax import random, tree_util
from jax.experimental.array_serialization import pytree_serialization as ser
from jax.experimental.pallas.ops.gpu import paged_attention
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
from jax.sharding import PartitionSpec as P
from jax.sharding import auto_axes, reshard

from . import ragged_attention

AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]

# Expected physical mesh axis names:
# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
PARTIAL_TENSOR_AXIS_NAME = "y"
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
    kv_heads: AxisName = PARTIAL_TENSOR_AXIS_NAME
    o_heads: AxisName = TENSOR_AXIS_NAME
    o_embed: AxisName = None
    # MLP
    embed_up: AxisName = None
    ffw_up: AxisName = TENSOR_AXIS_NAME
    ffw_down: AxisName = TENSOR_AXIS_NAME
    embed_down: AxisName = None
    # vocab
    vocab_in: AxisName = None
    vocab_out: AxisName = TENSOR_AXIS_NAME


def logical_to_physical(
    logical: Axes, rules: ShardingRules
) -> jax.sharding.PartitionSpec:
    """Translate logical to physically sharding."""
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
    ffw_size: int
    q_heads: int
    kv_heads: int
    num_layers: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    causal: bool
    use_prefill_attn_kernel: bool
    use_decode_attn_kernel: bool
    dtype: "jnp.dtype" = jnp.bfloat16
    # sharding
    rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
    mesh: jax.sharding.Mesh | None = None
    # Llama 3 specific frequency computation
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 8.0
    rope_scaling_low_freq_factor: float = 1.0
    rope_scaling_high_freq_factor: float = 4.0
    rope_scaling_original_max_position_embeddings: int = 8192
    quant_layer: bool = True
    quant_cache: bool = True
    quant_scale_dtype: "jnp.dtype" = jnp.float16


def llama_to_jax_config(llama_config: Any | dict[str, Any]) -> "Config":
    _get = lambda x, k, default=None: (
        getattr(x, k, default) if hasattr(x, k) else dict(x).get(k, default)
    )
    return Config(
        embed=_get(llama_config, "hidden_size"),
        ffw_size=_get(llama_config, "intermediate_size"),
        q_heads=_get(llama_config, "num_attention_heads"),
        kv_heads=_get(llama_config, "num_key_value_heads"),
        num_layers=_get(llama_config, "num_hidden_layers"),
        head_dim=_get(llama_config, "head_dim", 128),
        vocab_size=_get(llama_config, "vocab_size"),
        max_seq_len=128,
        dtype=jnp.bfloat16,
        causal=True,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
        rope_theta=_get(llama_config, "rope_theta"),
        rope_scaling_factor=_get(llama_config, "rope_scaling")["factor"],
        rope_scaling_low_freq_factor=_get(llama_config, "rope_scaling")[
            "low_freq_factor"
        ],
        rope_scaling_high_freq_factor=_get(llama_config, "rope_scaling")[
            "high_freq_factor"
        ],
        rope_scaling_original_max_position_embeddings=_get(
            llama_config, "rope_scaling"
        )["original_max_position_embeddings"],
    )


def load_config(config_path: str | os.PathLike[str] | Path) -> "Config":
    return llama_to_jax_config(json.loads(Path(config_path).read_text()))


PreTrainedTokenizerFast = TypeVar("PreTrainedTokenizerFast")


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path,
    tokenizer_config_path: str | os.PathLike[str] | Path,
) -> PreTrainedTokenizerFast:
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
which_platform = lambda cfg: cfg.mesh.devices.reshape(-1)[0].platform
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
    scale_dtype=jnp.float16,
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
    y: jax.Array | QuantArray,
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
        y_quant, y_scale = y.quant, y.scale
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
        )
        return dataclasses.replace(x, quant=new_quant, scale=new_scale)
    else:
        assert x.ndim == y.ndim
        y = reshard(y.astype(x.dtype), jax.typeof(x).sharding.spec)
        return jax.lax.dynamic_update_slice_in_dim(x, y, pos, axis=update_axis)


@jax_pytree_struct
class Layer(_Init):
    q: jax.Array | ArrayInfo | QuantArray
    k: jax.Array | ArrayInfo | QuantArray
    v: jax.Array | ArrayInfo | QuantArray
    o: jax.Array | ArrayInfo | QuantArray
    w_gate: jax.Array | ArrayInfo | QuantArray
    w_up: jax.Array | ArrayInfo | QuantArray
    w_down: jax.Array | ArrayInfo | QuantArray
    attn_pre_gamma: jax.Array | ArrayInfo
    attn_post_gamma: jax.Array | ArrayInfo

    @classmethod
    def abstract(cls, cfg: Config) -> "Layer":
        _init = _he_normal
        layer = Layer(
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
            w_gate=ArrayInfo(
                (cfg.embed, cfg.ffw_size),
                cfg.dtype,
                ("embed_up", "ffw_up"),
                _init(0, 1),
            ),
            w_up=ArrayInfo(
                (cfg.embed, cfg.ffw_size),
                cfg.dtype,
                ("embed_up", "ffw_up"),
                _init(0, 1),
            ),
            w_down=ArrayInfo(
                (cfg.ffw_size, cfg.embed),
                cfg.dtype,
                ("ffw_down", "embed_down"),
                _init(0, 1),
            ),
            attn_pre_gamma=ArrayInfo(
                (cfg.embed,), cfg.dtype, ("act_embed",), jax.nn.initializers.ones
            ),
            attn_post_gamma=ArrayInfo(
                (cfg.embed,), cfg.dtype, ("act_embed",), jax.nn.initializers.ones
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "Layer", cfg: Config):
        if not cfg.quant_layer:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            q=QuantArray(*quantize(layer.q, (1, 2), scale_dtype)),
            k=QuantArray(*quantize(layer.k, (1, 2), scale_dtype)),
            v=QuantArray(*quantize(layer.v, (1, 2), scale_dtype)),
            o=QuantArray(*quantize(layer.o, (0, 1), scale_dtype), out_scaling=True),
            w_gate=QuantArray(
                *quantize(layer.w_gate, 0, scale_dtype), out_scaling=True
            ),
            w_up=QuantArray(*quantize(layer.w_up, 0, scale_dtype), out_scaling=True),
            w_down=QuantArray(
                *quantize(layer.w_down, 0, scale_dtype), out_scaling=True
            ),
        )


@jax_pytree_struct
class Weights(_Init):
    layers: list[Layer]
    embedding: jax.Array | ArrayInfo
    gamma_final: jax.Array | ArrayInfo
    lm_head: jax.Array | ArrayInfo

    @classmethod
    def abstract(cls, cfg: Config):
        layers = [Layer.abstract(cfg) for _ in range(cfg.num_layers)]
        init = _he_normal
        return Weights(
            layers=layers,
            embedding=ArrayInfo(
                (cfg.vocab_size, cfg.embed), cfg.dtype, (None, "vocab_in"), init(0, 1)
            ),
            gamma_final=ArrayInfo(
                (cfg.embed,), cfg.dtype, ("act_embed",), jax.nn.initializers.ones
            ),
            lm_head=ArrayInfo(
                (cfg.embed, cfg.vocab_size),
                cfg.dtype,
                ("vocab_in", "vocab_out"),
                init(1, 0),
            ),
        )


@partial(
    jax_pytree_struct,
    meta_fields=("batch_size", "size", "time_axis", "insert_sequences"),
)
class KVCache(_Init):
    k: list[
        tuple[jax.Array | QuantArray, ...]
    ]  # (batch_size, key_heads, max_seq_len, head_dim)
    v: list[
        tuple[jax.Array | QuantArray, ...]
    ]  # (batch_size, key_heads, max_seq_len, head_dim)
    iter: jax.Array  # []  # sequences are right-aligned for slice update performance
    starts: (
        jax.Array
    )  # [batch_size]  # sequences are right-aligned, we need start indices
    batch_size: int = 1
    size: int = 2**30
    time_axis: int = 2
    # update_slice: Callable = None
    insert_sequences: Callable = None
    # get_sequence: Callable = None

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int):
        val_info = ArrayInfo(
            (batch_size, cfg.kv_heads, cfg.max_seq_len, cfg.head_dim),
            cfg.dtype,
            ("batch", "kv_heads", "sequence", "head_dim"),
            jax.nn.initializers.zeros,
        )
        cache = KVCache(
            k=[val_info for _ in range(cfg.num_layers)],
            v=[val_info for _ in range(cfg.num_layers)],
            # -1 means unintialized since iter (cursor) must be 0 <= iter < len - 1
            iter=ArrayInfo((), jnp.int32, (), _const_init(-1)),
            starts=ArrayInfo(
                (batch_size,), jnp.int32, ("batch",), jax.nn.initializers.zeros
            ),
            size=cfg.max_seq_len,
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
                    for idx in range(cfg.num_layers)
                ],
                v=[
                    QuantArray(
                        *_quantize(cache.v[idx]),
                        out_scaling=False,
                        scale_expand_dims=(-2, -3),
                    )
                    for idx in range(cfg.num_layers)
                ],
            )

        cache.batch_size, cache.size = batch_size, cfg.max_seq_len
        return cache

    def fill_len(self) -> jax.Array:
        return jnp.where(self.iter >= 0, (self.iter - self.starts) % self.size, 0)

    @property
    def buffers(self) -> tuple[jax.Array, ...]:
        return (self.k, self.v)

    # update_slice = None
    # insert_sequences = staticmethod(attention_cache_utils.kvcache_update_cache)
    # get_sequence = staticmethod(attention_cache_utils.kvcache_get_entry)


@partial(jax_pytree_struct, meta_fields=("batch_size", "size", "page_size"))
class PagedKVCache(_Init):
    k: list[jax.Array | QuantArray]  # [key_heads, total_num_pages, page_size, head_dim]
    v: list[jax.Array | QuantArray]  # [key_heads, total_num_pages, page_size, head_dim]
    lengths: jax.Array  # [batch_size]  # true length of the cache entries
    block_tables: jax.Array  # [batch_size, pages_per_seq]
    free_pages: jax.Array  # [total_num_pages]
    batch_size: int = 0
    size: int = 2**30
    page_size: int = 0

    @classmethod
    def abstract(
        cls, cfg: "Config", batch_size: int, total_num_pages: int, page_size: int
    ):
        pages_per_seq = math.ceil(cfg.max_seq_len / page_size)
        val_info = ArrayInfo(
            (cfg.kv_heads, total_num_pages, page_size, cfg.head_dim),
            cfg.dtype,
            ("kv_heads", None, None, "head_dim"),
            jax.nn.initializers.zeros,
        )
        cache = PagedKVCache(
            k=[val_info for _ in range(cfg.num_layers)],
            v=[val_info for _ in range(cfg.num_layers)],
            lengths=ArrayInfo((batch_size,), jnp.int32, (), jax.nn.initializers.zeros),
            block_tables=ArrayInfo(
                (batch_size, pages_per_seq), jnp.int32, (), jax.nn.initializers.zeros
            ),
            free_pages=ArrayInfo(
                (total_num_pages,), jnp.bool, (), jax.nn.initializers.ones
            ),
        )
        if cfg.quant_cache:
            _quantize = partial(quantize, axis=-1, scale_dtype=cfg.quant_scale_dtype)
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
        cache.batch_size, cache.page_size = batch_size, page_size
        return cache

    def fill_len(self) -> jax.Array:
        return self.lengths

    @property
    def buffers(self) -> tuple[jax.Array, ...]:
        return (self.k, self.v)

    # update_slice = staticmethod(paged_update_slice)
    # insert_sequences = staticmethod(attention_cache_utils.batch_paged_update_sequences)
    # get_sequence = staticmethod(attention_cache_utils.batch_paged_get_entry)

    @staticmethod
    def _find_empty_pages(
        free_pages: jax.Array, k: int, proposal_pages: jax.Array | None = None
    ):
        if proposal_pages is not None:
            assert proposal_pages.size == k
            proposal_mask = free_pages[proposal_pages]
            indicies = jnp.where(
                ~proposal_mask, jnp.cumsum(~proposal_mask, axis=-1) - 1, k - 1
            )
            newly_free_pages = free_pages.at[
                jnp.where(proposal_mask, proposal_pages, 2**30)
            ].set(False, mode="drop")
            return jnp.where(
                proposal_mask,
                proposal_pages,
                jax.lax.top_k(newly_free_pages, k)[1][indicies],
            )
        else:
            return jax.lax.top_k(free_pages, k)[1]

    @staticmethod
    def _paged_update_slice(
        cache, kv: tuple[jax.Array | QuantArray, ...], *, layer_idx: int
    ):
        # key_heads = cache.buffers[0][layer_idx].shape[0]
        # assert v.shape[:-1] == k.shape[:-1] == (cache.batch_size, key_heads, 1)  # TODO write this generically
        needs_next_page = (cache.lengths % cache.page_size) == 0
        page_table_idx = cache.lengths // cache.page_size
        current_page_cursor = jnp.take_along_axis(
            cache.block_tables, page_table_idx[:, None], axis=-1
        )[..., 0]
        avg_pages_per_batch_entry = round(
            cache.buffers[0][layer_idx].shape[0] / cache.batch_size
        )
        even_batch_spread = jnp.arange(cache.batch_size) * avg_pages_per_batch_entry
        proposal_pages = jnp.where(
            cache.lengths == 0, even_batch_spread, current_page_cursor + 1
        )
        free_pages = PagedKVCache._find_empty_pages(
            cache.free_pages, cache.batch_size, proposal_pages=proposal_pages
        )
        page_cursor = jnp.where(needs_next_page, free_pages, current_page_cursor)

        inpage_cursor = cache.lengths % cache.page_size

        new_lengths = cache.lengths + 1
        # for batch index update the target slice is (heads, i, j, head_dim)
        # so transpose update (batch, heads, seq, head_dim) -> (batch, heads, head_dim) -> (heads, batch, head_dim)
        _update = lambda dest, src: dest.at[:, page_cursor, inpage_cursor, ...].set(
            src.squeeze(2).swapaxes(0, 1)
        )
        for buffer, new_buffer in zip(cache.buffers, kv):
            buffer[layer_idx] = jax.tree.map(_update, buffer[layer_idx], new_buffer)

        batch_idx = jnp.arange(cache.batch_size)
        new_block_tables = cache.block_tables.at[
            batch_idx, new_lengths // cache.page_size
        ].set(page_cursor)

        new_free_pages = cache.free_pages.at[page_cursor].set(False, mode="drop")
        new_state = dict(
            lengths=new_lengths,
            block_tables=new_block_tables,
            free_pages=new_free_pages,
        )
        return tuple(buffer[layer_idx] for buffer in cache.buffers), new_state

    @staticmethod
    def update_slice(cache, kv: tuple[jax.Array | QuantArray, ...], *, layer_idx: int):
        repl_sharding = jax.typeof(cache.lengths).sharding
        kv_sharding = jax.tree.map(
            lambda x: jax.typeof(x).sharding,
            tuple(buffer[layer_idx] for buffer in cache.buffers),
        )
        sharding = (
            kv_sharding,
            dict(
                lengths=repl_sharding,
                block_tables=repl_sharding,
                free_pages=repl_sharding,
            ),
        )
        return auto_axes(
            partial(PagedKVCache._paged_update_slice, layer_idx=layer_idx),
            out_sharding=sharding,
        )(cache, kv)


def segment_ids_to_positions(segment_ids):
    """Counts positions for segment ids."""

    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

    vals = (jnp.zeros_like(segment_ids), segment_ids)
    return jnp.array(
        jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32"
    )


def _llama3_rope_freq_correction(rotational_frequency: jax.Array, cfg: Config):
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
    rotational_frequency = _llama3_rope_freq_correction(rotational_frequency, cfg)
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


def apply_rotary_embedding(x, sin, cos):
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = jnp.split(x, 2, axis=-1)
    # [B, T, head_dim] -> [B, h, T, head_dim]
    sin, cos = sin[:, None, :, :], cos[:, None, :, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def rms_norm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    """Apply RMS normalization."""
    rms = jnp.sqrt(
        jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6
    )
    return jnp.astype(gamma * x / rms, jnp.bfloat16)


def make_attention_mask(
    q_len,
    k_len,
    q_segment_ids,
    kv_segment_ids,
    q_offset,
    causal: bool,
    cache_starts: jax.Array,
):
    cache_size = kv_segment_ids.shape[-1]
    # [B, t, T]
    segment_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
    # [B, t, T] -> [B, 1, t, T]
    segment_mask = segment_mask[:, None, :, :]

    if causal:
        # [b, h, t, T]
        qk = (1, 1, q_len, k_len)
        q_positions = (
            jax.lax.broadcasted_iota(jnp.int32, qk, 2) + q_offset[:, None, None, None]
        )
        k_positions = (
            jax.lax.broadcasted_iota(jnp.int32, qk, 3)
            + (-1 * cache_starts)[:, None, None, None]
        ) % cache_size
        causal_mask = q_positions >= k_positions
        combined_mask = jnp.logical_and(segment_mask, causal_mask)
        return combined_mask
    else:
        return segment_mask


@partial(auto_axes, out_sharding=P(BATCH_AXIS_NAME, TENSOR_AXIS_NAME, None, None))
def attention(
    q: jax.Array,
    k: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_offset: jax.Array,
    cache_starts: jax.Array | None,
    cfg: Config,
) -> jax.Array:
    """Compute attention.

    Args:
    q: Query tensor of shape (batch_size, num_heads, q_len, head_dim)
    k: Key tensor of shape (batch_size, num_heads, k_len, head_dim)
    v: Value tensor of shape (batch_size, num_heads, k_len, head_dim)
    q_segment_ids: Query segment IDs of shape (batch_size, q_len)
    kv_segment_ids: Key segment IDs of shape (batch_size, k_len)
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

    mask = make_attention_mask(
        t, T, q_segment_ids, kv_segment_ids, q_offset, cfg.causal, cache_starts
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
    q, k, v, q_segment_ids, kv_segment_ids, q_offset, starts, lengths, cfg: Config
):
    """Flash attention kernel!"""
    # On TPUv3, pallas seems to only work with float32.
    # q, k, v = jnp.float32(q), jnp.float32(k), jnp.float32(v)

    k, k_scale = (k.quant, k.scale) if is_type(k, QuantArray) else (k, None)
    v, v_scale = (v.quant, v.scale) if is_type(v, QuantArray) else (v, None)

    # handle grouped query attention
    assert q.shape[-3] % k.shape[-3] == 0
    scale = q.shape[-1] ** -0.5

    l2p = lambda *logical: logical_to_physical(logical, cfg.rules)

    kv_repeats = q.shape[-3] // k.shape[-3]
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

        if q.shape[-2] != 1:
            mask = mask_lib.MultiHeadMask(
                [
                    mask_lib.CausalMask((q.shape[-2], k.shape[-2]))
                    for _ in range(q.shape[-3])
                ]
            )
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

            segment_ids = splash.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
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


def paged_attention_kernel(q, k, v, block_tables, lengths, cfg: Config):
    if which_platform(cfg) not in ("gpu", "cuda"):
        raise ValueError("Paged attention is only supported on GPU.")
    k, k_scale = (k.quant, k.scale) if is_type(k, QuantArray) else (k, None)
    v, v_scale = (v.quant, v.scale) if is_type(v, QuantArray) else (v, None)

    # handle grouped query attention
    assert q.shape[-3] % cfg.kv_heads == 0 and k.shape[0] == cfg.kv_heads
    scale = q.shape[-1] ** -0.5

    l2p = lambda *logical: logical_to_physical(logical, cfg.rules)

    kv_repeats = q.shape[-3] // cfg.kv_heads
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
        (q.shape[:-3] + (cfg.kv_heads, kv_repeats, q.shape[-2], q.shape[-1])),
        out_sharding=q_spec,
    )

    # shard_map
    in_specs = (
        q_spec,  # q
        l2p("kv_heads", None, "sequence", "head_dim"),  # k / k_quant
        None
        if k_scale is None
        else l2p("kv_heads", None, "sequence"),  # k_scale or None
        l2p("kv_heads", None, "sequence", "head_dim"),  # v / v_quant
        None
        if v_scale is None
        else l2p("kv_heads", None, "sequence"),  # v_scale or None
        l2p("batch", None),  # block_tables
        l2p("batch"),  # lengths
    )
    out_specs = q_spec

    @partial(
        jax.shard_map,
        mesh=cfg.mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    def _f(q, k, k_scale, v, v_scale, block_tables, lengths):
        # q in [batch_size, kv_heads_local, kv_repeats, 1, head_dim]
        if k_scale is not None:
            k = (k * k_scale[..., None]).astype(jnp.bfloat16)
        if v_scale is not None:
            v = (v * v_scale[..., None]).astype(jnp.bfloat16)
        q_ = q[..., 0, :].reshape((q.shape[0], -1, q.shape[-1]))
        ret = paged_attention.paged_attention(q_ * scale, k, v, block_tables, lengths)
        return ret.reshape(q.shape)

    ret = _f(q, k, k_scale, v, v_scale, block_tables, lengths).astype(jnp.bfloat16)
    return jax.lax.reshape(
        ret, q_shape__, out_sharding=l2p("batch", "q_heads", "sequence", "head_dim")
    )


def attention_block(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    sin: jax.Array,
    cos: jax.Array,
    cfg: Config,
    cache: KVCache | PagedKVCache | None = None,
    idx: int | None = None,
):
    l2p = lambda *specs: logical_to_physical(specs, cfg.rules)
    x = x.astype(cfg.dtype)

    # Multi-head attention
    with jax.named_scope("qkv_matmul"):
        q = einsum("btd,dhq->bhtq", x, layer.q).astype(cfg.dtype)
        k = einsum("btd,dhq->bhtq", x, layer.k).astype(cfg.dtype)
        v = einsum("btd,dhq->bhtq", x, layer.v).astype(cfg.dtype)

    # Apply rotary embeddings
    with jax.named_scope("rope"):
        q, k = apply_rotary_embedding(q, sin, cos), apply_rotary_embedding(k, sin, cos)

    if cfg.quant_cache:
        _quantize = partial(quantize, axis=-1, scale_dtype=cfg.quant_scale_dtype)
        k = QuantArray(*_quantize(k), out_scaling=True, scale_expand_dims=(-2, -3))
        v = QuantArray(*_quantize(v), out_scaling=False, scale_expand_dims=(-2, -3))

    with jax.named_scope("cache_update"):
        paged_state, starts = None, None
        if is_type(cache, KVCache):
            it = jnp.maximum(cache.iter, 0)
            k = update_slice(
                cache.k[idx], k, it, update_axis=cache.time_axis, quant_axis=-1
            )
            v = update_slice(
                cache.v[idx], v, it, update_axis=cache.time_axis, quant_axis=-1
            )
            time_indices = (
                jnp.arange(0, v.shape[cache.time_axis])[None, :] - cache.starts[:, None]
            ) % cache.size  # [B, T]

            q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
            incremental_position = jnp.max(_length_minus_right_padding(segment_ids))
            # i.e. valid below where we've written things [B, T]
            kv_segment_ids = (time_indices >= 0) & (
                time_indices < cache.fill_len()[:, None] + incremental_position
            )
            q_offset = cache.fill_len() - _count_left_padding(
                segment_ids, 0
            )  # 0 is the pad "token" for segment_ids
            starts, lengths = cache.starts, cache.fill_len()
            cache_updates = (k, v)
        elif is_type(cache, PagedKVCache):
            cache: PagedKVCache
            (k, v), paged_state = PagedKVCache.update_slice(
                cache, (k, v), layer_idx=idx
            )
            cache_updates = (k, v, paged_state)
        else:
            # this supports prefill only; no support for a ring cache buffer here
            q_segment_ids, kv_segment_ids = segment_ids, segment_ids
            q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)
            starts, lengths = (
                _count_left_padding(segment_ids, 0),
                _length_minus_right_padding(kv_segment_ids),
            )
            cache_updates = (k, v)

    # Compute attention
    with jax.named_scope("attention"):
        if is_type(cache, PagedKVCache):
            attn_out = paged_attention_kernel(
                q, k, v, paged_state["block_tables"], paged_state["lengths"], cfg
            )
        elif (cfg.use_prefill_attn_kernel and q.shape[-2] != 1) or (
            cfg.use_decode_attn_kernel and q.shape[-2] == 1
        ):
            attn_out = attention_kernel(
                q,
                k,
                v,
                q_segment_ids,
                kv_segment_ids,
                q_offset,
                starts=starts,
                lengths=lengths,
                cfg=cfg,
            )
        else:
            attn_out = attention(
                q, k, v, q_segment_ids, kv_segment_ids, q_offset, starts, cfg
            )

    # Project attention output
    with jax.named_scope("projection"):
        attn_out = einsum(
            "bhtq,hqd->btd",
            attn_out,
            layer.o,
            out_sharding=l2p("batch", "sequence", "act_embed"),
        ).astype(cfg.dtype)
    return attn_out, cache_updates


def ffn_block(x: jax.Array, layer: Layer, cfg: Config):
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
    cache: KVCache | PagedKVCache | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = x.astype(cfg.dtype)

    # Attention block
    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.attn_pre_gamma)
    attn_out, cache_updates = attention_block(
        attn_in, segment_ids, layer, sin, cos, cfg, cache, idx
    )
    with jax.named_scope("residual"):
        x = x + attn_out.astype(cfg.dtype)

    # FFN block
    with jax.named_scope("attn_post_norm"):
        ff_in = rms_norm(x, layer.attn_post_gamma)
    with jax.named_scope("ffn"):
        ff_out = ffn_block(ff_in, layer, cfg)
    with jax.named_scope("residual"):
        x = x + ff_out.astype(cfg.dtype)

    return x, cache_updates


def forward(
    x: jax.Array,
    segment_ids: jax.Array,
    weights: Weights,
    cfg: Config,
    cache: KVCache | PagedKVCache | None = None,
):
    l2p = lambda *args: logical_to_physical(args, cfg.rules)
    # Embed input tokens [B, T] -> [B, T D]
    x = weights.embedding.at[x, :].get(
        out_sharding=l2p("batch", "sequence", "act_embed")
    )
    positions = segment_ids_to_positions(segment_ids)  # already shifted by padding
    # Apply rotary embeddings: [B, T, head_dim]
    if cache is not None:
        # For inference with cache, we need to index the positional embeddings
        positions = cache.fill_len()[:, None] + positions
    # NOTE: At inference time this only works for UNPACKED sequences.
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

    x = rms_norm(x, weights.gamma_final)  # Final layer norm.
    logits = einsum("btd,dv->btv", x, weights.lm_head)  # Project to vocabulary size

    if is_type(cache, KVCache):
        cache.k, cache.v = (
            [z[0] for z in all_cache_updates],
            [z[1] for z in all_cache_updates],
        )
        new_iter = (
            jnp.maximum(0, cache.iter)
            + jnp.max(_length_minus_right_padding(segment_ids))
        ) % cache.size
        cache = dataclasses.replace(cache, iter=new_iter)
        return logits, cache
    elif is_type(cache, PagedKVCache):
        kv, new_state = (
            tuple(map(list, zip(*[z[:2] for z in all_cache_updates]))),
            all_cache_updates[0][2],
        )
        cache = dataclasses.replace(cache, k=kv[0], v=kv[1], **new_state)
        return logits, cache
    else:
        return logits, all_cache_updates


# serialization
# def save_pytree(data, path):
#    import orbax.checkpoint as ocp
#
#    with ocp.PyTreeCheckpointer() as ckptr:
#        ckptr.save(epath.Path(path), data, ocp.args.PyTreeSave(data, ocdbt_target_data_file_size=1024 * 1024 * 100))
#
#
# def load_pytree(path, sharding=None):
#    import orbax.checkpoint as ocp
#
#    item, transforms = sharding, None
#    restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
#    with ocp.PyTreeCheckpointer() as ckptr:
#        return ckptr.restore(
#            epath.Path(path), ocp.args.PyTreeRestore(item=item, transforms=transforms, restore_args=restore_args)
#        )


# serialization
def save_pytree(weights, path):
    flat_data = odict(
        ("weights" + "".join(map(str, k)), v)
        for k, v in jax.tree.flatten_with_path(weights)[0]
    )
    ser.save(flat_data, path)  # save a flatten with path to avoid custom


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
    chunk = jnp.pad(chunk, [(0, 0), (0, pad_to - chunk.shape[-1])])
    segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)
    return chunk, segment_ids


## serialization
# def save_pytree(data, path):
#    flat_data = odict(("weights" + "".join(map(str, k)), v) for k, v in jax.tree.flatten_with_path(data)[0])
#    ser.save(flat_data, path)  # save a flatten with path to avoid custom
#
#
# def load_pytree(path, sharding=None):
#    flat_sharding = odict(("weights" + "".join(map(str, k)), v) for k, v in jax.tree.flatten_with_path(sharding)[0])
#    return jax.tree.unflatten(jax.tree.structure(sharding), jax.tree.leaves(ser.load(path, flat_sharding)))


def prefill(
    tokens: jax.Array,
    weights: Weights,
    cache: KVCache | None,
    cfg: Config,
    pad_id: int = 0,
):
    """Samples from a prompt."""
    # Calculate the next power of 2 for padding, up to cfg.max_seq.
    assert tokens.shape[-1] <= cfg.max_seq_len
    pad_to = 2 ** math.ceil(math.log2(tokens.shape[-1]))

    prompt, prompt_segment_ids = prepare_chunk(tokens, pad_to=pad_to, pad_id=pad_id)
    assert prompt.ndim == 2

    logits_shardings = jax.sharding.NamedSharding(
        cfg.mesh, P(BATCH_AXIS_NAME, None, TENSOR_AXIS_NAME)
    )
    cache_shardings = KVCache.shardings(
        cfg, cache.batch_size if cache is not None else tokens.shape[0]
    )

    if is_type(cache, KVCache):
        cache = dataclasses.replace(
            cache,
            iter=-jnp.ones_like(cache.iter),
            starts=_count_left_padding(tokens, pad_id=pad_id),
        )
        logits, cache = jax.jit(
            forward,
            donate_argnums=(4,),
            out_shardings=(logits_shardings, cache_shardings),
        )(prompt, prompt_segment_ids, weights, cfg, cache)
    elif is_type(cache, PagedKVCache):
        raise ValueError("Prefill with Paged KV Cache is not currently supported.")
    else:
        cache_shardings = KVCache.shardings(dataclasses.replace(cfg), tokens.shape[0])
        kv_sharding = [
            (cache_shardings.k[idx], cache_shardings.v[idx])
            for idx in range(cfg.num_layers)
        ]
        logits, kv_list = jax.jit(
            forward, out_shardings=(logits_shardings, kv_sharding)
        )(prompt, prompt_segment_ids, weights, cfg, None)
        cache = kv_list
    next_tokens = jax.jit(jnp.argmax, static_argnames=("axis",))(logits, axis=-1)
    return next_tokens, logits, cache


prefill.forward = forward


@partial(jax.jit, donate_argnames=("cache",))
def decode_step(last_tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config):
    assert last_tokens.ndim == 2
    segment_ids = jnp.ones(last_tokens.shape, dtype=jnp.int32)
    next_logits, cache = forward(last_tokens, segment_ids, weights, cfg, cache)
    next_tokens = jnp.argmax(next_logits, -1)
    next_tokens = reshard(next_tokens, P())
    return next_tokens, cache
