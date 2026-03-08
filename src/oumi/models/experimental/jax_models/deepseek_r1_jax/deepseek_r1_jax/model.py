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
import gzip
import json
import math
import tempfile
from collections.abc import Callable
from dataclasses import field
from functools import lru_cache, partial
from pathlib import Path

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
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .decode_ragged_dot import decode_ragged_dot

PAD_ID = 1

AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]


# Expected physical mesh axis names:
# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
TENSOR_AXIS_NAME = ("y", "z")
TENSOR_ONLY_AXIS_NAME = "z"
EXPERT_AXIS_NAME = "y"


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
    head_dim: AxisName = None
    vocab_in: AxisName = None
    vocab_out: AxisName = TENSOR_AXIS_NAME
    act_embed: AxisName = None
    act_heads: AxisName = TENSOR_AXIS_NAME
    # attention layer
    qkv_heads: AxisName = TENSOR_AXIS_NAME
    qkv_embed: AxisName = None
    q_lora: AxisName = None
    kv_lora: AxisName = None
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
    moe_s_up_embed: AxisName = None
    moe_s_up_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_s_down_ffw: AxisName = TENSOR_ONLY_AXIS_NAME
    moe_s_down_embed: AxisName = None
    moe_e_tp: AxisName = (
        TENSOR_ONLY_AXIS_NAME  # moe forward function tensor parallelism
    )
    moe_e_ep: AxisName = EXPERT_AXIS_NAME  # moe forward function expert parallelism


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


@tree_util.register_static
@dataclasses.dataclass
class Config:
    embed: int = 7168
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    num_heads: int = 128
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    vocab_size: int = 129280
    num_layers: int = 61
    max_seq_len: int = 8192
    causal: bool = True
    use_prefill_attn_kernel: bool = False
    use_decode_attn_kernel: bool = False
    use_decode_ragged_dot_kernel: bool = True
    dtype: "jnp.dtype" = jnp.bfloat16
    # Sharding rules
    rules: ShardingRules = field(default_factory=lambda: ShardingRules())
    mesh: jax.sharding.Mesh | None = None
    # Deepseek Yarn RoPE
    rope_theta: float = 1e4
    rope_scaling_factor: float = 40.0
    rope_beta_fast: float = 32
    rope_beta_slow: float = 1
    rope_mscale: float = 1
    rope_mscale_all_dim: float = 1
    rope_original_max_position_embeddings: int = 4096
    # quantization
    quant_scale_dtype: "jnp.dtype" = jnp.float16
    quantize_moe: bool = True
    quantize_mlp: bool = False
    quantize_attn: bool = True
    quantize_cache: bool = True
    # attention
    causal: bool = True
    # MLP
    ffw_size: int = 18432
    # MoE
    first_k_dense: int = 3
    moe_gate_dtype: "jnp.dtype" = jnp.float32
    moe_ffw_size: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    n_shared_experts: int = 1
    psum_before_expert_reduce: bool = False
    strategy: str = "decode"


def load_tokenizer(
    tokenizer_path: Path | None = None, tokenizer_config_path: Path | None = None
) -> "PreTrainedTokenizerFast":  # noqa: F821
    from transformers import PreTrainedTokenizerFast

    if tokenizer_path is not None:
        tokenizer_path = epath.Path(tokenizer_path).expanduser().resolve()
    else:
        tokenizer_path = (
            epath.Path(__file__).parent
            / "third_party"
            / "tokenizer"
            / "tokenizer.json.gz"
        )
    if tokenizer_config_path is not None:
        tokenizer_config_path = epath.Path(tokenizer_config_path).expanduser().resolve()
    else:
        tokenizer_config_path = (
            epath.Path(__file__).parent
            / "third_party"
            / "tokenizer"
            / "tokenizer_config.json"
        )
    config = json.loads(tokenizer_config_path.expanduser().resolve().read_text())
    for k in list(config.keys()):
        v = config[k]
        if "token" in k and isinstance(v, dict):
            config[k] = v["content"]
    # return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)
    if tokenizer_path.suffix == ".gz":
        with tempfile.NamedTemporaryFile() as file:
            new_tokenizer_path = Path(file.name)
            new_tokenizer_path.write_bytes(gzip.decompress(tokenizer_path.read_bytes()))
            return PreTrainedTokenizerFast(
                tokenizer_file=str(new_tokenizer_path.resolve()), **config
            )
    else:
        return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)


# module reload friendly check for type(x) == cls
is_type = lambda x, cls: (
    (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
)
is_param = lambda x: is_type(x, ArrayInfo)
which_platform = lambda cfg: cfg.mesh.devices.reshape(-1)[0].platform
_count_left_padding = lambda ids, pad_id=PAD_ID: jnp.sum(
    jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1
)
_length_minus_right_padding = lambda segment_ids: jnp.sum(
    jnp.cumsum(jnp.flip(segment_ids != 0, -1), axis=-1) > 0, -1
)
_he_normal = lru_cache(jax.nn.initializers.he_normal)
_ones_init = jax.nn.initializers.ones


@partial(
    jax_pytree_struct, meta_fields=("shape", "dtype", "logical_axes", "initializer")
)
@dataclasses.dataclass(frozen=True)
class ArrayInfo:
    """Metadata describing a jax.Array, including its sharding.

    We create ArrayInfos before creating actual arrays, e.g. for model weights, so we can use the sharding and other
    metadata to set things up so we can efficiently create the actual arrays with the correct shardings.

    An alternative approach would be to use jax.eval_shape to more automatically generate the metadata we need. We use
    the ArrayInfo approach instead to decouple data and its sharding from the functions we'll apply the data to.

    """

    shape: tuple[int, ...]
    dtype: "jnp.dtype"
    logical_axes: tuple
    initializer: Callable | None = None


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
    """Base class for pytree data structures that will eventually contain jax.Arrays (e.g. layer definitions).

    Each subclass is responsible for defining abstract(), which returns an "abstract" version of the pytree containing
    ArrayInfos (i.e. metadata) instead of actual data. This class then helps generate the shardings and actual data.
    """

    @classmethod
    def abstract(cls, cfg: Config, *args, **kw):
        """Returns an instance of this class with ArrayInfos instead of jax.Arrays."""
        raise NotImplementedError

    @classmethod
    def shardings(cls, cfg: Config, *args, **kw):
        """Returns an instance of this class with Shardings instead of jax.Arrays.

        This is used to generate the Shardings needed for each array.
        """
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

    if isinstance(x, jax.Array):
        if not isinstance(axis, tuple):
            axis = (axis,)
        axis = tuple(z % x.ndim for z in axis)
        amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
        scale = (amax / 127.0 + jnp.finfo(scale_dtype).tiny).astype(scale_dtype)
        quant = jnp.round(x / scale).astype(jnp.int8)
        scale = scale.reshape([z for i, z in enumerate(scale.shape) if i not in axis])
        return quant, scale

    if is_type(x, ArrayInfo):
        if not isinstance(axis, tuple):
            axis = (axis,)
        axis = tuple(z % len(x.shape) for z in axis)
        new_shape = tuple(ax for i, ax in enumerate(x.shape) if i not in axis)
        new_logical_axes = tuple(
            ax for i, ax in enumerate(x.logical_axes) if i not in axis
        )
        if zero_init:
            quant_init, scale_init = jax.nn.initializers.zeros, jax.nn.initializers.ones
        else:
            quant_init, scale_init = _int8_quant_init, _int8_scale_init
        return (
            dataclasses.replace(
                x, shape=x.shape, dtype=jnp.int8, initializer=quant_init
            ),
            ArrayInfo(new_shape, scale_dtype, new_logical_axes, scale_init),
        )

    raise ValueError(f"quantize got unexpected type: {type(x)}")


def quantize_update_slice(
    x: QuantArray, y: jax.Array, pos: int, update_axis: int, quant_axis: int
):
    assert x.quant.ndim == y.ndim
    quant_axis, update_axis = (
        quant_axis % x.quant.ndim,
        update_axis % x.quant.ndim,
    )  # normalize axis numbers
    # y_quant, y_scale = quantize(y, axis=quant_axis, scale_dtype=x.scale.dtype)  # quantize rhs
    y_quant, y_scale = y.quant, y.scale
    scale_update_axis = [ax for ax in range(x.quant.ndim) if ax != quant_axis][
        update_axis
    ]  # update axis in `scale`
    z_quant = jax.lax.dynamic_update_slice_in_dim(
        x.quant, y_quant.astype(x.quant.dtype), pos, axis=update_axis
    )
    z_scale = jax.lax.dynamic_update_slice_in_dim(
        x.scale, y_scale.astype(x.scale.dtype), pos, axis=scale_update_axis
    )
    return z_quant, z_scale


@jax_pytree_struct
class MLPLayer(_Init):
    w_gate: jax.Array | ArrayInfo | QuantArray
    w_up: jax.Array | ArrayInfo | QuantArray
    w_down: jax.Array | ArrayInfo | QuantArray

    @classmethod
    def abstract(cls, cfg: Config):
        _init = _he_normal(in_axis=0, out_axis=1)
        dtype = cfg.dtype
        layer = MLPLayer(
            w_gate=ArrayInfo(
                (cfg.embed, cfg.ffw_size), dtype, ("mlp_up_embed", "mlp_up_ffw"), _init
            ),
            w_up=ArrayInfo(
                (cfg.embed, cfg.ffw_size), dtype, ("mlp_up_embed", "mlp_up_ffw"), _init
            ),
            w_down=ArrayInfo(
                (cfg.ffw_size, cfg.embed),
                dtype,
                ("mlp_down_ffw", "mlp_down_embed"),
                _init,
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "MLPLayer", cfg: Config):
        if not cfg.quantize_mlp:
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
    b_router: jax.Array | ArrayInfo | QuantArray
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
                (cfg.embed, cfg.n_routed_experts),
                cfg.moe_gate_dtype,
                ("moe_e_up_embed", None),
                _sinit,
            ),
            b_router=ArrayInfo(
                (cfg.n_routed_experts,),
                cfg.moe_gate_dtype,
                (None,),
                jax.nn.initializers.zeros,
            ),
            we_gate=ArrayInfo(
                (cfg.n_routed_experts, cfg.embed, cfg.moe_ffw_size),
                dtype,
                ("moe_e_experts", "moe_e_up_embed", "moe_e_up_ffw"),
                _einit,
            ),
            we_up=ArrayInfo(
                (cfg.n_routed_experts, cfg.embed, cfg.moe_ffw_size),
                dtype,
                ("moe_e_experts", "moe_e_up_embed", "moe_e_up_ffw"),
                _einit,
            ),
            we_down=ArrayInfo(
                (cfg.n_routed_experts, cfg.moe_ffw_size, cfg.embed),
                dtype,
                ("moe_e_experts", "moe_e_down_ffw", "moe_e_down_embed"),
                _einit,
            ),
            ws_gate=ArrayInfo(
                (cfg.embed, cfg.n_shared_experts * cfg.moe_ffw_size),
                dtype,
                ("moe_s_up_embed", "moe_s_up_ffw"),
                _sinit,
            ),
            ws_up=ArrayInfo(
                (cfg.embed, cfg.n_shared_experts * cfg.moe_ffw_size),
                dtype,
                ("moe_s_up_embed", "moe_s_up_ffw"),
                _sinit,
            ),
            ws_down=ArrayInfo(
                (cfg.moe_ffw_size, cfg.n_shared_experts * cfg.embed),
                dtype,
                ("moe_s_down_ffw", "moe_s_down_embed"),
                _sinit,
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "MoELayer", cfg: Config):
        if not cfg.quantize_moe:
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
class AttentionLayer(_Init):
    q_a: jax.Array | ArrayInfo | QuantArray
    q_gamma: jax.Array | ArrayInfo | QuantArray
    q_b: jax.Array | ArrayInfo | QuantArray
    kv_a: jax.Array | ArrayInfo | QuantArray
    k_pe: jax.Array | ArrayInfo | QuantArray
    kv_gamma: jax.Array | ArrayInfo | QuantArray
    k_b: jax.Array | ArrayInfo | QuantArray
    v_b: jax.Array | ArrayInfo | QuantArray
    o: jax.Array | ArrayInfo | QuantArray

    @classmethod
    def abstract(cls, cfg: Config):
        dtype = cfg.dtype
        q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        layer = AttentionLayer(
            q_a=ArrayInfo(
                (cfg.embed, cfg.q_lora_rank),
                dtype,
                ("qkv_embed", "q_lora"),
                _he_normal(0, 1),
            ),
            q_gamma=ArrayInfo((cfg.q_lora_rank,), dtype, ("q_lora",), _ones_init),
            q_b=ArrayInfo(
                (cfg.q_lora_rank, cfg.num_heads, q_head_dim),
                dtype,
                ("q_lora", "qkv_heads", "head_dim"),
                _he_normal(0, (1, 2)),
            ),
            kv_a=ArrayInfo(
                (cfg.embed, cfg.kv_lora_rank),
                dtype,
                ("qkv_embed", "kv_lora"),
                _he_normal(0, 1),
            ),
            k_pe=ArrayInfo(
                (cfg.embed, cfg.qk_rope_head_dim),
                dtype,
                ("qkv_embed", "head_dim"),
                _he_normal(0, 1),
            ),
            kv_gamma=ArrayInfo((cfg.kv_lora_rank,), dtype, ("kv_lora",), _ones_init),
            k_b=ArrayInfo(
                (cfg.kv_lora_rank, cfg.num_heads, cfg.qk_nope_head_dim),
                dtype,
                ("kv_lora", "qkv_heads", "head_dim"),
                _he_normal(0, (1, 2)),
            ),
            v_b=ArrayInfo(
                (cfg.kv_lora_rank, cfg.num_heads, cfg.v_head_dim),
                dtype,
                ("kv_lora", "qkv_heads", "head_dim"),
                _he_normal(0, (1, 2)),
            ),
            o=ArrayInfo(
                (cfg.num_heads, cfg.v_head_dim, cfg.embed),
                dtype,
                ("o_heads", "head_dim", "o_embed"),
                _he_normal(0, (1, 2)),
            ),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "AttentionLayer", cfg: Config):
        if not cfg.quantize_attn:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        return dataclasses.replace(
            layer,
            q_a=QuantArray(*quantize(layer.q_a, 1, scale_dtype)),
            q_b=QuantArray(*quantize(layer.q_b, (1, 2), scale_dtype)),
            kv_a=QuantArray(*quantize(layer.kv_a, 1, scale_dtype)),
            k_pe=QuantArray(*quantize(layer.k_pe, 1, scale_dtype)),
            k_b=QuantArray(*quantize(layer.k_b, (1, 2), scale_dtype)),
            v_b=QuantArray(*quantize(layer.v_b, (1, 2), scale_dtype)),
            o=QuantArray(*quantize(layer.o, (0, 1), scale_dtype), out_scaling=True),
        )


@jax_pytree_struct
class Layer(_Init):
    mlp: MLPLayer | MoELayer
    attn: AttentionLayer
    gamma_pre_attn: jax.Array | ArrayInfo
    gamma_post_attn: jax.Array | ArrayInfo

    @classmethod
    def abstract(cls, cfg: Config, use_moe: bool = True) -> "Layer":
        dtype = cfg.dtype
        return Layer(
            mlp=MoELayer.abstract(cfg) if use_moe else MLPLayer.abstract(cfg),
            attn=AttentionLayer.abstract(cfg),
            gamma_pre_attn=ArrayInfo((cfg.embed,), dtype, ("act_embed",), _ones_init),
            gamma_post_attn=ArrayInfo((cfg.embed,), dtype, ("act_embed",), _ones_init),
        )

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
        layers = [
            Layer.abstract(cfg, use_moe=i >= cfg.first_k_dense)
            for i in range(cfg.num_layers)
        ]
        return Weights(
            layers=layers,
            embedding=ArrayInfo(
                (cfg.vocab_size, cfg.embed),
                cfg.dtype,
                (None, "vocab_in"),
                _he_normal(in_axis=0, out_axis=1),
            ),
            gamma_final=ArrayInfo((cfg.embed,), cfg.dtype, ("act_embed",), _ones_init),
            lm_head=ArrayInfo(
                (cfg.embed, cfg.vocab_size),
                cfg.dtype,
                ("vocab_in", "vocab_out"),
                _he_normal(in_axis=1, out_axis=0),
            ),
        )

    @staticmethod
    def quantize(weights: "Weights", cfg: Config):
        return dataclasses.replace(
            weights, layers=[layer.quantize(layer, cfg) for layer in weights.layers]
        )


@partial(jax_pytree_struct, meta_fields=["time_axis", "size"])
class KVCache(_Init):
    k_nope: list[jax.Array]  # [batch_size, max_seq_len, kv_lora]
    k_pe: list[jax.Array]  # [batch_size, max_seq_len, qk_rope_head_dim]
    v: list[jax.Array]  # [batch_size, max_seq_len, kv_lora]
    iter: jax.Array  # []  # sequences are right-aligned for slice udpate performance
    starts: (
        jax.Array
    )  # [batch_size]  # sequences are right-aligned, we need start indices
    time_axis: int = 2
    size: int = -1

    @classmethod
    def abstract(
        cls, cfg: Config, batch_size: int, max_seq_len: int, dtype: int = jnp.bfloat16
    ):
        _init = jax.nn.initializers.zeros
        k_nope_info = ArrayInfo(
            (batch_size, cfg.num_heads, max_seq_len, cfg.qk_nope_head_dim),
            dtype,
            ("batch", "qkv_heads", "sequence", "head_dim"),
            _init,
        )
        k_pe_info = ArrayInfo(
            (batch_size, 1, max_seq_len, cfg.qk_rope_head_dim),
            dtype,
            ("batch", None, "sequence", "head_dim"),
            _init,
        )
        v_info = ArrayInfo(
            (batch_size, cfg.num_heads, max_seq_len, cfg.v_head_dim),
            dtype,
            ("batch", "qkv_heads", "sequence", "head_dim"),
            _init,
        )
        cache = KVCache(
            k_nope=[k_nope_info for _ in range(cfg.num_layers)],
            k_pe=[k_pe_info for _ in range(cfg.num_layers)],
            v=[v_info for _ in range(cfg.num_layers)],
            iter=ArrayInfo((), jnp.int32, (), _init),
            starts=ArrayInfo((batch_size,), jnp.int32, ("batch",), _init),
            size=max_seq_len,
        )
        if cfg.quantize_cache:
            _quantize = partial(
                quantize, axis=-1, scale_dtype=cfg.quant_scale_dtype, zero_init=True
            )
            cache.k_nope = [
                QuantArray(*_quantize(k_nope), out_scaling=True, scale_expand_dims=-2)
                for k_nope in cache.k_nope
            ]
            cache.k_pe = [
                QuantArray(*_quantize(k_pe), out_scaling=True, scale_expand_dims=-2)
                for k_pe in cache.k_pe
            ]
            cache.v = [
                QuantArray(*_quantize(v), out_scaling=False, scale_expand_dims=-2)
                for v in cache.v
            ]
        return cache

    def fill_len(self) -> jax.Array:
        return jnp.where(self.iter >= 0, (self.iter - self.starts) % self.size, 0)

    @property
    def buffers(self):
        return (self.k_nope, self.k_pe, self.v)


def einsum(subscripts: str, lhs: jax.Array, rhs: jax.Array | QuantArray):
    """jnp.einsum wrapper that handles regular arrays and QuantArrays"""
    if is_type(rhs, QuantArray):
        scale = jnp.expand_dims(rhs.scale, rhs.scale_expand_dims)
        if rhs.out_scaling:
            return jnp.einsum(subscripts, lhs, rhs.quant) * scale
        else:
            return jnp.einsum(subscripts, lhs * scale, rhs.quant)
    else:
        return jnp.einsum(subscripts, lhs, rhs)


def update_slice(
    x: jax.Array | QuantArray,
    y: jax.Array,
    pos: int,
    update_axis: int,
    quant_axis: int = -1,
):
    """dynamic_update_slice wrapper that handles regular arrays and QuantArrays"""
    if is_type(x, QuantArray):
        new_quant, new_scale = quantize_update_slice(
            x, y, pos, update_axis=update_axis, quant_axis=quant_axis
        )
        return dataclasses.replace(x, quant=new_quant, scale=new_scale)
    else:
        return jax.lax.dynamic_update_slice_in_dim(
            x, y.astype(x.dtype), pos, axis=update_axis
        )


def logical_sharding_constraint(
    x: jax.Array | QuantArray,
    logical_axes: Axes,
    mesh: jax.sharding.Mesh,
    rules: ShardingRules,
):
    """Generate a sharding constraint for a regular or QuantArray given its logical axes."""
    sharding = logical_to_sharding(logical_axes, mesh, rules)
    if is_type(x, QuantArray):
        return dataclasses.replace(
            x, quant=jax.lax.with_sharding_constraint(x.quant, sharding)
        )
    else:
        return jax.lax.with_sharding_constraint(x, sharding)


def segment_ids_to_positions(segment_ids):
    """Counts positions for segment ids."""

    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

    vals = (jnp.zeros_like(segment_ids), segment_ids)
    return jnp.array(
        jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32"
    )


def _yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim) - min) / (max - min)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func


def generate_pos_embeddings(positions, head_dim, cfg: Config):
    fractions = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    freq_extra = 1.0 / (cfg.rope_theta**fractions)
    freq_inter = 1.0 / (cfg.rope_scaling_factor * cfg.rope_theta**fractions)

    low, high = _yarn_find_correction_range(
        cfg.rope_beta_fast,
        cfg.rope_beta_slow,
        head_dim,
        cfg.rope_theta,
        cfg.rope_original_max_position_embeddings,
    )
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, head_dim // 2).astype(
        jnp.float32
    )
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    freqs = jnp.einsum(
        "...T,k->...Tk", positions, inv_freq, precision=jax.lax.Precision.HIGHEST
    )
    _yarn_get_mscale = lambda scale, mscale: jnp.where(
        scale <= 1, 1.0, 0.1 * mscale * jnp.log(scale) + 1.0
    )
    _mscale = _yarn_get_mscale(
        cfg.rope_scaling_factor, cfg.rope_mscale
    ) / _yarn_get_mscale(cfg.rope_scaling_factor, cfg.rope_mscale_all_dim)
    sin, cos = jnp.sin(freqs) * _mscale, jnp.cos(freqs) * _mscale
    return sin, cos


def apply_rotary_embedding(x, sin, cos):
    assert x.ndim == 4
    assert sin.ndim == 3 and cos.ndim == 3
    # [B, T, head_dim] -> [B, h, T, head_dim]
    sin, cos = (sin[:, None, :, :], cos[:, None, :, :])
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_attention_mask(
    q_len, k_len, q_segment_ids, kv_segment_ids, q_offset, kv_offset, causal: bool
):
    segment_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]  # [B, t, T]
    segment_mask = segment_mask[:, None, :, :]  # [B, t, T] -> [B, 1, t, T]

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


def _get_attn_scale(q_head_dim: int, cfg: Config):
    scale = q_head_dim**-0.5
    if cfg.rope_scaling_factor <= 1.0:
        _yarn_mscale = 1.0
    else:
        _yarn_mscale = (
            0.1 * cfg.rope_mscale_all_dim * math.log(cfg.rope_scaling_factor) + 1.0
        )
    return scale * _yarn_mscale**2


def attention(
    q_nope: jax.Array,
    q_pe: jax.Array,
    k_nope: jax.Array | tuple[jax.Array, jax.Array],
    k_pe: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_offset: jax.Array,
    kv_offset: jax.Array,
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
    scale = _get_attn_scale(q_nope.shape[-1] + q_pe.shape[-1], cfg)

    # grouped-query attention
    b, h, t, d = q_nope.shape
    _, h, T, _ = k_nope.shape

    qk = einsum("bhtd,bhTd->bhtT", q_nope, k_nope)
    # qk = qk + einsum("bhtd,bTd->bhtT", q_pe, k_pe)
    qk = qk + einsum("bhtd,b1Td->bhtT", q_pe, k_pe)
    qk = qk * scale  # [b, h, t, T]

    mask = make_attention_mask(
        t, T, q_segment_ids, kv_segment_ids, q_offset, kv_offset, cfg.causal
    )
    qk = jnp.where(mask, qk, -1e30)  # Apply the combined mask
    attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)

    # grouped-query attention
    attn_ = attn.reshape((b, h, t, T))
    qkv = einsum("bhtT,bhTd->bhtd", attn_, v).astype(cfg.dtype)
    return qkv.reshape((b, h, t, v.shape[-1]))


def attention_kernel(
    q_nope: jax.Array,
    q_pe: jax.Array,
    k_nope: jax.Array | tuple[jax.Array, jax.Array],
    k_pe: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_offset: jax.Array,
    kv_offset: jax.Array,
    cfg: Config,
) -> jax.Array:
    """Flash attention kernel!"""
    k_nope, k_nope_scale = (
        (k_nope.quant, k_nope.scale) if is_type(k_nope, QuantArray) else (k_nope, None)
    )
    k_pe, k_pe_scale = (
        (k_pe.quant, k_pe.scale) if is_type(k_pe, QuantArray) else (k_pe, None)
    )
    v, v_scale = (v.quant, v.scale) if is_type(v, QuantArray) else (v, None)
    scale = _get_attn_scale(q_nope.shape[-1] + q_pe.shape[-1], cfg)

    l2p = lambda *logical: logical_to_physical(logical, cfg.rules)
    q_spec = l2p("batch", "qkv_heads", "sequence", "head_dim")

    in_specs = (
        q_spec,  # q_nope
        q_spec,  # q_pe
        l2p("batch", "qkv_heads", "sequence", "head_dim"),  # k_nope
        l2p("batch", None, "sequence", "head_dim"),  # k_pe
        l2p("batch", "qkv_heads", "sequence", "head_dim"),  # v
        l2p("batch", "sequence"),  # q_segment_ids
        l2p("batch", "sequence"),  # kv_segment_ids
        None
        if k_nope_scale is None
        else l2p("batch", "qkv_heads", "sequence"),  # k_nope_scale
        None if k_pe_scale is None else l2p("batch", None, "sequence"),  # k_pe_scale
        None if v_scale is None else l2p("batch", "qkv_heads", "sequence"),  # v_scale
    )
    out_specs = q_spec

    @partial(
        jax.shard_map,
        mesh=cfg.mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    def _f(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        v,
        q_segment_ids,
        kv_segment_ids,
        k_nope_scale,
        k_pe_scale,
        v_scale,
    ):
        q_seq, kv_seq, heads = q_nope.shape[-2], v.shape[-2], v.shape[-3]
        block_q, block_kv = min(q_seq, 512), min(kv_seq, 1024)
        block_sizes = splash.BlockSizes(
            block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv
        )

        mask = mask_lib.MultiHeadMask(
            [mask_lib.CausalMask((q_seq, kv_seq)) for _ in range(heads)]
        )
        attn_static_fn = splash.make_splash_mha_single_device(
            mask=mask, block_sizes=block_sizes
        )
        attn_static_fn = jax.vmap(
            attn_static_fn, in_axes=(0, 0, 0, 0)
        )  # for prefill with an empty cache

        def attn_dynamic_fn(
            q, k, v, segment_ids
        ):  # when the offsets are different (chunked prefill)
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
                splash.make_splash_mha_single_device(
                    mask=mask, block_sizes=block_sizes
                )(q, k, v, segment_ids)
            )
            return jax.vmap(attn_fn, in_axes=(0, 0, 0, 0, 0))(
                q, k, v, segment_ids, mask
            )

        segment_ids = splash.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        if k_nope_scale is not None:
            k_nope = (k_nope * k_nope_scale[..., None]).astype(jnp.bfloat16)
        if k_pe_scale is not None:
            k_pe = (k_pe * k_pe_scale[..., None]).astype(jnp.bfloat16)
        if v_scale is not None:
            v = (v * v_scale[..., None]).astype(jnp.bfloat16)
        k = jnp.concatenate(
            [k_nope, jnp.broadcast_to(k_pe, k_nope.shape[:-1] + k_pe.shape[-1:])], -1
        )
        q = jnp.concatenate([q_nope, q_pe], -1)
        return jax.lax.cond(
            jnp.all(q_offset == kv_offset),
            attn_static_fn,
            attn_dynamic_fn,
            q * scale,
            k,
            v,
            segment_ids,
        )

    return _f(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        v,
        q_segment_ids,
        kv_segment_ids,
        k_nope_scale,
        k_pe_scale,
        v_scale,
    ).astype(jnp.bfloat16)


def rms_norm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    """Apply RMS normalization."""
    rms = jnp.sqrt(
        jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6
    )
    return jnp.astype(gamma * x / rms, jnp.bfloat16)


def mla_attention_block(
    x: jax.Array,
    segment_ids: jax.Array,
    attn_layer: AttentionLayer,
    sin: jax.Array,
    cos: jax.Array,
    cfg: Config,
    cache: KVCache | None = None,
    idx: int = 0,
) -> jax.Array:
    dtype = cfg.dtype
    with jax.named_scope("q_embed"):
        q_lora = einsum("btd,dr->btr", x, attn_layer.q_a).astype(dtype)
        q_lora = rms_norm(q_lora, attn_layer.q_gamma).astype(dtype)
        q = einsum("btr,rhq->bhtq", q_lora, attn_layer.q_b).astype(dtype)
        q_nope = q[..., : cfg.qk_nope_head_dim]
        q_pe = apply_rotary_embedding(q[..., cfg.qk_nope_head_dim :], sin, cos).astype(
            dtype
        )

    with jax.named_scope("kv_compressed_embed"):
        kv_compressed = einsum("btd,dr->btr", x, attn_layer.kv_a).astype(dtype)
        kv_compressed = rms_norm(kv_compressed, attn_layer.kv_gamma).astype(dtype)
        # k_pe = einsum("btd,dq->btq", x, attn_layer.k_pe)
        # k_pe = apply_rotary_embedding(k_pe[..., None, :, :], sin, cos)[..., 0, :, :].astype(dtype)
        k_pe = einsum("btd,dq->btq", x, attn_layer.k_pe)[..., None, :, :]
        k_pe = apply_rotary_embedding(k_pe, sin, cos).astype(dtype)

    with jax.named_scope("kv_embed"):
        k_nope = einsum("btr,rhq->bhtq", kv_compressed, attn_layer.k_b)
        v = einsum("btr,rhv->bhtv", kv_compressed, attn_layer.v_b)

    if cfg.quantize_cache:
        _quantize = partial(quantize, axis=-1, scale_dtype=cfg.quant_scale_dtype)
        k_nope = QuantArray(*_quantize(k_nope), out_scaling=True, scale_expand_dims=-2)
        k_pe = QuantArray(*_quantize(k_pe), out_scaling=True, scale_expand_dims=-2)
        v = QuantArray(*_quantize(v), out_scaling=False, scale_expand_dims=-2)

    with jax.named_scope("full_cache_update"):
        if is_type(cache, KVCache):
            it = jnp.maximum(cache.iter, 0)
            k_nope = update_slice(
                cache.k_nope[idx], k_nope, it, update_axis=cache.time_axis
            )
            # k_pe = update_slice(cache.k_pe[idx], k_pe, it, update_axis=cache.time_axis - 1)
            k_pe = update_slice(cache.k_pe[idx], k_pe, it, update_axis=cache.time_axis)
            v = update_slice(cache.v[idx], v, it, update_axis=cache.time_axis)
            cache_updates = (k_nope, k_pe, v)

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
            cache_updates = (k_nope, k_pe, v)

    # constrain the sharding of intermediates for the attention operation
    lsc = partial(logical_sharding_constraint, mesh=cfg.mesh, rules=cfg.rules)
    spec = ("batch", "act_heads", "sequence", "head_dim")
    q_nope, q_pe = lsc(q_nope, spec), lsc(q_pe, spec)
    # k_nope, k_pe, v = lsc(k_nope, spec), lsc(k_pe, ("batch", "sequence", "head_dim")), lsc(v, spec)
    k_nope, k_pe, v = (
        lsc(k_nope, spec),
        lsc(k_pe, ("batch", None, "sequence", "head_dim")),
        lsc(v, spec),
    )

    # Compute attention
    with jax.named_scope("attention"):
        if (
            which_platform(cfg) == "tpu"
            and cfg.use_prefill_attn_kernel
            and q.shape[-2] != 1
        ):
            attn_out = attention_kernel(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                v,
                q_segment_ids,
                kv_segment_ids,
                q_offset,
                kv_offset,
                cfg=cfg,
            )
        else:
            attn_out = attention(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                v,
                q_segment_ids,
                kv_segment_ids,
                q_offset,
                kv_offset,
                cfg,
            )

    with jax.named_scope("o_proj"):
        attn_out = einsum("bhtv,hvd->btd", attn_out, attn_layer.o)
    attn_out = lsc(attn_out.astype(cfg.dtype), ("batch", "sequence", "act_embed"))
    return attn_out, cache_updates


@partial(jax.jit, static_argnames=("replicated_routing",))
def _route_tokens_to_moe_experts(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    replicated_routing: bool,
    cfg: Config,
):
    lsc = partial(logical_sharding_constraint, mesh=cfg.mesh, rules=cfg.rules)
    x_shape = x.shape
    x = x.reshape((-1, x.shape[-1]))
    if (
        replicated_routing
    ):  # not distributing the routing work avoids communication for small batches
        x = lsc(x, (None, None))
    else:
        x = jax.lax.with_sharding_constraint(
            x, NamedSharding(cfg.mesh, P(TENSOR_AXIS_NAME, None))
        )
    weight, bias = lsc(weight, (None, None)), lsc(bias, (None,))

    scores = jax.nn.sigmoid(
        jnp.einsum("Sk,kj->Sj", x, weight).astype(cfg.moe_gate_dtype)
    )
    scores_with_bias = scores + bias
    group_scores = jnp.sum(
        jax.lax.top_k(
            scores_with_bias.reshape(scores.shape[:-1] + (cfg.n_group, -1)), 2
        )[0],
        axis=-1,
    )
    group_idx = jax.lax.top_k(group_scores, cfg.topk_group)[1]
    mask = jnp.any(jnp.arange(cfg.n_group)[:, None] == group_idx[..., None, :], axis=-1)
    mask = jnp.repeat(mask, scores.shape[-1] // mask.shape[-1], -1)
    masked_scores = jnp.where(mask, scores_with_bias, 0.0)
    topk_idx = jax.lax.top_k(masked_scores, cfg.num_experts_per_tok)[1]
    topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1).astype(
        cfg.moe_gate_dtype
    )
    topk_weights = (
        cfg.routed_scaling_factor
        * topk_weights
        / (jnp.sum(topk_weights, axis=-1)[..., None] + 1e-20)
    )

    topk_weights = lsc(topk_weights, (None, None)).reshape(
        x_shape[:-1] + (cfg.num_experts_per_tok,)
    )
    topk_idx = lsc(topk_idx, (None, None)).reshape(
        x_shape[:-1] + (cfg.num_experts_per_tok,)
    )
    return topk_weights, topk_idx


def _moe_gmm(lhs, rhs, group_sizes, topk_idx, cfg: Config):
    assert lhs.ndim == 2 and rhs.ndim == 3, f"{lhs.ndim=} != 2 and {rhs.ndim=} != 3"
    group_sizes = group_sizes.astype(jnp.int32)
    if cfg.use_decode_ragged_dot_kernel and lhs.shape[0] <= 1024:
        with jax.named_scope("decode_ragged_dot"):
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
                    block_g=4,
                    block_n=1024,
                    interpret=False,
                )
                ret = ret * scale
            else:
                ret = decode_ragged_dot(
                    lhs, rhs, group_sizes, block_g=4, block_n=1024, interpret=False
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
    _psc = lambda z, spec: jax.lax.with_sharding_constraint(
        z, NamedSharding(cfg.mesh, P(*spec))
    )
    _qpsc = lambda z, spec: dataclasses.replace(
        z, quant=_psc(z.quant, spec.quant), scale=_psc(z.scale, spec.scale)
    )
    psc = lambda z, spec: _qpsc(z, spec) if is_type(z, QuantArray) else _psc(z, spec)

    replicated_routing = x.shape[-2] == 1  # we're decoding
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
    if cfg.strategy == "prefill":
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
    assert cfg.n_routed_experts % expert_count == 0
    expert_size = cfg.n_routed_experts // expert_count

    @partial(
        jax.shard_map,
        mesh=cfg.mesh,
        in_specs=in_specs,
        out_specs=out_spec,
        check_vma=False,
    )
    def _expert_fn(x, we_gate, we_up, we_down, topk_weights, topk_idx):
        (b, s, d), e = x.shape, cfg.num_experts_per_tok
        expert_idx = (
            jax.lax.axis_index(expert_axname) if expert_axname is not None else 0
        )
        tensor_idx = (
            jax.lax.axis_index(tensor_axname) if tensor_axname is not None else 0
        )
        topk_idx_ = topk_idx.reshape(-1)
        valid_group_mask_ = (topk_idx_ >= expert_size * expert_idx) & (
            topk_idx_ < expert_size * (expert_idx + 1)
        )
        expert_mapped_topk_idx_ = jnp.where(
            valid_group_mask_, topk_idx_ - expert_idx * expert_size, 2**30
        )

        sort_idx_ = jnp.argsort(expert_mapped_topk_idx_, axis=-1)  # [b * s * e]
        isort_idx_ = jnp.argsort(sort_idx_)

        if cfg.strategy == "prefill":
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

        group_sizes = jnp.bincount(topk_idx_sort_, length=cfg.n_routed_experts)
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

        if cfg.strategy == "prefill":
            rs_shape = (
                math.ceil((ff_out.shape[-1] // tensor_count) / 256) * 256 * tensor_count
            )
            pad_size = rs_shape - ff_out.shape[-1]
            ff_out = jnp.pad(ff_out, ((0, 0), (0, pad_size)))
            ff_out = jax.lax.psum_scatter(
                ff_out, axis_name=tensor_axname, scatter_dimension=1, tiled=True
            )

        if cfg.strategy == "prefill":
            with jax.named_scope("expert_weighting"):
                ff_out = ff_out * topk_weights.reshape(-1)[sort_idx_][..., None]
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
            with jax.named_scope("expert_weighting"):
                ff_out_expert = jnp.einsum(
                    "Ted,Te->Td",
                    ff_out.reshape((b * s, e, -1)),
                    topk_weights.reshape((b * s, -1)),
                )
                ff_out_expert = ff_out_expert.astype(cfg.dtype)

        with jax.named_scope("experts_collective"):
            if cfg.strategy == "prefill":
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


def mlp_block(x: jax.Array, layer: MLPLayer, cfg: Config):
    lsc = partial(logical_sharding_constraint, mesh=cfg.mesh, rules=cfg.rules)
    dtype = cfg.dtype
    with jax.named_scope("gate"):
        ff_gate = jax.nn.silu(einsum("btd,df->btf", x, layer.w_gate)).astype(dtype)
    with jax.named_scope("up_proj"):
        ff_up = einsum("btd,df->btf", x, layer.w_up).astype(dtype)
    with jax.named_scope("down_proj"):
        ff_out = einsum("btf,fd->btd", ff_gate * ff_up, layer.w_down).astype(dtype)
    return lsc(ff_out, ("batch", "sequence", "act_embed"))


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
    x = jax.lax.with_sharding_constraint(
        x, logical_to_sharding(("batch", "sequence", "act_embed"), cfg.mesh, cfg.rules)
    )

    # Attention block
    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.gamma_pre_attn)
    attn_out, cache_updates = mla_attention_block(
        attn_in, segment_ids, layer.attn, sin, cos, cfg, cache, idx
    )
    with jax.named_scope("residual"):
        x = x + attn_out.astype(cfg.dtype)

    # FFN block
    with jax.named_scope("attn_post_norm"):
        ff_in = rms_norm(x, layer.gamma_post_attn)
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
    with jax.named_scope("vocab_in_proj"):
        # Embed input tokens [B, T] -> [B, T D]
        x = jax.lax.with_sharding_constraint(
            weights.embedding[x, :],
            logical_to_sharding(
                ("batch", "sequence", "act_embed"), cfg.mesh, cfg.rules
            ),
        )
    positions = segment_ids_to_positions(segment_ids)
    if is_type(cache, KVCache):
        positions = positions + cache.fill_len()[:, None]
    sin, cos = generate_pos_embeddings(
        positions, cfg.qk_rope_head_dim, cfg
    )  # [B, T, head_dim]
    sin, cos = sin.astype(cfg.dtype), cos.astype(cfg.dtype)

    all_cache_updates = []
    for idx, layer in enumerate(weights.layers):
        x, cache_updates = forward_layer(
            x, segment_ids, layer, sin, cos, idx, cfg, cache
        )
        all_cache_updates.append(cache_updates)

    x = rms_norm(x, weights.gamma_final)  # Final layer norm.

    with jax.named_scope("vocab_out_proj"):
        x = jax.lax.with_sharding_constraint(
            x, logical_to_sharding(("batch", "sequence", None), cfg.mesh, cfg.rules)
        )
        logits = jnp.einsum(
            "btd,dv->btv", x, weights.lm_head
        )  # Project to vocabulary size

    if is_type(cache, KVCache):
        cache.k_nope, cache.k_pe, cache.v = (
            [z[i] for z in all_cache_updates] for i in range(3)
        )
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
            ocp.args.PyTreeSave(data, ocdbt_target_data_file_size=1024 * 1024 * 100),
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
):
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
        cache_shardings = [
            tuple(z[idx] for z in cache_shardings.buffers)
            for idx in range(cfg.num_layers)
        ]
    logits_shardings = logical_to_sharding(
        ("batch", "sequence", "act_embed"), cfg.mesh, cfg.rules
    )
    logits, cache = jax.jit(
        forward, donate_argnums=(4,), out_shardings=(logits_shardings, cache_shardings)
    )(prompt, prompt_segment_ids, weights, cfg, cache)
    max_tokens = jax.jit(
        partial(jnp.argmax, axis=-1), out_shardings=NamedSharding(cfg.mesh, P())
    )(logits)
    return max_tokens, logits, cache


@partial(jax.jit, donate_argnames=("cache",))
def decode_step(last_tokens: jax.Array, weights, cache: KVCache, cfg: Config):
    assert last_tokens.ndim == 2
    segment_ids = jnp.ones(last_tokens.shape, dtype=jnp.int32)
    next_logits, cache = forward(last_tokens, segment_ids, weights, cfg, cache)
    next_tokens = jnp.argmax(next_logits, -1)
    next_tokens = jax.lax.with_sharding_constraint(
        next_tokens, NamedSharding(cfg.mesh, P())
    )
    return next_tokens, cache
