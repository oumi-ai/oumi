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

import math
import time
from functools import partial

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

NUM_LANES = 128


@partial(jax.named_call, name="ragged_decode_kernel")
def ragged_decode_kernel_fwd(
    # prefetch scalars:
    start_ref,  # [bs]
    length_ref,  # [bs]
    chunked_start_ref,  # [bs // block_bs]
    chunked_length_ref,  # [bs // block_bs]
    # inputs:
    q_ref,  # [bs // block_bs, heads, head_dim]
    k_ref,  # [bs // block_bs, block_kv, head_dim]
    v_ref,  # [bs // block_bs, block_kv, head_dim]
    k_scale_ref,  # optional [bs // block_bs, heads] not None if k is quantized
    v_scale_ref,  # optional [bs // block_bs, heads] not None if v is quantized
    qk_prev_ref,  # optional [bs // block_vs, heads, block_kv] not None if some qk is precomputed (Deepseek on TPU)
    # outputs:
    o_ref,  # [bs // block_bs, heads, head_dim]
    # scratch memory:
    o_scratch_ref,  # [bs // block_bs, heads, head_dim]
    l_scratch_ref,  # [bs // block_bs, heads, TPU_MIN_SIZE]
    m_scratch_ref,  # [bs // block_bs, heads, TPU_MIN_SIZE]
    # parameters:
    kv_seq_len: int,
    block_kv: int,
    block_bs: int,
    scale: float,
    scale_qk_not_k: bool = True,
    scale_s_not_v: bool = True,
):
    del chunked_start_ref, chunked_length_ref
    mask_value = jnp.finfo(o_scratch_ref.dtype).min
    b_, i = pl.program_id(0), pl.program_id(1)

    @pl.when(i == 0)
    def init():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    def resize(x, new_size_in_dim, axis=-1):
        """Resize the shape of array x to the target size along axis `axis`."""
        if x.shape[axis] > new_size_in_dim:
            assert axis in (-1, x.ndim - 1)
            return x[..., :new_size_in_dim]
        return pltpu.repeat(x, new_size_in_dim // x.shape[axis], axis=axis % x.ndim)

    def loop_fn(b, _):
        b_global = block_bs * b_ + b
        start, length = start_ref[b_global], length_ref[b_global]
        block_start, block_end = i * block_kv, (i + 1) * block_kv
        should_compute = (start < length) & (
            (block_start < length) & (block_end >= start)
        )

        @pl.when(should_compute)
        def compute():
            # compute qk
            q, k = q_ref[b, ...], k_ref[b, ...]
            if k_scale_ref is not None and not scale_qk_not_k:
                k = k * k_scale_ref[b, ...].astype(jnp.float32).reshape(
                    k.shape[:-1] + (1,)
                ).astype(jnp.bfloat16)
            qk = jax.lax.dot_general(
                q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            if k_scale_ref is not None and scale_qk_not_k:
                qk = qk * k_scale_ref[b, ...]
            if qk_prev_ref is not None:
                qk += qk_prev_ref[b, ...]
            qk *= scale
            indices = i * block_kv + jax.lax.broadcasted_iota(
                jnp.int32, qk.shape, dimension=1
            )
            mask = (indices >= start) & (indices < length)
            qk += jnp.where(mask, 0, mask_value)

            # adjust maximum shift value, shift and softmax
            m_prev, l_prev = m_scratch_ref[b, ...], l_scratch_ref[b, ...]
            m_curr = resize(jnp.max(qk, axis=-1)[:, None], m_prev.shape[-1])
            m_next = jnp.maximum(m_prev, m_curr)
            s_curr = jnp.exp(qk - resize(m_next, qk.shape[-1]))
            l_curr = jax.lax.broadcast_in_dim(
                jnp.sum(s_curr, axis=-1), l_prev.shape, (0,)
            )

            # compute the (qk v)
            v = v_ref[b, ...]
            if v_scale_ref is not None and not scale_s_not_v:
                v = v * v_scale_ref[b, ...].astype(jnp.float32).reshape(
                    v.shape[:-1] + (1,)
                ).astype(jnp.bfloat16)
            elif v_scale_ref is not None and scale_s_not_v:
                s_curr = s_curr * v_scale_ref[b, ...]
            o_curr = jax.lax.dot_general(
                s_curr, v, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )

            # accumulate the results
            o_prev = o_scratch_ref[b, ...]
            m_next = jnp.maximum(m_prev, m_curr)
            alpha = jnp.exp(m_prev - m_next)
            l_next = l_prev * alpha + l_curr
            l_next_safe = l_next
            o_next = resize(alpha, o_prev.shape[-1]) * o_prev + o_curr

            # store scratch values
            m_scratch_ref[b, ...] = m_next
            l_scratch_ref[b, ...] = l_next_safe
            o_scratch_ref[b, ...] = o_next

    jax.lax.fori_loop(0, block_bs, loop_fn, init_val=None)

    @pl.when(i == (kv_seq_len // block_kv) - 1)
    def done():
        l = l_scratch_ref[...]
        l_inv = jnp.where(l == 0.0, 1.0, 1.0 / l)
        o_ref[...] = (
            o_scratch_ref[...] * resize(l_inv, o_scratch_ref.shape[-1])
        ).astype(o_ref.dtype)


def ragged_decode_fwd(
    q: jax.Array,  # [bs, q_heads, head_dim]
    k: jax.Array,  # [bs, kv_seq_len, head_dim]
    v: jax.Array,  # [bs, kv_seq_len, head_dim]
    starts: jax.Array | None = None,  # [bs]
    lengths: jax.Array | None = None,  # [bs]
    k_scale: jax.Array | None = None,  # [bs, kv_seq_len]
    v_scale: jax.Array | None = None,  # [bs, kv_seq_len]
    qk_prev: jax.Array | None = None,  # [bs, q_heads, kv_seq_len]
    block_bs: int = 4,
    block_kv: int = 256,
    scale: float | None = None,
    scale_qk_not_k: bool = True,
    scale_s_not_v: bool = True,
    interpret: bool = False,
):
    """Pallas kernel for ragged batched attention decoding.

    Args:
        q: Query tensor of shape [bs, q_heads, head_dim].
        k: Key tensor of shape [bs, kv_seq_len, head_dim].
        v: Value tensor of shape [bs, kv_seq_len, head_dim].
        starts: Optional start indices for each batch in the key/value sequences.  Shape [bs].
        lengths: Optional lengths of the key/value sequences for each batch.  Shape [bs].
        k_scale: Optional scaling factors for the key tensor.  Shape [bs, kv_seq_len].
        v_scale: Optional scaling factors for the value tensor.  Shape [bs, kv_seq_len].
        qk_prev: Optional previous query-key attention scores.  Shape [bs, q_heads, kv_seq_len].
        block_bs: Block size for batch dimension.
        block_kv: Block size for key/value sequence length dimension.
        scale: Optional scaling factor for attention scores. If None, defaults to sqrt(head_dim).
        scale_qk_not_k: Whether to scale the query-key product or to scale the rhs key. Defaults to True.
        scale_s_not_v: Whether to scale the attention scores or to scale the rhs value. Defaults to True.
        interpret: Whether to run the kernel in interpret mode for debugging.

    Returns:
        Attention output tensor of shape [bs, q_heads, head_dim].
    """
    scale = math.sqrt(q.shape[-1]) if scale is None else scale
    bs_q, q_heads, head_dim_q = q.shape
    bs_k, kv_seq_len_k, head_dim_k = k.shape
    assert bs_q == bs_k and head_dim_q == head_dim_k
    bs, kv_seq_len = bs_q, kv_seq_len_k
    bs_v, kv_seq_len_v, head_dim_v = v.shape
    assert bs == bs_v and kv_seq_len == kv_seq_len_v

    block_bs = min(bs, block_bs)
    assert bs % block_bs == 0

    if starts is None:
        starts = jnp.zeros((bs,), dtype=jnp.int32)
    if lengths is None:
        lengths = kv_seq_len * jnp.ones((bs,), dtype=jnp.int32)

    assert starts.ndim == 1 and starts.size == bs
    assert lengths.ndim == 1 and lengths.size == bs
    block_kv = min(kv_seq_len, block_kv)
    assert kv_seq_len % block_kv == 0

    chunked_starts = jnp.min(starts.reshape((-1, block_bs)), axis=-1)
    chunked_lengths = jnp.max(lengths.reshape((-1, block_bs)), axis=-1)

    def kv_prefetch_map(
        b, i, starts_ref, lengths_ref, chunked_starts_ref, chunked_lengths_ref
    ):
        # return b, i, 0
        del starts_ref, lengths_ref
        start, length = chunked_starts_ref[b], chunked_lengths_ref[b]
        s_idx = i * block_kv
        last_batch, seq_done = b == (bs // block_bs) - 1, s_idx > length
        start_next = chunked_starts_ref[b + (~last_batch)]
        first_start_i, next_start_i = start // block_kv, start_next // block_kv
        b = jnp.where(seq_done & (~last_batch), b + 1, b)
        i = jnp.where(
            seq_done,
            jnp.where(last_batch, i, next_start_i),
            jnp.maximum(first_start_i, i),
        )
        i = jnp.where(last_batch & seq_done, pl.cdiv(length, block_kv) - 1, i)
        return b, i, 0

    def kv_scale_prefetch_map(
        b, i, starts_ref, lengths_ref, chunked_starts_ref, chunked_lengths_ref
    ):
        b_, i_, _ = kv_prefetch_map(
            b, i, starts_ref, lengths_ref, chunked_starts_ref, chunked_lengths_ref
        )
        return b_, 0, i_

    in_specs = []
    in_specs += [
        pl.BlockSpec((block_bs, q_heads, q.shape[-1]), lambda b, i, *_: (b, 0, 0))
    ]  # q
    in_specs += [pl.BlockSpec((block_bs, block_kv, k.shape[-1]), kv_prefetch_map)]  # k
    in_specs += [pl.BlockSpec((block_bs, block_kv, head_dim_v), kv_prefetch_map)]  # v
    if k_scale is not None:
        in_specs += [pl.BlockSpec((block_bs, 1, block_kv), kv_scale_prefetch_map)]
        k_scale = k_scale.reshape(k_scale.shape[:-1] + (1, k_scale.shape[-1])).astype(
            jnp.bfloat16
        )
    else:
        in_specs += [None]

    if v_scale is not None:
        in_specs += [pl.BlockSpec((block_bs, 1, block_kv), kv_scale_prefetch_map)]
        v_scale = v_scale.reshape(v_scale.shape[:-1] + (1, v_scale.shape[-1])).astype(
            jnp.bfloat16
        )
    else:
        in_specs += [None]

    if qk_prev is not None:
        qk_prev_prefetch_map = kv_scale_prefetch_map
        in_specs += [pl.BlockSpec((block_bs, q_heads, block_kv), qk_prev_prefetch_map)]
    else:
        in_specs += [None]

    out_shape = jax.ShapeDtypeStruct((bs, q_heads, head_dim_v), q.dtype)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(bs // block_bs, kv_seq_len // block_kv),
        in_specs=in_specs,
        out_specs=pl.BlockSpec(
            (block_bs, q_heads, head_dim_v), lambda b, i, *_: (b, 0, 0)
        ),
        scratch_shapes=[
            pltpu.VMEM((block_bs, q_heads, head_dim_v), dtype=jnp.float32),
            pltpu.VMEM((block_bs, q_heads, NUM_LANES), dtype=jnp.float32),
            pltpu.VMEM((block_bs, q_heads, NUM_LANES), dtype=jnp.float32),
        ],
    )
    kernel = partial(
        ragged_decode_kernel_fwd,
        kv_seq_len=kv_seq_len,
        block_kv=block_kv,
        block_bs=block_bs,
        scale=scale,
        scale_qk_not_k=scale_qk_not_k,
        scale_s_not_v=scale_s_not_v,
    )
    attn = pl.pallas_call(
        kernel, grid_spec=grid_spec, out_shape=out_shape, interpret=interpret
    )(
        starts,
        lengths,
        chunked_starts,
        chunked_lengths,
        q,
        k,
        v,
        k_scale,
        v_scale,
        qk_prev,
    )
    return attn


################################################################################


def ragged_decode_fwd_ref(
    q: list[jax.Array],  # list[[bs, q_heads, head_dim]]
    k: list[jax.Array],  # list[[bs, kv_seq_len, head_dim]]
    v: jax.Array,  # [bs, kv_seq_len, head_dim]
    starts: jax.Array | None = None,  # [bs]
    lengths: jax.Array | None = None,  # [bs]
    k_scale: list[jax.Array] | None = None,  # list[[bs, kv_seq_len]]
    v_scale: jax.Array | None = None,  # [bs, kv_seq_len]
    qk_prev: jax.Array | None = None,  # [bs, q_heads, kv_seq_len]
    block_qheads: int = 16,
    block_kv: int = 256,
    scale: float | None = None,
):
    scale = math.sqrt(q.shape[-1]) if scale is None else scale
    bs, q_heads, _ = q.shape
    bs_k, kv_seq_len, _ = k.shape
    bs_v, kv_seq_len_v, head_dim_v = v.shape

    if starts is None:
        starts = jnp.zeros((bs,), dtype=jnp.int32)
    if lengths is None:
        lengths = kv_seq_len * jnp.ones((bs,), dtype=jnp.int32)

    qk = jnp.einsum("bqh,bTh->bqT", q, k)
    if k_scale is not None:
        qk = qk * k_scale[..., None, :]
    if qk_prev is not None:
        qk = qk + qk_prev
    qk = qk * scale
    indices = jnp.arange(k.shape[-2])
    mask = (indices >= starts[:, None]) & (indices < lengths[:, None])
    qk = jnp.where(mask[:, None, :], qk, jnp.finfo(qk.dtype).min)
    s = jax.nn.softmax(qk, axis=-1) * (jnp.sum(mask, -1) > 0)[:, None, None]
    if v_scale is not None:
        s = s * v_scale[..., None, :]
    return jnp.einsum("bqT,bTh->bqh", s, v)


def _simple_quantize(
    x: jax.Array, axis: int | tuple[int, ...], scale_dtype=jnp.float16
):
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)
    axis = tuple(z % x.ndim for z in axis)
    amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
    scale = (amax / 127.0 + jnp.finfo(scale_dtype).tiny).astype(scale_dtype)
    quant = jnp.round(x / scale).astype(jnp.int8)
    return quant, scale.reshape([z for i, z in enumerate(scale.shape) if i not in axis])


def test_main(interpret=False):
    bs, q_heads, kv_heads, kv_seq_len, head_dim = 128, 16, 4, 8192, 128
    print((bs, q_heads, kv_heads, kv_seq_len, head_dim))
    dtype = jnp.bfloat16
    mesh = jax.make_mesh((jax.device_count(),), ("x",))

    @partial(jax.jit, static_argnames=("which", "block_kv", "block_bs"))
    def fn(
        q,
        k,
        v,
        starts,
        lengths,
        qk_prev=None,
        which: str = "pallas",
        block_kv: int = 128,
        block_bs: int = 8,
    ):
        k, k_scale = k if isinstance(k, tuple) else (k, None)
        v, v_scale = v if isinstance(v, tuple) else (v, None)
        kv_heads = k.shape[1]
        q_ = q.reshape(q.shape[:1] + (kv_heads, -1) + q.shape[2:])
        if qk_prev is not None:
            qk_prev_ = qk_prev.reshape(
                qk_prev.shape[:1] + (kv_heads, -1) + qk_prev.shape[2:]
            )
        else:
            qk_prev_ = None
        qkv_spec = P(None, "x", None, None)
        in_specs = 3 * (qkv_spec,) + 2 * (P(),)
        in_specs += (P(None, "x", None) if k_scale is not None else None,)
        in_specs += (P(None, "x", None) if v_scale is not None else None,)
        in_specs += (P(None, "x", None, None) if qk_prev is not None else None,)
        out_specs = qkv_spec

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
        def _fn(q, k, v, starts, lengths, k_scale, v_scale, qk_prev):
            in_axes = (1, 1, 1, None, None)
            in_axes += (1 if k_scale is not None else None,)
            in_axes += (1 if v_scale is not None else None,)
            in_axes += (1 if qk_prev is not None else None,)
            if which == "pallas":
                opts = dict(block_kv=block_kv, block_bs=block_bs, interpret=interpret)
                return jax.vmap(
                    partial(ragged_decode_fwd, **opts), in_axes=in_axes, out_axes=1
                )(q, k, v, starts, lengths, k_scale, v_scale, qk_prev)
            else:
                return jax.vmap(ragged_decode_fwd_ref, in_axes=in_axes, out_axes=1)(
                    q, k, v, starts, lengths, k_scale, v_scale, qk_prev
                )

        return _fn(q_, k, v, starts, lengths, k_scale, v_scale, qk_prev_).reshape(
            q.shape
        )

    keyit = iter(random.split(random.key(round(time.time())), 1024))
    q = random.normal(next(keyit), (bs, q_heads, head_dim), dtype=dtype)
    k = random.normal(next(keyit), (bs, kv_heads, kv_seq_len, head_dim), dtype=dtype)
    v = random.normal(next(keyit), (bs, kv_heads, kv_seq_len, head_dim), dtype=dtype)
    mesh = jax.make_mesh((jax.device_count(),), P("x"))
    repl_sharding = NamedSharding(mesh, P())
    q = jax.device_put(
        q / jnp.linalg.norm(q, axis=-1)[..., None],
        NamedSharding(mesh, P(None, "x", None)),
    )
    k = jax.device_put(
        k / jnp.linalg.norm(k, axis=-1)[..., None],
        NamedSharding(mesh, P(None, "x", None, None)),
    )
    v = jax.device_put(
        v / jnp.linalg.norm(v, axis=-1)[..., None],
        NamedSharding(mesh, P(None, "x", None, None)),
    )
    # qk_prev = random.normal(next(keyit), (bs, q_heads, kv_seq_len), dtype=dtype)
    # qk_prev = jax.device_put(qk_prev, NamedSharding(mesh, P(None, "x", None)))
    qk_prev = None

    k = _simple_quantize(k, axis=-1, scale_dtype=jnp.bfloat16)
    v = _simple_quantize(v, axis=-1, scale_dtype=jnp.bfloat16)
    print("quant")

    # starts = random.randint(next(keyit), (bs,), 0, kv_seq_len, dtype=jnp.int32)
    # lengths = jnp.clip(starts + random.randint(next(keyit), (bs,), 0, kv_seq_len, dtype=jnp.int32), max=kv_seq_len)
    starts = jnp.zeros((bs,), dtype=jnp.int32)
    sparsity_factor = 8
    lengths = (
        round(kv_seq_len / sparsity_factor) * jnp.ones((bs,), dtype=jnp.int32)
        + random.randint(
            next(keyit),
            (bs,),
            -kv_seq_len / sparsity_factor,
            kv_seq_len / sparsity_factor,
            dtype=jnp.int32,
        )
        // 2
    )
    print(f"{lengths = }")
    print("sparse cache")
    # lengths = 256 * jnp.ones((bs,), dtype=jnp.int32)
    starts, lengths = (
        jax.device_put(starts, repl_sharding),
        jax.device_put(lengths, repl_sharding),
    )

    k_ = k if not isinstance(k, tuple) else k[0]
    v_ = v if not isinstance(v, tuple) else v[0]
    total_mem = (
        (k_.size * k_.itemsize) + (v_.size * v_.itemsize) + (q.size * q.itemsize)
    ) / jax.device_count()
    print(f"Total memory: {total_mem:.4e}")
    print(f"HBM BW speed: {1e6 * total_mem / 819e9:.4e} us")

    total_mem = (
        (k_.size * k_.itemsize / sparsity_factor)
        + (v_.size * v_.itemsize / sparsity_factor)
        + (q.size * q.itemsize)
    ) / jax.device_count()
    print(f"Sparse total memory: {total_mem:.4e}")
    print(f"Sparse HBM BW speed: {1e6 * total_mem / 819e9:.4e} us")

    ret = fn(q, k, v, starts, lengths, qk_prev, which="pallas")
    ret_ref = fn(q, k, v, starts, lengths, qk_prev, which="ref")
    err = jnp.linalg.norm((ret - ret_ref).astype(jnp.float32), axis=-1)
    err = err / jnp.linalg.norm(ret_ref.astype(jnp.float32), axis=-1)
    err = jnp.mean(err, -1)
    print(f"{err = }")


if __name__ == "__main__":
    test_main()
