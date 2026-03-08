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

import random as pyrandom
from functools import partial
from pathlib import Path

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tqdm import tqdm


def decode_ragged_dot_kernel(
    # scalar prefetch
    lhs_idx_map_ref,  # [g // block_g, n // block_n]
    rhs_idx_map_ref,  # [g // block_g, n // block_n]
    # inputs
    x_ref,  # [block_n, k]
    A_ref,  # [block_g, k, m]
    group_sizes_ref,  # [g]
    # outputs
    y_ref,  # [block_n, m]  # hbm scratch output, to-be-reduced over 0-axis
    # (scratch) persistent lhs idx
    lhs_idx_ref,  # [1]
    group_id_ref,  # [1]
    group_size_ref,  # [1]
    # hyperparameters
    block_n: int,
    block_g: int,
    block_compute: int,
    n: int,
    g: int,
):
    del rhs_idx_map_ref
    pid_g, pid_i = (
        pl.program_id(1),
        pl.program_id(2),
    )  # (out column tiles, matrix groups, lhs row tiles)
    (_, k), _, m = x_ref.shape, A_ref.shape[0], A_ref.shape[-1]
    block_n_id = lhs_idx_map_ref[pid_g, pid_i]

    lhs_idx = jnp.where((pid_g == 0) & (pid_i == 0), 0, lhs_idx_ref[0])
    group_id = jnp.where((pid_g == 0) & (pid_i == 0), 0, group_id_ref[0])
    group_size = jnp.where(
        pid_i == 0, group_sizes_ref[pid_g * block_g], group_size_ref[0]
    )

    idx = jnp.maximum(pid_g * lhs_idx_map_ref.shape[-1] + pid_i - 1, 0)
    prev_block_n_id = lhs_idx_map_ref[
        idx // lhs_idx_map_ref.shape[-1], idx % lhs_idx_map_ref.shape[-1]
    ]
    is_block_n_new = ((pid_g == 0) & (pid_i == 0)) | (prev_block_n_id != block_n_id)

    @pl.when(is_block_n_new)
    def _():
        y_ref[...] = jnp.zeros_like(y_ref)

    # for i in range(lhs_idx // block_compute, n // block_compute): # blockwise over rows in lhs
    def outer_body_fn(i, carry):
        lhs_idx, group_id, group_size = carry
        local_i = i - block_n_id * (block_n // block_compute)
        y = y_ref[pl.ds(local_i * block_compute, block_compute), :].astype(jnp.float32)
        x = x_ref[pl.ds(local_i * block_compute, block_compute), :]

        # iterate until lhs rows are exhausted or we use up all rhs groups
        def cond_fn(val):
            y, lhs_idx, group_id, group_size = val
            del y, group_size
            local_group_id = group_id - pid_g * block_g
            return (lhs_idx < (i + 1) * block_compute) & (
                local_group_id < A_ref.shape[0]
            )

        def body_fn(val):
            y, lhs_idx, group_id, group_size = val

            # check how many valid elements we computed and
            els2compute = jnp.maximum(
                jnp.minimum(
                    group_size, ((i + 1) * block_compute - lhs_idx).astype(jnp.int32)
                ),
                0,
            )
            group_exhausted = els2compute >= group_size
            local_group_id = group_id - pid_g * block_g

            def _compute():
                # compute the actual product with the group_id group
                A = A_ref[local_group_id, :, :]
                xA = jax.lax.dot_general(
                    x, A, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
                )
                xA = xA.astype(y.dtype)
                # write to y accumulator masking already computed values
                iota = (
                    jax.lax.broadcasted_iota(jnp.int32, (block_compute, m), dimension=0)
                    + i * block_compute
                )
                mask = (iota >= lhs_idx) & (iota < (lhs_idx + els2compute))
                return jnp.where(mask, xA, y)

            new_y = jax.lax.cond(els2compute > 0, _compute, lambda: y)

            new_group_id = jnp.where(group_exhausted, group_id + 1, group_id)
            next_group_size = group_sizes_ref[
                jnp.clip(pid_g * block_g + local_group_id + 1, max=g - 1)
            ]
            new_group_size = jnp.where(
                group_exhausted, next_group_size, group_size - els2compute
            )
            new_lhs_idx = lhs_idx + els2compute

            return new_y, new_lhs_idx, new_group_id, new_group_size

        y, new_lhs_idx, new_group_id, new_group_size = jax.lax.while_loop(
            cond_fn, body_fn, (y, lhs_idx, group_id, group_size)
        )
        y_ref[pl.ds(local_i * block_compute, block_compute), :] = y.astype(y_ref.dtype)
        return new_lhs_idx, new_group_id, new_group_size

    start_idx = jnp.maximum(lhs_idx, block_n_id * block_n) // block_compute
    end_idx = jnp.minimum(n, (block_n_id + 1) * block_n) // block_compute
    new_lhs_idx, new_group_id, new_group_size = jax.lax.fori_loop(
        start_idx, end_idx, outer_body_fn, (lhs_idx, group_id, group_size)
    )
    lhs_idx_ref[0], group_id_ref[0], group_size_ref[0] = (
        new_lhs_idx,
        new_group_id,
        new_group_size,
    )


################################################################################


@partial(
    jax.jit,
    static_argnames=("block_n", "block_g", "block_compute", "block_out", "interpret"),
)
def decode_ragged_dot(
    lhs: jax.Array,  # [n, k]
    rhs: jax.Array,  # [g, k, m]
    group_sizes: jax.Array,  # g[]
    block_n: int = int(1e20),  # by default replicate activations fully
    block_g: int = 2,
    block_compute: int = 8,
    block_out: int = int(1e20),  # by default write full output columns at the same time
    interpret: bool = False,
) -> jax.Array:
    """Computes y = x @ A, x.shape=[n, k], A.shape=[g, k, m]; rows in x are assigned to groups via `group_sizes`.

    To use a quantized version pass quantized arguments and either pre-scale lhs before or post-scale the result.

    The implementation attempts to maximize HBM BW of rhs by loading batches along g axis. It works most efficiently
    when lhs fits into VMEM entirely. Alternatively provide `block_n` splits to split lhs.
    THIS REQUIRES [g, n, m] EXTRA HBM SCRATCH.

    Args:
        lhs: The input array of shape (n, k) where groups of rows are raggedly assigned to g axis via `group_sizes`.
        rhs: The stack of matrices (k, m) [g, k, m]. For example g axis can represent experts.
        group_sizes: An array of shape (g,) containing the sizes of each group in the ragged array `A`.
        block_n: Splitting rows in x if activations do not fit in VMEM memory - do not use unless necessary.
        block_g: The batch of group entries in A to preload at the same time.
        block_compute: The compute window moving over dimension n.
        block_out: The tiling of the output columns (to manage vmem usage).
        interpret: Enable the pallas interpret mode.

    Returns:
        The result of the ragged dot product, an array of shape (g, n).
    """
    block_n = min(block_n, lhs.shape[0])
    block_compute = min(block_compute, block_n)
    block_out = 128 * pl.cdiv(
        max(128, min(block_out, rhs.shape[-1])), 128
    )  # min 128, multiple of 128
    assert rhs.ndim == 3 and lhs.ndim == 2, "lhs must have 2 dims, rhs 3 dims"
    assert rhs.shape[0] % block_g == 0, (
        f"{block_g=} must divide {rhs.shape[0]=} (# of groups) must divide"
    )
    assert block_n % block_compute == 0, (
        f"{block_n = } {block_compute = } {lhs.shape = }"
    )
    assert rhs.shape[:1] == group_sizes.shape
    (n, k), (g, _, m) = lhs.shape, rhs.shape

    grid = (pl.cdiv(rhs.shape[-1], block_out), g // block_g, n // block_n)

    # compute lhs prefetch map, only increment lhs idx if work is exhausted to avoid revisiting rows in lhs/output
    # [[ 0 0 1 1 ]
    #  [ 1 1 1 1 ]
    #  [ 1 2 2 2 ]
    #  [ 3 4 4 4 ]]
    work_total = jnp.pad(
        jnp.cumsum(jnp.sum(group_sizes.reshape((-1, block_g)), -1), axis=-1), (1, 0)
    )
    min_lhs_j = work_total[:-1] // block_n
    lhs_idx_map = min_lhs_j[:, None] + jnp.arange(grid[-1])[None, :]
    max_lhs_j = jnp.concatenate([min_lhs_j[1:], jnp.array([grid[-1] - 1])])
    lhs_idx_map = jnp.clip(
        lhs_idx_map, max=jnp.minimum(max_lhs_j[:, None], grid[-1] - 1)
    )

    # compute rhs prefetch map
    # [ 1 1 1 3 4 5 5 5] if 8 groups, but only {1, 3, 4, 5} active
    rhs_work_mask = jnp.sum(group_sizes.reshape((-1, block_g)), -1) > 0
    unique_rhs_groups = jnp.sort(
        jnp.arange(rhs_work_mask.shape[-1]) * rhs_work_mask, descending=True
    )
    flipped_rhs_groups_mapping = jnp.maximum(
        jnp.cumsum(jnp.flip(rhs_work_mask, axis=-1)) - 1, 0
    )
    rhs_idx_map = jnp.flip(unique_rhs_groups[flipped_rhs_groups_mapping], axis=-1)

    def lhs_prefetch(out_i, i, j, lhs_idx_map_ref, rhs_idx_map_ref):
        # as opposed to: `return j, 0`
        del rhs_idx_map_ref
        return lhs_idx_map_ref[i, j], 0

    def rhs_prefetch(out_i, i, j, lhs_idx_map_ref, rhs_idx_map_ref):
        # as opposed to: `return i, 0, 0`
        del j, lhs_idx_map_ref
        return rhs_idx_map_ref[i], 0, out_i

    def out_prefetch(out_i, i, j, lhs_idx_map_ref, rhs_idx_map_ref):
        # as opposed to: `return j, out_i`
        del rhs_idx_map_ref
        return lhs_idx_map_ref[i, j], out_i

    in_specs = [
        pl.BlockSpec((block_n, k), lhs_prefetch),
        pl.BlockSpec((block_g, rhs.shape[-2], block_out), rhs_prefetch),
        pl.BlockSpec(
            (group_sizes.size,), lambda i, j, *_: (0,), memory_space=pltpu.SMEM
        ),
    ]
    out_specs = pl.BlockSpec((block_n, block_out), out_prefetch)

    scratch_shapes = [
        pltpu.SMEM((1,), dtype=jnp.int32),
        pltpu.SMEM((1,), dtype=jnp.int32),
        pltpu.SMEM((1,), dtype=jnp.int32),
    ]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
    )
    out_shape = jax.ShapeDtypeStruct((n, m), dtype=lhs.dtype)
    return pl.pallas_call(
        partial(
            decode_ragged_dot_kernel,
            block_n=block_n,
            block_g=block_g,
            block_compute=block_compute,
            n=n,
            g=g,
        ),
        out_shape=out_shape,
        grid_spec=grid_spec,
        interpret=interpret,
    )(lhs_idx_map, rhs_idx_map, lhs, rhs, group_sizes.astype(jnp.int32))


@partial(jax.jit, static_argnames=("block_n", "block_g", "block_compute", "block_out"))
def decode_ragged_dot_ref(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    block_n: int = 64,
    block_g: int = 16,
    block_out: int = 2**30,
    block_compute: int = 8,
) -> jax.Array:
    return jax.lax.ragged_dot(lhs, rhs, group_sizes)


def test_tune():
    import tune_jax
    from jax.experimental.layout import Format, Layout

    tune_jax.CONFIG.allow_fallback_timing = False
    tune_jax.logger.setLevel("DEBUG")
    seed = 25
    g, n, k, m = 32, 1024, 2880, 1440

    keys = iter(random.split(random.key(seed), 1024))
    x = random.normal(next(keys), (n, k), dtype=jnp.bfloat16)
    A = random.normal(next(keys), (g, k, m), dtype=jnp.bfloat16)
    A = A / jnp.linalg.norm(A, axis=-1)[..., None]
    A = jnp.round(A * 127).astype(jnp.int8)
    A = jax.device_put(
        A, Format(Layout((0, 1, 2), tiling=((8, 128), (4, 1))), A.sharding)
    )

    group_sizes = jnp.exp(1e-1 * random.uniform(next(keys), g))
    group_sizes = jnp.round(n * (group_sizes / jnp.sum(group_sizes))).astype(jnp.int32)
    while jnp.sum(group_sizes) > n:
        idx = jnp.argmax(group_sizes)
        group_sizes = group_sizes.at[idx].set(group_sizes[idx] - 1)
    while jnp.sum(group_sizes) < n:
        idx = jnp.argmax(group_sizes)
        group_sizes = group_sizes.at[idx].set(group_sizes[idx] + 1)

    print(jnp.sum(group_sizes))
    print(group_sizes)
    assert jnp.sum(group_sizes) <= n

    # place the inputs with optimal shardings so that no copies for data-reformatting are included in tuning
    auto_layouts = jax.tree.map(lambda x: Format(Layout.AUTO), (x, A, group_sizes))
    shapes = jax.tree.map(jax.typeof, (x, A, group_sizes))
    opt_shrd = (
        jax.jit(decode_ragged_dot, in_shardings=auto_layouts)
        .lower(*shapes)
        .compile()
        .input_formats[0]
    )
    x, A, group_sizes = jax.device_put((x, A, group_sizes), opt_shrd)

    hyperparams = dict(
        block_n=[8, 16, 1e20],
        block_compute=[4, 8, 16, 32],
        block_g=[1, 2, 4, 8],
        block_out=[128, 256, 512, 1024, 2048, 4096],
    )

    fn = tune_jax.tune(decode_ragged_dot, hyperparams=hyperparams)
    fn(x, A, group_sizes)
    print(tune_jax.tabulate(fn))


def test_profile_speed(interpret):
    seed = 25
    # n, k, g, m = 32, 128, 64, 256
    # n, k, g, m = 64, 128, 64, 7168
    n, k, g, m = 64, 7168, 64, 128

    # k, m = m, k

    keys = iter(random.split(random.key(seed), 1024))
    x = random.normal(next(keys), (n, k), dtype=jnp.bfloat16)
    A = random.normal(next(keys), (g, k, m), dtype=jnp.bfloat16)
    A = A / jnp.linalg.norm(A, axis=-1)[..., None]
    A = jnp.round(A * 127).astype(jnp.int8)

    block_g, block_compute, block_n = g // 8, 8, n // 4

    group_sizes = jnp.exp(1e1 * random.uniform(next(keys), g))
    group_sizes = jnp.round(n * (group_sizes / jnp.sum(group_sizes))).astype(jnp.int32)

    # group_sizes = jnp.zeros(g, dtype=jnp.int32)
    # group_sizes = group_sizes.at[7].set(n)
    print(group_sizes)
    print(group_sizes.reshape((-1, block_g)))
    print(jnp.sum(group_sizes))
    assert jnp.sum(group_sizes) <= n

    opts = dict(
        block_n=block_n,
        block_g=block_g,
        block_compute=block_compute,
        interpret=interpret,
    )
    for _ in range(1):
        ret = decode_ragged_dot(x, A, group_sizes, **opts).block_until_ready()
        ret_ref = decode_ragged_dot_ref(x, A, group_sizes).block_until_ready()
        print(
            f"error = {float(jnp.linalg.norm(ret - ret_ref) / (jnp.linalg.norm(ret_ref) + 1e-5)):.4e}"
        )
    rowwise_error = jnp.linalg.norm((ret - ret_ref).astype(jnp.float32), axis=-1) / (
        jnp.linalg.norm(ret_ref.astype(jnp.float32), axis=-1) + 1e-7
    )
    print(f"mean row error = {jnp.mean(rowwise_error):.4e}")
    print(f"row-wise error = {rowwise_error}")
    print(1 * (jnp.arange(group_sizes.size) < jnp.sum(group_sizes)))

    opts = dict(block_n=block_n, block_g=block_g, block_compute=block_compute)
    with jax.profiler.trace(str(Path("~/profiles/decode_ragged2").expanduser())):
        for _ in range(3):
            ret = decode_ragged_dot(x, A, group_sizes, **opts).block_until_ready()
            s = jnp.linalg.norm(
                ret
            ).block_until_ready()  # dummy computation as a profile barrier

        for _ in range(3):
            ret = decode_ragged_dot_ref(x, A, group_sizes, **opts).block_until_ready()
            s = jnp.linalg.norm(
                ret
            ).block_until_ready()  # dummy computation as a profile barrier


########################################################################################################################


def _numeric_test_case(seed, interpret, n, k, g, m, block_g, block_n, block_compute):
    keys = iter(random.split(random.key(seed), 1024))
    x = random.normal(next(keys), (n, k), dtype=jnp.bfloat16)
    A = random.normal(next(keys), (g, k, m), dtype=jnp.bfloat16)
    # A = A / jnp.linalg.norm(A, axis=-1)[..., None]
    # A = jnp.round(A * 127).astype(jnp.int8)

    group_sizes = jnp.exp(1e-2 * random.uniform(next(keys), g))
    group_sizes = jnp.round(n * (group_sizes / jnp.sum(group_sizes))).astype(jnp.int32)
    while jnp.sum(group_sizes) > n:
        idx = jnp.argmax(group_sizes)
        group_sizes = group_sizes.at[idx].set(group_sizes[idx] - 1)
    assert jnp.sum(group_sizes) <= n

    opts = dict(block_n=block_n, block_g=block_g, block_compute=block_compute)
    try:
        ret = decode_ragged_dot(
            x, A, group_sizes, **opts, interpret=interpret
        ).block_until_ready()
    except jax.errors.JaxRuntimeError:
        return float("nan")
    ret_ref = decode_ragged_dot_ref(x, A, group_sizes).block_until_ready()
    error = float(jnp.linalg.norm(ret - ret_ref) / (jnp.linalg.norm(ret_ref) + 1e-5))
    return error


def test_numerics():
    tests = [
        (seed, n, k, g, m, g // g_splits, n // n_splits, block_compute)
        for seed in [0, 1, 2]
        for n in [128, 64, 32]
        for k in [128, 7168]
        for g in [32, 64, 256]
        for m in [7168, 128]
        for block_compute in [8, 16]
        for g_splits in [1, 2, 4, 8]
        for n_splits in [1, 2, 4, 8]
        if n // n_splits >= block_compute
        and g // g_splits >= 8
        and not (m == 7168 and k == 7168)
    ]
    pyrandom.shuffle(tests)

    max_error = 0
    it = 0
    for seed, n, k, g, m, block_g, block_n, block_compute in tqdm(tests):
        error = _numeric_test_case(
            seed, False, n, k, g, m, block_g, block_n, block_compute
        )
        error = max(error, max_error)
        if max_error > 1e-4:
            raise ValueError(
                f"failing {(seed, n, k, g, m, block_g, block_n, block_compute)=} with {error=:.4}"
            )
        if (it + 1) % 100 == 0:
            tqdm.write(f"{max_error = :.4e}")
        it += 1


if __name__ == "__main__":
    # test_numerics()
    test_tune()
