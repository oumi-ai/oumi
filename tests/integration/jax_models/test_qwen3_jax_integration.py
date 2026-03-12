# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for Qwen3 JAX model implementation.

Follows upstream jax-llm-examples/qwen3/tests/test_model.py patterns.
Tests both MOE and Dense configurations.
"""

import dataclasses
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax  # noqa: E402
import pytest  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax import random  # noqa: E402
from jax.sharding import AxisType  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

jax.config.update("jax_platforms", "cpu")

from oumi.models.experimental.jax_models.qwen3.qwen3_jax import (  # noqa: E402
    model as q3jax,
)

try:
    from jax.sharding import set_mesh  # noqa: E402
except ImportError:
    from jax.sharding import use_mesh as set_mesh  # noqa: E402

pytestmark = pytest.mark.jax

MOE_CFG = q3jax.Config(
    embed=2048,
    q_heads=32,
    kv_heads=4,
    num_layers=48,
    head_dim=128,
    vocab_size=151936,
    max_seq_len=128,
    causal=True,
    moe_ffw_size=768,
    moe_experts_per_tok=8,
    moe_num_experts=128,
    ep_strategy="decode",
    mlp_ffw_size=6144,
    mlp_layer_idxs=[],
)

DENSE_CFG = q3jax.Config(
    embed=5120,
    q_heads=64,
    kv_heads=8,
    num_layers=64,
    head_dim=128,
    vocab_size=151936,
    max_seq_len=128,
    causal=True,
    moe_ffw_size=-1,
    moe_experts_per_tok=None,
    moe_num_experts=None,
    ep_strategy="decode",
    mlp_ffw_size=25600,
    mlp_layer_idxs=[],
)


@pytest.fixture()
def mesh():
    return jax.make_mesh(
        (1, len(jax.devices()), 1),
        P("x", "y", "z"),
        axis_types=(AxisType.Explicit,) * 3,
    )


@pytest.fixture()
def small_moe_cfg(mesh):
    return dataclasses.replace(
        MOE_CFG, mesh=mesh, num_layers=2, embed=32, vocab_size=128
    )


@pytest.fixture()
def small_dense_cfg(mesh):
    return dataclasses.replace(
        DENSE_CFG, mesh=mesh, num_layers=2, embed=32, vocab_size=128
    )


class TestQwen3JAXIntegration:
    """Integration tests for Qwen3 JAX model, following upstream patterns."""

    def test_model_init_moe(self, small_moe_cfg):
        """MOE weights init should succeed."""
        cfg = dataclasses.replace(
            small_moe_cfg, quant_attn=False, quant_moe=False, quant_mlp=False
        )
        weights = q3jax.Weights.init(random.key(0), cfg)
        assert weights is not None

    def test_model_init_dense(self, small_dense_cfg):
        """Dense weights init should succeed."""
        cfg = dataclasses.replace(
            small_dense_cfg, quant_attn=False, quant_moe=False, quant_mlp=False
        )
        weights = q3jax.Weights.init(random.key(0), cfg)
        assert weights is not None

    def test_cache_init(self, small_moe_cfg):
        """KVCache init with 4 args for Qwen3."""
        cache = q3jax.KVCache.init(
            random.key(0), small_moe_cfg, 2, small_moe_cfg.max_seq_len
        )
        assert cache is not None

    def test_init_hashing(self, small_moe_cfg):
        """Abstract shapes and shardings should be deterministically hashable."""

        def hash_fn(x):
            return hash(tuple(jax.tree.leaves(x, is_leaf=q3jax.is_param)))

        abstract1 = q3jax.Weights.abstract(small_moe_cfg)
        abstract2 = q3jax.Weights.abstract(small_moe_cfg)
        assert hash_fn(abstract1) == hash_fn(abstract2)

    def test_prefill_decode_moe(self, small_moe_cfg):
        """Prefill then decode with MOE config."""
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = q3jax.Weights.init(random.key(0), small_moe_cfg)
        cache = q3jax.KVCache.init(
            random.key(0), small_moe_cfg, tokens.shape[0], small_moe_cfg.max_seq_len
        )
        with set_mesh(small_moe_cfg.mesh):
            max_tokens, _, cache = q3jax.prefill(tokens, weights, cache, small_moe_cfg)
        next_tokens = max_tokens[:, :-1]
        with set_mesh(small_moe_cfg.mesh):
            for _ in range(2):
                next_tokens, cache = q3jax.decode_step(
                    next_tokens, weights, cache, small_moe_cfg
                )
        assert next_tokens.shape[0] == 1
        assert next_tokens.dtype == jnp.int32
