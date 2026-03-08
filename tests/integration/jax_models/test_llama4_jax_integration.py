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

"""Integration tests for Llama4 JAX model implementation.

Follows upstream jax-llm-examples/llama4/tests/test_model.py patterns.
Tests both Scout and Maverick configurations.
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

from oumi.models.experimental.jax_models.llama4.llama4_jax import (  # noqa: E402
    model as l4jax,
)

try:
    from jax.sharding import set_mesh  # noqa: E402
except ImportError:
    from jax.sharding import use_mesh as set_mesh  # noqa: E402

pytestmark = pytest.mark.jax

SCOUT_CFG = l4jax.Config(
    embed=5120,
    q_heads=40,
    kv_heads=8,
    num_layers=48,
    head_dim=128,
    vocab_size=202048,
    max_seq_len=128,
    causal=True,
    nope_layer_interval=4,
    use_qk_norm=True,
    attn_chunk_size=8192,
    mlp_ffw_size=16384,
    moe_ffw_size=8192,
    moe_layer_interval=1,
    moe_experts_per_tok=1,
    moe_num_experts=16,
    moe_num_shared_experts=1,
    ep_strategy="decode",
)

MAVERICK_CFG = l4jax.Config(
    embed=5120,
    q_heads=40,
    kv_heads=8,
    num_layers=48,
    head_dim=128,
    vocab_size=202048,
    max_seq_len=128,
    causal=True,
    nope_layer_interval=4,
    use_qk_norm=False,
    attn_chunk_size=8192,
    mlp_ffw_size=16384,
    moe_ffw_size=8192,
    moe_layer_interval=2,
    moe_experts_per_tok=1,
    moe_num_experts=128,
    moe_num_shared_experts=1,
    ep_strategy="decode",
)


@pytest.fixture()
def mesh():
    return jax.make_mesh(
        (1, len(jax.devices()), 1),
        P("x", "y", "z"),
        axis_types=(AxisType.Explicit,) * 3,
    )


@pytest.fixture()
def small_scout_cfg(mesh):
    return dataclasses.replace(
        SCOUT_CFG, mesh=mesh, num_layers=4, embed=64, moe_num_experts=4, vocab_size=128
    )


@pytest.fixture()
def small_maverick_cfg(mesh):
    return dataclasses.replace(
        MAVERICK_CFG,
        mesh=mesh,
        num_layers=4,
        embed=64,
        moe_num_experts=4,
        vocab_size=128,
    )


class TestLlama4JAXIntegration:
    """Integration tests for Llama4 JAX model, following upstream patterns."""

    def test_model_init_scout(self, small_scout_cfg):
        """Scout weights init should succeed."""
        cfg = dataclasses.replace(
            small_scout_cfg, quant_attn=False, quant_mlp=False, quant_moe=False
        )
        weights = l4jax.Weights.init(random.key(0), cfg)
        assert weights is not None

    def test_model_init_maverick(self, small_maverick_cfg):
        """Maverick weights init should succeed."""
        cfg = dataclasses.replace(
            small_maverick_cfg, quant_attn=False, quant_mlp=False, quant_moe=False
        )
        weights = l4jax.Weights.init(random.key(0), cfg)
        assert weights is not None

    def test_cache_init(self, small_scout_cfg):
        """KVCache init with 4 args (including max_seq_len) for Llama4."""
        cache = l4jax.KVCache.init(
            random.key(0), small_scout_cfg, 2, small_scout_cfg.max_seq_len
        )
        assert cache is not None

    def test_init_hashing(self, small_scout_cfg):
        """Abstract shapes and shardings should be deterministically hashable."""

        def hash_fn(x):
            return hash(tuple(jax.tree.leaves(x, is_leaf=l4jax.is_param)))

        abstract1 = l4jax.Weights.abstract(small_scout_cfg)
        abstract2 = l4jax.Weights.abstract(small_scout_cfg)
        assert hash_fn(abstract1) == hash_fn(abstract2)

    def test_prefill_decode_scout(self, small_scout_cfg):
        """Prefill then decode with Scout config."""
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = l4jax.Weights.init(random.key(0), small_scout_cfg)
        cache = l4jax.KVCache.init(
            random.key(0), small_scout_cfg, tokens.shape[0], small_scout_cfg.max_seq_len
        )
        with set_mesh(small_scout_cfg.mesh):
            max_tokens, _, cache = l4jax.prefill(
                tokens, weights, cache, small_scout_cfg
            )
        next_tokens = max_tokens[:, :-1]
        with set_mesh(small_scout_cfg.mesh):
            for _ in range(2):
                next_tokens, cache = l4jax.decode_step(
                    next_tokens, weights, cache, small_scout_cfg
                )
        assert next_tokens.shape[0] == 1
        assert next_tokens.dtype == jnp.int32
