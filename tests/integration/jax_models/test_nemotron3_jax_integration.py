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

"""Integration tests for Nemotron3 JAX model implementation.

Nemotron 3 Nano is a hybrid Mamba-Transformer model.
No upstream test exists yet; this follows the same patterns as other models.
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

from oumi.models.experimental.jax_models.nemotron3.nemotron3_jax import (  # noqa: E402
    model as n3jax,
)

try:
    from jax.sharding import set_mesh  # noqa: E402
except ImportError:
    from jax.sharding import use_mesh as set_mesh  # noqa: E402

pytestmark = pytest.mark.jax

# Minimal config: 2 layers with pattern M* (one Mamba, one MLP)
SMALL_CFG = n3jax.Config(
    embed=64,
    q_heads=8,
    kv_heads=2,
    num_layers=2,
    head_dim=32,
    vocab_size=128,
    max_seq_len=64,
    layer_pattern="M*",
    causal=True,
    # MoE params (minimal)
    moe_ffw_size=64,
    moe_experts_per_tok=2,
    moe_num_experts=4,
    moe_shared_ffw_size=64,
    moe_router_n_groups=1,
    moe_router_topk_groups=1,
    moe_routed_scaling_factor=1.0,
    ep_strategy="decode",
    moe_gate_dtype=jnp.bfloat16,
    # Mamba params (minimal)
    mamba_intermediate_size=128,
    mamba_conv_dim=64,
    mamba_num_heads=4,
    mamba_conv_kernel_size=4,
    mamba_n_groups=4,
    mamba_ssm_state_size=16,
    mamba_time_step_limit=(1, 100),
    mamba_head_dim=32,
    mamba_chunk_size=32,
)


@pytest.fixture()
def mesh():
    return jax.make_mesh(
        (1, len(jax.devices()), 1),
        P("x", "y", "z"),
        axis_types=(AxisType.Explicit,) * 3,
    )


@pytest.fixture()
def small_cfg(mesh):
    return dataclasses.replace(SMALL_CFG, mesh=mesh)


class TestNemotron3JAXIntegration:
    """Integration tests for Nemotron3 JAX model."""

    def test_model_init(self, small_cfg):
        """Weights init should succeed on CPU."""
        cfg = dataclasses.replace(
            small_cfg, quant_attn=False, quant_moe=False, quant_mamba=False
        )
        weights = n3jax.Weights.init(random.key(0), cfg)
        assert weights is not None

    def test_cache_init(self, small_cfg):
        """KVCache init with 4 args for Nemotron3."""
        cache = n3jax.KVCache.init(random.key(0), small_cfg, 2, small_cfg.max_seq_len)
        assert cache is not None

    def test_init_hashing(self, small_cfg):
        """Abstract shapes and shardings should be deterministically hashable."""

        def hash_fn(x):
            return hash(tuple(jax.tree.leaves(x, is_leaf=n3jax.is_param)))

        abstract1 = n3jax.Weights.abstract(small_cfg)
        abstract2 = n3jax.Weights.abstract(small_cfg)
        assert hash_fn(abstract1) == hash_fn(abstract2)

    def test_prefill_decode(self, small_cfg):
        """Prefill then decode with set_mesh (Nemotron3 pattern)."""
        cfg = dataclasses.replace(
            small_cfg,
            quant_attn=False,
            quant_moe=False,
            quant_mamba=False,
            quant_cache=False,
        )
        tokens = jnp.ones((1, 16), dtype=jnp.int32)
        weights = n3jax.Weights.init(random.key(0), cfg)
        cache = n3jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
        with set_mesh(cfg.mesh):
            max_tokens, _, cache = n3jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        with set_mesh(cfg.mesh):
            for _ in range(2):
                next_tokens, cache = n3jax.decode_step(next_tokens, weights, cache, cfg)
        assert next_tokens.shape[0] == 1
        assert next_tokens.dtype == jnp.int32
