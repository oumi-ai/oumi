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

"""Integration tests for DeepSeek R1 JAX model implementation.

Follows upstream jax-llm-examples/deepseek_r1_jax/tests/test_model.py patterns.
Note: DeepSeek R1 does NOT use set_mesh context, and uses quantize_* naming.
"""

import dataclasses
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax  # noqa: E402
import pytest  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax import random  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

jax.config.update("jax_platforms", "cpu")

from oumi.models.experimental.jax_models.deepseek_r1_jax.deepseek_r1_jax import (  # noqa: E402
    model as dsjax,
)

pytestmark = pytest.mark.jax

SMALL_CFG = dsjax.Config(
    num_layers=4,
    embed=256,
    n_routed_experts=32,
    ffw_size=128,
    moe_ffw_size=64,
    q_lora_rank=32,
    kv_lora_rank=64,
    num_heads=4,
    qk_nope_head_dim=56,
    qk_rope_head_dim=48,
    v_head_dim=16,
    use_decode_ragged_dot_kernel=False,
)


@pytest.fixture()
def mesh():
    return jax.make_mesh(
        (1, len(jax.devices()), 1),
        P("x", "y", "z"),
        axis_types=(jax.sharding.AxisType.Auto,) * 3,
    )


@pytest.fixture()
def small_cfg(mesh):
    return dataclasses.replace(SMALL_CFG, mesh=mesh)


class TestDeepSeekR1JAXIntegration:
    """Integration tests for DeepSeek R1 JAX model, following upstream patterns."""

    def test_model_init(self, small_cfg):
        """Weights init should succeed on CPU."""
        cfg = dataclasses.replace(small_cfg, quantize_attn=False, quantize_moe=False)
        weights = dsjax.Weights.init(random.key(0), cfg)
        assert weights is not None

    def test_cache_init(self, small_cfg):
        """KVCache init with 4 args for DeepSeek R1."""
        cache = dsjax.KVCache.init(random.key(0), small_cfg, 2, small_cfg.max_seq_len)
        assert cache is not None

    def test_init_hashing(self, small_cfg):
        """Abstract shapes and shardings should be deterministically hashable."""

        def hash_fn(x):
            return hash(tuple(jax.tree.leaves(x, is_leaf=dsjax.is_param)))

        abstract1 = dsjax.Weights.abstract(small_cfg)
        abstract2 = dsjax.Weights.abstract(small_cfg)
        assert hash_fn(abstract1) == hash_fn(abstract2)

        shardings1 = dsjax.Weights.shardings(small_cfg)
        shardings2 = dsjax.Weights.shardings(small_cfg)
        assert hash_fn(shardings1) == hash_fn(shardings2)

    def test_prefill_decode(self, small_cfg):
        """Prefill then decode without set_mesh (DeepSeek R1 pattern)."""
        cfg = dataclasses.replace(
            small_cfg, quantize_attn=False, quantize_moe=False, quantize_cache=False
        )
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = dsjax.Weights.init(random.key(0), cfg)
        cache = dsjax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
        # DeepSeek R1 does NOT use set_mesh context
        max_tokens, _, cache = dsjax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        for _ in range(2):
            next_tokens, cache = dsjax.decode_step(next_tokens, weights, cache, cfg)
        assert next_tokens.shape[0] == 1
        assert next_tokens.dtype == jnp.int32
