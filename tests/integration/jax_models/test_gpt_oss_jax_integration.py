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

"""Integration tests for GPT-OSS JAX model implementation.

Follows upstream jax-llm-examples/gpt_oss/tests/test_model.py patterns.
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

from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (  # noqa: E402
    model as gpt_jax,
)

try:
    from jax.sharding import set_mesh  # noqa: E402
except ImportError:
    from jax.sharding import use_mesh as set_mesh  # noqa: E402

pytestmark = pytest.mark.jax

MOE_CFG = gpt_jax.Config(
    embed=2880,
    q_heads=64,
    kv_heads=8,
    num_layers=24,
    head_dim=64,
    vocab_size=201088,
    max_seq_len=128,
    causal=True,
    moe_ffw_size=2880,
    moe_experts_per_tok=4,
    moe_num_experts=32,
    sliding_window_size=128,
    sliding_attention_map=[
        "sliding_attention" if i % 2 == 0 else "full_attention" for i in range(24)
    ],
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
def small_moe_cfg(mesh):
    return dataclasses.replace(
        MOE_CFG,
        mesh=mesh,
        num_layers=2,
        embed=32,
        vocab_size=128,
        sliding_attention_map=["sliding_attention", "full_attention"],
    )


class TestGptOssJAXIntegration:
    """Integration tests for GPT-OSS JAX model, following upstream patterns."""

    def test_model_init(self, small_moe_cfg):
        """Weights init should succeed on CPU."""
        cfg = dataclasses.replace(small_moe_cfg, quant_attn=False, quant_moe=False)
        weights = gpt_jax.Weights.init(random.key(0), cfg)
        assert weights is not None

    def test_cache_init(self, small_moe_cfg):
        """KVCache init with 4 args for GPT-OSS."""
        cache = gpt_jax.KVCache.init(
            random.key(0), small_moe_cfg, 2, small_moe_cfg.max_seq_len
        )
        assert cache is not None

    def test_init_hashing(self, small_moe_cfg):
        """Abstract shapes and shardings should be deterministically hashable."""

        def hash_fn(x):
            return hash(tuple(jax.tree.leaves(x, is_leaf=gpt_jax.is_param)))

        abstract1 = gpt_jax.Weights.abstract(small_moe_cfg)
        abstract2 = gpt_jax.Weights.abstract(small_moe_cfg)
        assert hash_fn(abstract1) == hash_fn(abstract2)

    def test_prefill_decode(self, small_moe_cfg):
        """Prefill then decode with set_mesh (GPT-OSS pattern)."""
        cfg = dataclasses.replace(
            small_moe_cfg, quant_attn=False, quant_moe=False, quant_cache=False
        )
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = gpt_jax.Weights.init(random.key(0), cfg)
        cache = gpt_jax.KVCache.init(
            random.key(0), cfg, tokens.shape[0], cfg.max_seq_len
        )
        with set_mesh(cfg.mesh):
            max_tokens, _, cache = gpt_jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        with set_mesh(cfg.mesh):
            for _ in range(2):
                next_tokens, cache = gpt_jax.decode_step(
                    next_tokens, weights, cache, cfg
                )
        assert next_tokens.shape[0] == 1
        assert next_tokens.dtype == jnp.int32
