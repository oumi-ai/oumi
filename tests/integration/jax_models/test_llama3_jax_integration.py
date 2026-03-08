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

"""Integration tests for Llama3 JAX model implementation.

Follows upstream jax-llm-examples/llama3/tests/test_model.py patterns.
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

from oumi.models.experimental.jax_models.llama3.llama3_jax import (  # noqa: E402
    model as l3jax,
)

try:
    from jax.sharding import set_mesh  # noqa: E402
except ImportError:
    from jax.sharding import use_mesh as set_mesh  # noqa: E402

pytestmark = pytest.mark.jax

# Full config with reduced vocab for testing
CFG = l3jax.Config(
    embed=4096,
    ffw_size=14336,
    q_heads=32,
    kv_heads=8,
    num_layers=32,
    head_dim=128,
    vocab_size=128,
    max_seq_len=128,
    causal=True,
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
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
    return dataclasses.replace(CFG, mesh=mesh, num_layers=2, embed=256)


class TestLlama3JAXIntegration:
    """Integration tests for Llama3 JAX model, following upstream patterns."""

    def test_model_init(self, small_cfg):
        """Weights init should succeed on CPU."""
        cfg = dataclasses.replace(small_cfg, quant_layer=False)
        weights = l3jax.Weights.init(random.key(0), cfg)
        assert weights is not None
        assert hasattr(weights, "layers")

    def test_cache_init(self, small_cfg):
        """KVCache init with 3 args (no max_seq_len) for Llama3."""
        cache = l3jax.KVCache.init(random.key(0), small_cfg, 2)
        assert cache is not None

    def test_init_hashing(self, small_cfg):
        """Abstract shapes and shardings should be deterministically hashable."""

        def hash_fn(x):
            return hash(tuple(jax.tree.leaves(x, is_leaf=l3jax.is_param)))

        abstract1 = l3jax.Weights.abstract(small_cfg)
        abstract2 = l3jax.Weights.abstract(small_cfg)
        assert hash_fn(abstract1) == hash_fn(abstract2)

        shardings1 = l3jax.Weights.shardings(small_cfg)
        shardings2 = l3jax.Weights.shardings(small_cfg)
        assert hash_fn(shardings1) == hash_fn(shardings2)

    def test_prefill_decode(self, small_cfg):
        """Prefill then decode should produce valid token outputs."""
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = l3jax.Weights.init(random.key(0), small_cfg)
        cache = l3jax.KVCache.init(random.key(0), small_cfg, tokens.shape[0])
        with set_mesh(small_cfg.mesh):
            max_tokens, _, cache = l3jax.prefill(tokens, weights, cache, small_cfg)
        next_tokens = max_tokens[:, :-1]
        with set_mesh(small_cfg.mesh):
            for _ in range(2):
                next_tokens, cache = l3jax.decode_step(
                    next_tokens, weights, cache, small_cfg
                )
        assert next_tokens.shape[0] == 1
        assert next_tokens.dtype == jnp.int32
