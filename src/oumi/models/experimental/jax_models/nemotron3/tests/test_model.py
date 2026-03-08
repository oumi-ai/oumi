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

import dataclasses

import jax
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random
from jax.sharding import AxisType, set_mesh
from jax.sharding import PartitionSpec as P

try:
    from jax.sharding import use_mesh as set_mesh  # for jax < 0.7.0
except ImportError:
    pass

from nemotron3_jax import model as n3jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)

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


class TestNemotron3Model(parameterized.TestCase):
    def setUp(self):
        self.mesh = jax.make_mesh(
            (1, len(jax.devices()), 1),
            P("x", "y", "z"),
            axis_types=(AxisType.Explicit,) * 3,
        )
        self.small_cfg = dataclasses.replace(SMALL_CFG, mesh=self.mesh)

    @parameterized.product(quant=[False, True])
    def test_model_init(self, quant):
        cfg = dataclasses.replace(
            self.small_cfg, quant_attn=quant, quant_moe=quant, quant_mamba=quant
        )
        weights = n3jax.Weights.init(random.key(0), cfg)
        del weights

    @parameterized.product(quant=[False, True])
    def test_cache_init(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quant_cache=quant)
        cache = n3jax.KVCache.init(random.key(0), cfg, 2, cfg.max_seq_len)
        del cache

    @parameterized.product(quant=[False, True])
    def test_init_hashing(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quant_cache=quant)
        hash_fn = lambda x: hash(tuple(jax.tree.leaves(x, is_leaf=n3jax.is_param)))
        with self.subTest("Testing weights abstract and shardings hashing"):
            abstract = n3jax.Weights.abstract(cfg)
            abstract2 = n3jax.Weights.abstract(cfg)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))

        with self.subTest("Testing kv-cache abstract and shardings hashing"):
            abstract = n3jax.KVCache.abstract(cfg, 2)
            abstract2 = n3jax.KVCache.abstract(cfg, 2)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))

    @parameterized.product(quant_weights=[False, True], quant_cache=[True, False])
    def test_prefill_decode(self, quant_weights, quant_cache):
        cfg = dataclasses.replace(
            self.small_cfg,
            quant_attn=quant_weights,
            quant_moe=quant_weights,
            quant_mamba=quant_weights,
            quant_cache=quant_cache,
        )
        tokens = jnp.ones((1, 16), dtype=jnp.int32)
        weights = n3jax.Weights.init(random.key(0), cfg)
        cache = n3jax.KVCache.init(
            random.key(0), cfg, tokens.shape[0], cfg.max_seq_len
        )
        with set_mesh(cfg.mesh):
            max_tokens, _, cache = n3jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        with set_mesh(cfg.mesh):
            for _ in range(2):
                next_tokens, cache = n3jax.decode_step(
                    next_tokens, weights, cache, cfg
                )


if __name__ == "__main__":
    absltest.main()
