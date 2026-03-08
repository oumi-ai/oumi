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
    from jax.sharding import use_mesh as set_mesh  # jax < 0.7.0
except ImportError:
    pass

from llama4_jax import model as l4jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)

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


class TestModel(parameterized.TestCase):
    def setUp(self):
        self.mesh = jax.make_mesh(
            (1, len(jax.devices()), 1),
            P("x", "y", "z"),
            axis_types=(AxisType.Explicit,) * 3,
        )
        self.small_scout_cfg = dataclasses.replace(
            SCOUT_CFG,
            mesh=self.mesh,
            num_layers=4,
            embed=64,
            moe_num_experts=4,
            vocab_size=128,
        )
        self.small_maverick_cfg = dataclasses.replace(
            MAVERICK_CFG,
            mesh=self.mesh,
            num_layers=4,
            embed=64,
            moe_num_experts=4,
            vocab_size=128,
        )

    @parameterized.product(scout=[True, False], quant=[False, True])
    def test_model_init(self, scout, quant):
        cfg = self.small_scout_cfg if scout else self.small_maverick_cfg
        cfg = dataclasses.replace(
            cfg, quant_attn=quant, quant_mlp=quant, quant_moe=quant
        )
        weights = l4jax.Weights.init(random.key(0), cfg)
        del weights

    @parameterized.product(quant=[False, True])
    def test_init_hashing(self, quant):
        cfg = dataclasses.replace(self.small_scout_cfg, quant_cache=quant)
        hash_fn = lambda x: hash(tuple(jax.tree.leaves(x, is_leaf=l4jax.is_param)))
        with self.subTest("Testing weights abstract and shardings hashing"):
            abstract = l4jax.Weights.abstract(cfg)
            abstract2 = l4jax.Weights.abstract(cfg)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = l4jax.Weights.shardings(cfg)
            shardings2 = l4jax.Weights.shardings(cfg)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

        with self.subTest("Testing kv-cache abstract and shardings hashing"):
            abstract = l4jax.KVCache.abstract(cfg, 2, cfg.max_seq_len)
            abstract2 = l4jax.KVCache.abstract(cfg, 2, cfg.max_seq_len)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = l4jax.KVCache.shardings(cfg, 2, cfg.max_seq_len)
            shardings2 = l4jax.KVCache.shardings(cfg, 2, cfg.max_seq_len)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

    @parameterized.product(scout=[True, False], quant=[False, True])
    def test_cache_init(self, scout, quant):
        cfg = self.small_scout_cfg if scout else self.small_maverick_cfg
        cfg = dataclasses.replace(cfg, quant_cache=quant)
        cache = l4jax.KVCache.init(random.key(0), cfg, 2, cfg.max_seq_len)
        del cache

    @parameterized.product(
        scout=[True, False], quant_weights=[False, True], quant_cache=[True, False]
    )
    def test_prefill_decode(self, scout, quant_weights, quant_cache):
        cfg = self.small_scout_cfg if scout else self.small_maverick_cfg
        cfg = dataclasses.replace(
            cfg,
            quant_attn=quant_weights,
            quant_mlp=quant_weights,
            quant_moe=quant_weights,
            quant_cache=quant_cache,
        )
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = l4jax.Weights.init(random.key(0), cfg)
        cache = l4jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
        with set_mesh(cfg.mesh):
            max_tokens, _, cache = l4jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        with set_mesh(cfg.mesh):
            for _ in range(2):
                next_tokens, cache = l4jax.decode_step(next_tokens, weights, cache, cfg)


if __name__ == "__main__":
    absltest.main()
