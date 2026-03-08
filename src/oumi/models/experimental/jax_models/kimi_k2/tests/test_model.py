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
from jax.sharding import PartitionSpec as P
from kimi_k2_jax import model as k2jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)


SMALL_CFG = k2jax.Config(
    num_layers=2,
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


class TestModel(parameterized.TestCase):
    def setUp(self):
        self.mesh = jax.make_mesh(
            (1, len(jax.devices()), 1),
            P("x", "y", "z"),
            axis_types=(jax.sharding.AxisType.Auto,) * 3,
        )
        self.small_cfg = dataclasses.replace(SMALL_CFG, mesh=self.mesh)

    @parameterized.product(quant=[False, True])
    def test_model_init(self, quant):
        cfg = dataclasses.replace(
            self.small_cfg, quantize_attn=quant, quantize_moe=quant
        )
        weights = k2jax.Weights.init(random.key(0), cfg)
        del weights

    @parameterized.product(quant=[False, True])
    def test_init_hashing(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quantize_cache=quant)
        hash_fn = lambda x: hash(tuple(jax.tree.leaves(x, is_leaf=k2jax.is_param)))
        with self.subTest("Testing weights abstract and shardings hashing"):
            abstract = k2jax.Weights.abstract(cfg)
            abstract2 = k2jax.Weights.abstract(cfg)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = k2jax.Weights.shardings(cfg)
            shardings2 = k2jax.Weights.shardings(cfg)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

        with self.subTest("Testing kv-cache abstract and shardings hashing"):
            abstract = k2jax.KVCache.abstract(cfg, 2, cfg.max_seq_len)
            abstract2 = k2jax.KVCache.abstract(cfg, 2, cfg.max_seq_len)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = k2jax.KVCache.shardings(cfg, 2, cfg.max_seq_len)
            shardings2 = k2jax.KVCache.shardings(cfg, 2, cfg.max_seq_len)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

    @parameterized.product(quant=[False, True])
    def test_cache_init(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quantize_cache=quant)
        cache = k2jax.KVCache.init(random.key(0), cfg, 2, cfg.max_seq_len)
        del cache

    @parameterized.product(quant_weights=[False, True], quant_cache=[True, False])
    def test_prefill_decode(self, quant_weights, quant_cache):
        cfg = dataclasses.replace(
            self.small_cfg,
            quantize_attn=quant_weights,
            quantize_moe=quant_weights,
            quantize_cache=quant_cache,
        )
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = k2jax.Weights.init(random.key(0), cfg)
        cache = k2jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
        max_tokens, _, cache = k2jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        for _ in range(2):
            next_tokens, cache = k2jax.decode_step(next_tokens, weights, cache, cfg)


if __name__ == "__main__":
    absltest.main()
