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
    from jax.sharding import use_mesh as set_mesh
except ImportError:
    pass

from qwen3_jax import model as q3jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)

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


class TestModel(parameterized.TestCase):
    def setUp(self):
        self.mesh = jax.make_mesh(
            (1, len(jax.devices()), 1),
            P("x", "y", "z"),
            axis_types=(AxisType.Explicit,) * 3,
        )
        self.small_moe_cfg = dataclasses.replace(
            MOE_CFG, mesh=self.mesh, num_layers=2, embed=32, vocab_size=128
        )
        self.small_dense_cfg = dataclasses.replace(
            DENSE_CFG, mesh=self.mesh, num_layers=2, embed=32, vocab_size=128
        )

    @parameterized.product(moe=[True, False], quant=[False, True])
    def test_model_init(self, moe, quant):
        cfg = self.small_moe_cfg if moe else self.small_dense_cfg
        cfg = dataclasses.replace(
            cfg, quant_attn=quant, quant_moe=quant, quant_mlp=quant
        )
        weights = q3jax.Weights.init(random.key(0), cfg)
        del weights

    @parameterized.product(quant=[False, True])
    def test_init_hashing(self, quant):
        cfg = dataclasses.replace(self.small_moe_cfg, quant_cache=quant)
        hash_fn = lambda x: hash(tuple(jax.tree.leaves(x, is_leaf=q3jax.is_param)))
        with self.subTest("Testing weights abstract and shardings hashing"):
            abstract = q3jax.Weights.abstract(cfg)
            abstract2 = q3jax.Weights.abstract(cfg)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = q3jax.Weights.shardings(cfg)
            shardings2 = q3jax.Weights.shardings(cfg)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

        with self.subTest("Testing kv-cache abstract and shardings hashing"):
            abstract = q3jax.KVCache.abstract(cfg, 2, cfg.max_seq_len)
            abstract2 = q3jax.KVCache.abstract(cfg, 2, cfg.max_seq_len)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = q3jax.KVCache.shardings(cfg, 2, cfg.max_seq_len)
            shardings2 = q3jax.KVCache.shardings(cfg, 2, cfg.max_seq_len)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

    @parameterized.product(moe=[True, False], quant=[False, True])
    def test_cache_init(self, moe, quant):
        cfg = self.small_moe_cfg if moe else self.small_dense_cfg
        cache = q3jax.KVCache.init(random.key(0), cfg, 2, cfg.max_seq_len)
        del cache

    @parameterized.product(
        moe=[True, False], quant_weights=[False, True], quant_cache=[True, False]
    )
    def test_prefill_decode(self, moe, quant_weights, quant_cache):
        cfg = self.small_moe_cfg if moe else self.small_dense_cfg
        cfg = dataclasses.replace(
            cfg,
            quant_attn=quant_weights,
            quant_moe=quant_weights,
            quant_mlp=quant_weights,
            quant_cache=quant_cache,
        )
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = q3jax.Weights.init(random.key(0), cfg)
        cache = q3jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
        with set_mesh(cfg.mesh):
            max_tokens, _, cache = q3jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        with set_mesh(cfg.mesh):
            for _ in range(2):
                next_tokens, cache = q3jax.decode_step(next_tokens, weights, cache, cfg)


if __name__ == "__main__":
    absltest.main()
