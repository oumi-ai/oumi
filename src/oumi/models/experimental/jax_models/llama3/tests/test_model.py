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

from llama3_jax import model as l3jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)

CFG = l3jax.Config(
    embed=4096,
    ffw_size=14336,
    q_heads=32,
    kv_heads=8,
    num_layers=32,
    head_dim=128,
    vocab_size=128,  # actually 128256
    max_seq_len=128,
    causal=True,
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
)


class TestModel(parameterized.TestCase):
    def setUp(self):
        self.mesh = jax.make_mesh(
            (1, len(jax.devices()), 1),
            P("x", "y", "z"),
            axis_types=(AxisType.Explicit,) * 3,
        )
        self.small_cfg = dataclasses.replace(
            CFG, mesh=self.mesh, num_layers=2, embed=256
        )

    @parameterized.product(quant=[False, True])
    def test_model_init(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quant_layer=quant)
        weights = l3jax.Weights.init(random.key(0), cfg)
        del weights

    @parameterized.product(quant=[False, True])
    def test_cache_init(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quant_cache=quant)
        cache = l3jax.KVCache.init(random.key(0), cfg, 2)
        del cache

    @parameterized.product(quant=[False, True])
    def test_init_hashing(self, quant):
        cfg = dataclasses.replace(self.small_cfg, quant_cache=quant)
        hash_fn = lambda x: hash(tuple(jax.tree.leaves(x, is_leaf=l3jax.is_param)))
        with self.subTest("Testing weights abstract and shardings hashing"):
            abstract = l3jax.Weights.abstract(cfg)
            abstract2 = l3jax.Weights.abstract(cfg)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = l3jax.Weights.shardings(cfg)
            shardings2 = l3jax.Weights.shardings(cfg)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

        with self.subTest("Testing kv-cache abstract and shardings hashing"):
            abstract = l3jax.KVCache.abstract(cfg, 2)
            abstract2 = l3jax.KVCache.abstract(cfg, 2)
            self.assertEqual(hash_fn(abstract), hash_fn(abstract2))
            shardings = l3jax.KVCache.shardings(cfg, 2)
            shardings2 = l3jax.KVCache.shardings(cfg, 2)
            self.assertEqual(hash_fn(shardings), hash_fn(shardings2))

    @parameterized.product(quant_weights=[False, True], quant_cache=[True, False])
    def test_prefill_decode(self, quant_weights, quant_cache):
        cfg = dataclasses.replace(
            self.small_cfg, quant_layer=quant_weights, quant_cache=quant_cache
        )
        tokens = jnp.ones((1, 32), dtype=jnp.int32)
        weights = l3jax.Weights.init(random.key(0), cfg)
        cache = l3jax.KVCache.init(random.key(0), cfg, tokens.shape[0])
        with set_mesh(cfg.mesh):
            max_tokens, _, cache = l3jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        with set_mesh(cfg.mesh):
            for _ in range(2):
                next_tokens, cache = l3jax.decode_step(next_tokens, weights, cache, cfg)


if __name__ == "__main__":
    absltest.main()
