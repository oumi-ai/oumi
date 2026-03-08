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

import copy
import dataclasses
import json
import random as pyrandom
from pathlib import Path

import jax
import numpy as np
import torch
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 4)
torch.use_deterministic_algorithms(
    False
)  # non-deterministic is necessary for deepseek reference

from deepseek_r1_jax import chkpt_utils as utils
from deepseek_r1_jax import model as dsjax
from deepseek_r1_jax.third_party import modeling_deepseek as deepseek

config = deepseek.DeepseekV3Config()
cfg = utils.convert_config(config)
config_json = json.loads(
    (
        Path(__file__).parents[1] / "deepseek_r1_jax" / "third_party" / "r1_config.json"
    ).read_text()
)
config.rope_scaling = config_json["rope_scaling"]
config.rope_scaling_factor = config_json["rope_scaling"]["factor"]
config.rope_scaling_type = config_json["rope_scaling"]["type"]
for k, v in config_json["rope_scaling"].items():
    setattr(config, f"rope_{k}", v)


def err_fn(x, y, axis=-1):
    x, y = np.array(x), np.array(y)
    diff = jnp.linalg.norm(x.astype(jnp.float32) - y.astype(jnp.float32), axis=axis)
    norm = jnp.linalg.norm(y.astype(jnp.float32), axis=axis)
    return diff / (norm + 1e-9)


mesh = jax.make_mesh((1, 2, jax.device_count() // 2), P("x", "y", "z"))
cfg = dataclasses.replace(cfg, mesh=mesh, rules=dsjax.ShardingRules())
cfg = dataclasses.replace(
    cfg, quantize_mlp=False, quantize_attn=False, quantize_moe=False
)

_replicate = lambda x, mesh: jax.device_put(x, NamedSharding(mesh, P()))
replicate = lambda *args: jax.tree.map(
    lambda z: _replicate(z, mesh=cfg.mesh), args if len(args) > 1 else args[0]
)

config_small = copy.deepcopy(config)
config_small.hidden_size = 128
config_small.intermediate_size = 256
config_small.moe_intermediate_size = 128 * 2
config_small.num_experts_per_tok = 2
config_small.num_hidden_layers = 5
config_small.n_routed_experts = 16
config_small.n_group = 4
cfg_small = dataclasses.replace(
    cfg,
    moe_ffw_size=config_small.moe_intermediate_size,
    num_experts_per_tok=config_small.num_experts_per_tok,
    n_routed_experts=config_small.n_routed_experts,
    n_group=config_small.n_group,
    embed=config_small.hidden_size,
    ffw_size=config_small.intermediate_size,
    num_layers=config_small.num_hidden_layers,
)


def _set_seed(seed):
    np.random.seed(seed), torch.manual_seed(seed), pyrandom.seed(seed)
    return iter(random.split(random.key(seed), 8192))


class TestNumerics(parameterized.TestCase):
    def setUp(self):
        pass

    @parameterized.product(seed=[0, 1, 2])
    def test_rms_norm(self, seed):
        n = 256
        keyit = _set_seed(seed)
        normlayer = deepseek.DeepseekV3RMSNorm(n, eps=1e-6)
        gamma = utils.t2j(list(normlayer.parameters())[0].data)
        x = random.normal(next(keyit), (4, n))
        y1 = dsjax.rms_norm(x, gamma)
        y2 = utils.t2j(normlayer.forward(utils.j2t(x)).detach())
        np.testing.assert_allclose(y1, y2, atol=1e-3, rtol=1e-2)

    @parameterized.product(seed=[0, 1, 2])
    def test_rotary_embeddings(self, seed):
        keyit = _set_seed(seed)
        head_dim = config.qk_rope_head_dim
        max_position_embeddings = 768
        rotary_embed = deepseek.DeepseekV3YarnRotaryEmbedding(
            head_dim,
            max_position_embeddings,
            config.rope_theta,
            scaling_factor=cfg.rope_scaling_factor,
            beta_fast=cfg.rope_beta_fast,
            beta_slow=cfg.rope_beta_slow,
            mscale=cfg.rope_mscale,
            mscale_all_dim=cfg.rope_mscale_all_dim,
        )
        x = random.normal(next(keyit), (4, 32, 8, 128))
        cos2, sin2 = utils.t2j(
            rotary_embed.forward(utils.j2t(x), seq_len=max_position_embeddings)
        )

        q = random.normal(next(keyit), (3, 8, max_position_embeddings, head_dim))
        k = random.normal(next(keyit), (3, 8, max_position_embeddings, head_dim))
        position_ids = jnp.broadcast_to(
            jnp.arange(max_position_embeddings), (q.shape[0], max_position_embeddings)
        )

        sin1, cos1 = dsjax.generate_pos_embeddings(position_ids, head_dim, cfg)
        q_emb1 = dsjax.apply_rotary_embedding(q, sin1, cos1)
        k_emb1 = dsjax.apply_rotary_embedding(k, sin1, cos1)

        q_emb2, k_emb2 = utils.t2j(
            deepseek.apply_rotary_pos_emb(*utils.j2t((q, k, cos2, sin2, position_ids)))
        )
        tol = dict(atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            sin1[0, ...], sin2[..., : sin2.shape[-1] // 2], **tol
        )
        np.testing.assert_allclose(
            cos1[0, ...], cos2[..., : sin2.shape[-1] // 2], **tol
        )
        np.testing.assert_allclose(q_emb1, q_emb2, **tol)
        np.testing.assert_allclose(k_emb1, k_emb2, **tol)

    @parameterized.product(seed=[0, 1, 2])
    def test_attention(self, seed):
        keyit = _set_seed(seed)
        attn_layer = deepseek.DeepseekV3Attention(config_small, 0)
        layer = utils.convert_attn_layer(attn_layer, cfg_small)
        x = random.normal(next(keyit), (2, 64, cfg_small.embed), dtype=jnp.float32)
        position_ids = jnp.broadcast_to(jnp.arange(x.shape[-2]), x.shape[:2])
        segment_ids = jnp.ones_like(x[..., 0], dtype=jnp.int32)
        sin1, cos1 = dsjax.generate_pos_embeddings(
            position_ids, cfg_small.qk_rope_head_dim, cfg_small
        )
        cfg_quant = dataclasses.replace(cfg_small, quantize_attn=True)
        layer_quant = dsjax.AttentionLayer.quantize(layer, cfg_quant)

        layer = jax.tree.map(
            jax.device_put, layer, dsjax.AttentionLayer.shardings(cfg_small)
        )
        layer_quant = jax.tree.map(
            jax.device_put, layer_quant, dsjax.AttentionLayer.shardings(cfg_quant)
        )

        x, segment_ids, layer, layer_quant, sin1, cos1 = replicate(
            x, segment_ids, layer, layer_quant, sin1, cos1
        )
        attn_out1 = dsjax.mla_attention_block(
            x, segment_ids, layer, sin1, cos1, cfg=cfg_small
        )[0]
        attn_out3 = dsjax.mla_attention_block(
            x, segment_ids, layer_quant, sin1, cos1, cfg=cfg_small
        )[0]

        attention_mask = (
            jnp.arange(x.shape[-2])[:, None] >= jnp.arange(x.shape[-2])[None, :]
        )[None, None, :, :]
        attention_mask = (
            ~jnp.broadcast_to(attention_mask, (x.shape[0], 1, x.shape[-2], x.shape[-2]))
            * -jnp.inf
        )
        attn_out2 = utils.t2j(
            attn_layer.forward(*utils.j2t((x, attention_mask)))[0].detach()
        )

        tol = dict(atol=1e-1, rtol=1e-2)
        np.testing.assert_allclose(attn_out1, attn_out2, **tol)
        np.testing.assert_allclose(attn_out3, attn_out2, **tol)

    @parameterized.product(seed=[0, 1, 2])
    def test_mlp(self, seed):
        keyit = _set_seed(seed)
        mlp_deepseek = deepseek.DeepseekV3MLP(
            config_small,
            hidden_size=cfg_small.embed,
            intermediate_size=cfg_small.ffw_size,
        ).to(torch.float32)
        mlp = utils.convert_mlp_layer(mlp_deepseek, cfg_small)
        cfg_quant = dataclasses.replace(cfg_small, quantize_mlp=True)
        mlp_quant = dsjax.MLPLayer.quantize(mlp, cfg_quant)
        x = random.normal(next(keyit), (2, 8, cfg_small.embed), dtype=jnp.bfloat16)
        y2 = utils.t2j(mlp_deepseek(utils.j2t(x.astype(jnp.float32))).detach())

        mlp = jax.tree.map(jax.device_put, mlp, dsjax.MLPLayer.shardings(cfg_small))
        mlp_quant = jax.tree.map(
            jax.device_put, mlp_quant, dsjax.MLPLayer.shardings(cfg_quant)
        )

        x = replicate(x)
        y1 = dsjax.mlp_block(x, mlp, cfg)
        y3 = dsjax.mlp_block(x, mlp_quant, cfg_quant)
        tol = dict(atol=1e-2, rtol=1e-2)
        np.testing.assert_allclose(y1, y2, **tol)
        np.testing.assert_allclose(y2, y3, **tol)

    @parameterized.product(seed=[0, 1, 2])
    def test_moe_router(self, seed):
        keyit = _set_seed(seed)
        gate = deepseek.MoEGate(config)
        gate.eval()
        gate.e_score_correction_bias.data = 1e-2 * torch.rand_like(
            gate.e_score_correction_bias.data
        )
        weight, bias = utils.t2j([x.data for x in gate.parameters()])
        assert weight.ndim == 2 and bias.ndim == 1
        weight = weight.T
        x = random.normal(next(keyit), (2, 8, cfg.embed))
        x, weight, bias = replicate(x, weight, bias)
        topk_weights1, topk_idx1 = dsjax._route_tokens_to_moe_experts(
            x, weight, bias, True, cfg
        )

        topk_idx2, topk_weights2 = gate.forward(utils.j2t(x))
        topk_idx2, topk_weights2 = utils.t2j(
            [
                y.reshape(x.shape[:2] + (-1,)).detach()
                for y in (topk_idx2, topk_weights2)
            ]
        )
        topk_idx2, topk_weights2 = jax.vmap(
            jax.vmap(lambda x, y: (x[jnp.argsort(x)], y[jnp.argsort(x)]))
        )(topk_idx2, topk_weights2)
        topk_idx1, topk_weights1 = jax.vmap(
            jax.vmap(lambda x, y: (x[jnp.argsort(x)], y[jnp.argsort(x)]))
        )(topk_idx1, topk_weights1)
        tol = dict(atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(topk_weights1, topk_weights2, **tol)

    @parameterized.product(seed=[0, 1, 2])
    def test_moe_layer(self, seed):
        keyit = _set_seed(seed)
        errs, errs_quant = [], []
        # random init MoE layers experience instabilities sometimes, get minimum error of 5 trials
        for _ in range(5):
            moe_deepseek = deepseek.DeepseekV3MoE(config_small)
            moe_deepseek.eval()
            moe_layer = utils._cast_dtype(
                utils.convert_moe_layer(moe_deepseek, cfg_small),
                dsjax.MoELayer.abstract(cfg_small),
            )
            cfg_quant = dataclasses.replace(cfg_small, quantize_moe=True)
            moe_layer_quant = dsjax.MoELayer.quantize(moe_layer, cfg_quant)

            moe_layer = jax.tree.map(
                lambda x, s: jax.device_put(x, s),
                moe_layer,
                dsjax.MoELayer.shardings(cfg_small),
            )
            moe_layer_quant = jax.tree.map(
                lambda x, s: jax.device_put(x, s),
                moe_layer_quant,
                dsjax.MoELayer.shardings(cfg_quant),
            )
            cfg_small_ = dataclasses.replace(cfg_small)
            cfg_quant_ = dataclasses.replace(cfg_quant)
            x = dsjax.rms_norm(
                random.normal(
                    next(keyit), (2, 16, cfg_small_.embed), dtype=jnp.float32
                ),
                1,
            ).astype(jnp.bfloat16)
            y2 = utils.t2j(moe_deepseek(utils.j2t(x.astype(jnp.float32))).detach())
            x = replicate(x)
            y1 = jax.jit(dsjax.moe_block_ep)(x, moe_layer, cfg_small_)
            y3 = jax.jit(dsjax.moe_block_ep)(x, moe_layer_quant, cfg_quant_)

            tol = dict(atol=1e-1, rtol=1e-3)
            errs.append(np.max(err_fn(y1, y2, axis=-1)))
            errs_quant.append(np.max(err_fn(y3, y2, axis=-1)))
        np.testing.assert_allclose(min(errs), 0, **tol)
        np.testing.assert_allclose(min(errs_quant), 0, **tol)

    @parameterized.product(seed=[0, 1, 2], layer_idx=[0, 10])
    def test_combined_layer(self, seed, layer_idx):
        keyit = _set_seed(seed)
        cfg_small_ = dataclasses.replace(cfg_small)
        errs, errs_quant = [], []
        # random init MoE layers experience instabilities sometimes, get minimum error of 5 trials
        for _ in range(5):
            layer_deepseek = deepseek.DeepseekV3DecoderLayer(config_small, layer_idx)
            layer_deepseek.eval()
            cfg_quant = dataclasses.replace(
                cfg_small_, quantize_mlp=True, quantize_attn=True, quantize_moe=True
            )
            layer = utils.convert_layer(layer_deepseek, cfg_small_)
            use_moe = dsjax.is_type(layer.mlp, dsjax.MoELayer)
            layer_quant = dsjax.Layer.quantize(layer, cfg_quant)

            layer = jax.tree.map(
                jax.device_put,
                layer,
                dsjax.Layer.shardings(cfg_small_, use_moe=use_moe),
            )
            layer_quant = jax.tree.map(
                jax.device_put,
                layer_quant,
                dsjax.Layer.shardings(cfg_quant, use_moe=use_moe),
            )

            x = dsjax.rms_norm(
                random.normal(next(keyit), (2, 16, cfg_small.embed), dtype=jnp.float32),
                1,
            ).block_until_ready()
            attention_mask = (
                jnp.arange(x.shape[-2])[:, None] >= jnp.arange(x.shape[-2])[None, :]
            )[None, None, :, :]
            attention_mask = (
                ~jnp.broadcast_to(
                    attention_mask, (x.shape[0], 1, x.shape[-2], x.shape[-2])
                )
                * -jnp.inf
            )

            y2 = utils.t2j(
                layer_deepseek.forward(
                    utils.j2t(x), attention_mask=utils.j2t(attention_mask)
                )[0].detach()
            ).block_until_ready()

            position_ids = jnp.broadcast_to(jnp.arange(x.shape[-2]), x.shape[:2])
            sin, cos = dsjax.generate_pos_embeddings(
                position_ids, cfg.qk_rope_head_dim, cfg
            )
            segment_ids = jnp.ones_like(x[..., 0], dtype=jnp.int32)
            x, segment_ids, sin, cos = replicate(x, segment_ids, sin, cos)
            y1 = jax.jit(dsjax.forward_layer)(
                x,
                segment_ids=segment_ids,
                layer=layer,
                sin=sin,
                cos=cos,
                cfg=cfg_small_,
                idx=0,
            )[0]
            y3 = jax.jit(dsjax.forward_layer)(
                x,
                segment_ids=segment_ids,
                layer=layer_quant,
                sin=sin,
                cos=cos,
                cfg=cfg_small_,
                idx=0,
            )[0]

            errs.append(np.max(err_fn(y1, y2, axis=-1)))
            errs_quant.append(np.max(err_fn(y3, y2, axis=-1)))
        np.testing.assert_allclose(min(errs), 0, atol=2e-1)
        np.testing.assert_allclose(min(errs_quant), 0, atol=2e-1)

    @parameterized.product(
        seed=[0, 1, 2], use_cache=[True, False], quantize_cache=[True, False]
    )
    def test_model(self, seed, use_cache, quantize_cache):
        keyit = _set_seed(seed)
        model_deepseek = deepseek.DeepseekV3ForCausalLM(config_small)
        model_deepseek.eval()
        cfg_quant = dataclasses.replace(
            cfg_small, quantize_attn=True, quantize_moe=True, quantize_mlp=False
        )
        weights = utils.convert_model(model_deepseek, cfg_small)
        weights_quant = dsjax.Weights.quantize(weights, cfg_quant)
        weights = jax.tree.map(
            jax.device_put, weights, dsjax.Weights.shardings(cfg_small)
        )
        weights_quant = jax.tree.map(
            jax.device_put, weights_quant, dsjax.Weights.shardings(cfg_quant)
        )

        input = random.randint(
            next(keyit), (4, 16), 0, cfg_small.vocab_size, dtype=jnp.int32
        )
        segment_ids = jnp.ones_like(input, dtype=jnp.int32)

        # causal by default
        y2 = utils.t2j(
            model_deepseek.forward(utils.j2t(input), attention_mask=None)[0].detach()
        )
        input, segment_ids = replicate((input, segment_ids))

        if use_cache:
            cache = dsjax.KVCache.init(
                next(keyit),
                dataclasses.replace(cfg_small, quantize_cache=quantize_cache),
                input.shape[0],
                32,
            )
        else:
            cache = None

        ret1 = jax.jit(dsjax.forward)(
            input, segment_ids=segment_ids, weights=weights, cfg=cfg_small, cache=cache
        )
        ret3 = jax.jit(dsjax.forward)(
            input,
            segment_ids=segment_ids,
            weights=weights_quant,
            cfg=cfg_quant,
            cache=cache,
        )

        y1 = ret1 if not use_cache else ret1[0]
        y3 = ret3 if not use_cache else ret3[0]

        err = err_fn(y1, y2, axis=-1)
        err_quant = err_fn(y3, y2, axis=-1)

        tol = dict(atol=2e-1)
        np.testing.assert_allclose(err, 0, **tol)
        np.testing.assert_allclose(err_quant, 0, **tol)


if __name__ == "__main__":
    absltest.main()
