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

import jax.numpy as jnp

from .model import Config


QWEN3_30B_A3B = Config(
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
    moe_gate_dtype=jnp.float32,
    ep_strategy="decode",
    mlp_ffw_size=6144,
    mlp_layer_idxs=[],
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
    use_ragged_dot_kernel=False,
    dtype=jnp.bfloat16,
    norm_eps=1e-06,
    mesh=None,
    rope_theta=1000000.0,
    quant_moe=False,
    quant_mlp=False,
    quant_attn=False,
    quant_cache=True,
    quant_scale_dtype=jnp.bfloat16,
)


QWEN3_235B_A22B = Config(
    embed=4096,
    q_heads=64,
    kv_heads=4,
    num_layers=94,
    head_dim=128,
    vocab_size=151936,
    max_seq_len=128,
    causal=True,
    moe_ffw_size=1536,
    moe_experts_per_tok=8,
    moe_num_experts=128,
    moe_gate_dtype=jnp.float32,
    ep_strategy="decode",
    mlp_ffw_size=12288,
    mlp_layer_idxs=[],
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
    use_ragged_dot_kernel=False,
    dtype=jnp.bfloat16,
    norm_eps=1e-06,
    mesh=None,
    rope_theta=1000000.0,
    quant_moe=False,
    quant_mlp=False,
    quant_attn=False,
    quant_cache=True,
    quant_scale_dtype=jnp.bfloat16,
)

QWEN3_32B = Config(
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
    moe_gate_dtype=jnp.float32,
    ep_strategy="decode",
    mlp_ffw_size=25600,
    mlp_layer_idxs=[],
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
    use_ragged_dot_kernel=False,
    dtype=jnp.bfloat16,
    norm_eps=1e-06,
    mesh=None,
    rope_theta=1000000,
    quant_moe=False,
    quant_mlp=False,
    quant_attn=False,
    quant_cache=True,
    quant_scale_dtype=jnp.bfloat16,
)
