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

LLAMA4_SCOUT = Config(
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
    moe_gate_dtype=jnp.float32,
    ep_strategy="decode",
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
    use_ragged_dot_kernel=True,
    dtype=jnp.bfloat16,
    norm_eps=1e-05,
    mesh=None,
    rope_theta=500000.0,
    rope_scaling_factor=8.0,
    rope_scaling_low_freq_factor=1.0,
    rope_scaling_high_freq_factor=4.0,
    rope_scaling_original_max_position_embeddings=8192,
    quant_mlp=False,
    quant_moe=False,
    quant_attn=False,
    quant_cache=True,
    quant_scale_dtype=jnp.bfloat16,
)

LLAMA4_MAVERICK = Config(
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
    moe_gate_dtype=jnp.float32,
    ep_strategy="decode",
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
    use_ragged_dot_kernel=True,
    dtype=jnp.bfloat16,
    norm_eps=1e-05,
    mesh=None,
    rope_theta=500000.0,
    rope_scaling_factor=8.0,
    rope_scaling_low_freq_factor=1.0,
    rope_scaling_high_freq_factor=4.0,
    rope_scaling_original_max_position_embeddings=8192,
    quant_mlp=False,
    quant_moe=False,
    quant_attn=False,
    quant_cache=True,
    quant_scale_dtype=jnp.bfloat16,
)
