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

LLAMA3_8B = Config(
    embed=4096,
    ffw_size=14336,
    q_heads=32,
    kv_heads=8,
    num_layers=32,
    head_dim=128,
    vocab_size=128256,
    max_seq_len=128,
    causal=True,
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
    dtype=jnp.bfloat16,
    mesh=None,
    rope_theta=500000.0,
    rope_scaling_factor=8.0,
    rope_scaling_low_freq_factor=1.0,
    rope_scaling_high_freq_factor=4.0,
    rope_scaling_original_max_position_embeddings=8192,
    quant_layer=True,
    quant_cache=True,
    quant_scale_dtype=jnp.bfloat16,
)

LLAMA3_70B = Config(
    embed=8192,
    ffw_size=28672,
    q_heads=64,
    kv_heads=8,
    num_layers=80,
    head_dim=128,
    vocab_size=128256,
    max_seq_len=128,
    causal=True,
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
    dtype=jnp.bfloat16,
    mesh=None,
    rope_theta=500000.0,
    rope_scaling_factor=8.0,
    rope_scaling_low_freq_factor=1.0,
    rope_scaling_high_freq_factor=4.0,
    rope_scaling_original_max_position_embeddings=8192,
    quant_layer=True,
    quant_cache=True,
    quant_scale_dtype=jnp.bfloat16,
)

LLAMA3_405B = Config(
    embed=16384,
    ffw_size=53248,
    q_heads=128,
    kv_heads=8,
    num_layers=126,
    head_dim=128,
    vocab_size=128256,
    max_seq_len=128,
    causal=True,
    use_prefill_attn_kernel=False,
    use_decode_attn_kernel=False,
    dtype=jnp.bfloat16,
    mesh=None,
    rope_theta=500000.0,
    rope_scaling_factor=8.0,
    rope_scaling_low_freq_factor=1.0,
    rope_scaling_high_freq_factor=4.0,
    rope_scaling_original_max_position_embeddings=8192,
    quant_layer=True,
    quant_cache=True,
    quant_scale_dtype=jnp.bfloat16,
)
