#!/usr/bin/env python3
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

"""Verify all JAX model architectures work end-to-end with random weights.

This script initializes each supported JAX model with random weights
and runs a short inference (prefill + decode) to verify the full pipeline works.
No model download is needed.

Usage:
    # Verify all models:
    python scripts/examples/jax_verify_models.py

    # Verify a specific model:
    python scripts/examples/jax_verify_models.py --model llama3

    # Verbose output:
    python scripts/examples/jax_verify_models.py --verbose
"""

import argparse
import dataclasses
import os
import sys
import time

# Configure JAX for CPU testing with multiple devices
# Use jax_num_cpu_devices (same as upstream jax-llm-examples tests)
import jax  # noqa: E402

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", int(os.environ.get("JAX_NUM_DEVICES", "4")))

from jax import numpy as jnp  # noqa: E402
from jax import random  # noqa: E402
from jax.sharding import AxisType  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

try:
    from jax.sharding import set_mesh  # noqa: E402
except ImportError:
    from jax.sharding import use_mesh as set_mesh  # noqa: E402


def _make_mesh(auto=False):
    """Creates a test mesh matching upstream jax-llm-examples patterns.

    Args:
        auto: Use AxisType.Auto (for deepseek_r1, kimi_k2).
            Other models use AxisType.Explicit per upstream tests.
    """
    axis_type = AxisType.Auto if auto else AxisType.Explicit
    return jax.make_mesh(
        (1, len(jax.devices()), 1),
        P("x", "y", "z"),
        axis_types=(axis_type,) * 3,
    )


def verify_llama3(verbose: bool = False) -> bool:
    """Verify Llama3 JAX model."""
    from oumi.models.experimental.jax_models.llama3.llama3_jax import (
        model as l3jax,
    )

    mesh = _make_mesh()
    cfg = l3jax.Config(
        embed=256,
        ffw_size=512,
        q_heads=4,
        kv_heads=4,
        num_layers=2,
        head_dim=64,
        vocab_size=128,
        max_seq_len=64,
        causal=True,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
    )
    cfg = dataclasses.replace(cfg, mesh=mesh)

    tokens = jnp.ones((1, 16), dtype=jnp.int32)
    weights = l3jax.Weights.init(random.key(0), cfg)
    cache = l3jax.KVCache.init(random.key(0), cfg, tokens.shape[0])
    with set_mesh(cfg.mesh):
        max_tokens, _, cache = l3jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        for i in range(5):
            next_tokens, cache = l3jax.decode_step(next_tokens, weights, cache, cfg)
            if verbose:
                print(f"  decode step {i + 1}: token={int(next_tokens[0, 0])}")
    return True


def verify_llama4(verbose: bool = False) -> bool:
    """Verify Llama4 JAX model."""
    from oumi.models.experimental.jax_models.llama4.llama4_jax import (
        model as l4jax,
    )

    mesh = _make_mesh()
    # Start from upstream SCOUT_CFG then shrink for testing
    base_cfg = l4jax.Config(
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
    cfg = dataclasses.replace(
        base_cfg,
        mesh=mesh,
        num_layers=4,
        embed=64,
        moe_num_experts=4,
        vocab_size=128,
    )

    tokens = jnp.ones((1, 16), dtype=jnp.int32)
    weights = l4jax.Weights.init(random.key(0), cfg)
    cache = l4jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
    with set_mesh(cfg.mesh):
        max_tokens, _, cache = l4jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        for i in range(5):
            next_tokens, cache = l4jax.decode_step(next_tokens, weights, cache, cfg)
            if verbose:
                print(f"  decode step {i + 1}: token={int(next_tokens[0, 0])}")
    return True


def verify_deepseek_r1(verbose: bool = False) -> bool:
    """Verify DeepSeek R1 JAX model."""
    from oumi.models.experimental.jax_models.deepseek_r1_jax.deepseek_r1_jax import (
        model as dsjax,
    )

    mesh = _make_mesh(auto=True)
    cfg = dsjax.Config(
        num_layers=4,
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
    cfg = dataclasses.replace(cfg, mesh=mesh)

    tokens = jnp.ones((1, 16), dtype=jnp.int32)
    weights = dsjax.Weights.init(random.key(0), cfg)
    cache = dsjax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
    # DeepSeek R1 does NOT use set_mesh context
    max_tokens, _, cache = dsjax.prefill(tokens, weights, cache, cfg)
    next_tokens = max_tokens[:, :-1]
    for i in range(5):
        next_tokens, cache = dsjax.decode_step(next_tokens, weights, cache, cfg)
        if verbose:
            print(f"  decode step {i + 1}: token={int(next_tokens[0, 0])}")
    return True


def verify_qwen3(verbose: bool = False) -> bool:
    """Verify Qwen3 JAX model."""
    from oumi.models.experimental.jax_models.qwen3.qwen3_jax import (
        model as q3jax,
    )

    mesh = _make_mesh()
    # Start from upstream MOE_CFG then shrink for testing
    base_cfg = q3jax.Config(
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
    cfg = dataclasses.replace(
        base_cfg, mesh=mesh, num_layers=2, embed=32, vocab_size=128
    )

    tokens = jnp.ones((1, 16), dtype=jnp.int32)
    weights = q3jax.Weights.init(random.key(0), cfg)
    cache = q3jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
    with set_mesh(cfg.mesh):
        max_tokens, _, cache = q3jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        for i in range(5):
            next_tokens, cache = q3jax.decode_step(next_tokens, weights, cache, cfg)
            if verbose:
                print(f"  decode step {i + 1}: token={int(next_tokens[0, 0])}")
    return True


def verify_kimi_k2(verbose: bool = False) -> bool:
    """Verify Kimi K2 JAX model."""
    from oumi.models.experimental.jax_models.kimi_k2.kimi_k2_jax import (
        model as k2jax,
    )

    mesh = _make_mesh(auto=True)
    cfg = k2jax.Config(
        num_layers=4,
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
    cfg = dataclasses.replace(cfg, mesh=mesh)

    tokens = jnp.ones((1, 16), dtype=jnp.int32)
    weights = k2jax.Weights.init(random.key(0), cfg)
    cache = k2jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
    # Kimi K2 does NOT use set_mesh context
    max_tokens, _, cache = k2jax.prefill(tokens, weights, cache, cfg)
    next_tokens = max_tokens[:, :-1]
    for i in range(5):
        next_tokens, cache = k2jax.decode_step(next_tokens, weights, cache, cfg)
        if verbose:
            print(f"  decode step {i + 1}: token={int(next_tokens[0, 0])}")
    return True


def verify_gpt_oss(verbose: bool = False) -> bool:
    """Verify GPT-OSS JAX model."""
    from oumi.models.experimental.jax_models.gpt_oss.gpt_oss_jax import (
        model as gpt_jax,
    )

    mesh = _make_mesh()
    # Start from upstream MOE_CFG then shrink for testing
    base_cfg = gpt_jax.Config(
        embed=2880,
        q_heads=64,
        kv_heads=8,
        num_layers=24,
        head_dim=64,
        vocab_size=201088,
        max_seq_len=128,
        causal=True,
        moe_ffw_size=2880,
        moe_experts_per_tok=4,
        moe_num_experts=32,
        sliding_window_size=128,
        sliding_attention_map=["sliding_attention", "full_attention"] * 12,
        ep_strategy="decode",
    )
    cfg = dataclasses.replace(
        base_cfg, mesh=mesh, num_layers=2, embed=32, vocab_size=128
    )

    tokens = jnp.ones((1, 16), dtype=jnp.int32)
    weights = gpt_jax.Weights.init(random.key(0), cfg)
    cache = gpt_jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
    with set_mesh(cfg.mesh):
        max_tokens, _, cache = gpt_jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        for i in range(5):
            next_tokens, cache = gpt_jax.decode_step(next_tokens, weights, cache, cfg)
            if verbose:
                print(f"  decode step {i + 1}: token={int(next_tokens[0, 0])}")
    return True


def verify_nemotron3(verbose: bool = False) -> bool:
    """Verify Nemotron3 JAX model."""
    from oumi.models.experimental.jax_models.nemotron3.nemotron3_jax import (
        model as n3jax,
    )

    mesh = _make_mesh()
    cfg = n3jax.Config(
        embed=64,
        q_heads=8,
        kv_heads=2,
        num_layers=2,
        head_dim=32,
        vocab_size=128,
        max_seq_len=64,
        layer_pattern="M*",
        causal=True,
        moe_ffw_size=64,
        moe_experts_per_tok=2,
        moe_num_experts=4,
        moe_shared_ffw_size=64,
        moe_router_n_groups=1,
        moe_router_topk_groups=1,
        moe_routed_scaling_factor=1.0,
        ep_strategy="decode",
        moe_gate_dtype=jnp.bfloat16,
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
    cfg = dataclasses.replace(
        cfg, mesh=mesh, quant_attn=False, quant_moe=False, quant_mamba=False
    )

    tokens = jnp.ones((1, 16), dtype=jnp.int32)
    weights = n3jax.Weights.init(random.key(0), cfg)
    cache = n3jax.KVCache.init(random.key(0), cfg, tokens.shape[0], cfg.max_seq_len)
    with set_mesh(cfg.mesh):
        max_tokens, _, cache = n3jax.prefill(tokens, weights, cache, cfg)
        next_tokens = max_tokens[:, :-1]
        for i in range(5):
            next_tokens, cache = n3jax.decode_step(next_tokens, weights, cache, cfg)
            if verbose:
                print(f"  decode step {i + 1}: token={int(next_tokens[0, 0])}")
    return True


# Map of model name -> verification function
MODELS = {
    "llama3": verify_llama3,
    "llama4": verify_llama4,
    "deepseek_r1": verify_deepseek_r1,
    "qwen3": verify_qwen3,
    "kimi_k2": verify_kimi_k2,
    "gpt_oss": verify_gpt_oss,
    "nemotron3": verify_nemotron3,
}


def main():
    """Entry point for the JAX model verification CLI."""
    parser = argparse.ArgumentParser(
        description="Verify JAX model architectures with random weights."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Verify a specific model (default: all)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-step decode output"
    )
    args = parser.parse_args()

    models_to_test = {args.model: MODELS[args.model]} if args.model else MODELS

    print(f"JAX devices: {len(jax.devices())} ({jax.devices()[0].platform})")
    print(f"Verifying {len(models_to_test)} model(s)...\n")

    results = {}
    for name, verify_fn in models_to_test.items():
        print(f"[{name}] Running prefill + 5 decode steps...")
        start = time.time()
        try:
            verify_fn(verbose=args.verbose)
            elapsed = time.time() - start
            results[name] = ("PASS", elapsed)
            print(f"[{name}] PASS ({elapsed:.1f}s)\n")
        except Exception as e:
            elapsed = time.time() - start
            results[name] = ("FAIL", elapsed)
            print(f"[{name}] FAIL ({elapsed:.1f}s): {e}\n")

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    passed = sum(1 for s, _ in results.values() if s == "PASS")
    failed = sum(1 for s, _ in results.values() if s == "FAIL")
    for name, (status, elapsed) in results.items():
        print(f"  {name:15s} {status:4s} ({elapsed:.1f}s)")
    print(f"\n{passed}/{passed + failed} models passed.")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
