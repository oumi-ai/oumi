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
from etils import epath
import math
import json

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import set_mesh, AxisType, PartitionSpec as P
from jax.experimental.layout import Format, Layout
import numpy as np

from nemotron3_jax import model as n3jax


def encode_input(tokenizer, texts, pad_id: int = n3jax.PAD_ID):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True
        )
        for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return np.array(inputs)


if __name__ == "__main__":
    # jax.distributed.initialize()  # if you want to run multi-host
    quant = False

    ckpt_path = epath.Path("~/bucket/nemotron3/nemotron3_jax_30b_a3b").expanduser()
    if quant:
        ckpt_path = ckpt_path.parent / f"{ckpt_path.name}-quant"
    tokenizer = n3jax.load_tokenizer(ckpt_path)

    mesh = jax.make_mesh(
        (1, 4, jax.device_count() // 4),
        ("x", "y", "z"),
        devices=jax.devices(),
        axis_types=(AxisType.Explicit,) * 3,
    )
    cfg = n3jax.hf_to_jax_config(json.loads((ckpt_path / "config.json").read_text()))
    cfg = dataclasses.replace(
        cfg,
        mesh=mesh,
        quant_attn=quant,
        quant_moe=quant,
        quant_mlp=quant,
        quant_cache=quant,
        quant_mamba=quant,
    )
    cfg = dataclasses.replace(
        cfg, use_prefill_attn_kernel=False, max_seq_len=2048, mamba_dtype=jnp.bfloat16
    )

    decode_step = jax.jit(
        n3jax.decode_step.lower,
        donate_argnames=("cache",),
        in_shardings=Format(Layout.AUTO),
    )
    weights_formats, _ = n3jax.optimal_formats(cfg)
    weights = n3jax.load_pytree(ckpt_path, weights_formats)

    input = encode_input(
        tokenizer,
        [
            "What's 2 plus 2?",
            "What is the weather like expressed in long prose in Old English",
            "Do you like ice cream, be extremely precise",
        ],
    )

    with set_mesh(cfg.mesh):
        zero_cache = n3jax.KVCache.init(
            random.key(1), cfg, input.shape[0], cfg.max_seq_len
        )
        next_tokens, logits, cache = n3jax.prefill(input, weights, zero_cache, cfg)
        curr_tokens = next_tokens.at[:, cache.iter - 1 : cache.iter].get(
            out_sharding=P(None, None)
        )
        tokens_list = []
        for i in range(32):
            if i == 2:
                jax.profiler.start_trace("/tmp/mamba_profiles")
            tokens_list.append(curr_tokens)
            curr_tokens, cache = n3jax.decode_step(curr_tokens, weights, cache, cfg)
            if i == 5:
                jax.block_until_ready(tokens_list)
                jax.profiler.stop_trace()
        tokens = np.concatenate(jax.tree.map(np.array, tokens_list), axis=-1)
    responses = [tokenizer.decode(row) for row in tokens]
    print("Responses:")
    for response in responses:
        print(f"{response}\n--------------------------------\n")
