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
import json

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import set_mesh, AxisType, PartitionSpec as P
import numpy as np

from transformers import AutoTokenizer
from gpt_oss_jax import model as gpt_jax


jax.config.update(
    "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax_cache").expanduser())
)


def encode_input(tokenizer, texts, pad_id: int = gpt_jax.PAD_ID):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_bos=True,
            add_generation_prompt=True,
        )
        for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return np.array(inputs)


if __name__ == "__main__":
    # jax.distributed.initialize()  # if you want to run multi-host
    quant = True

    ckpt_path = epath.Path("~/bucket/gpt_oss_jax/gpt_oss_20b").expanduser()
    if quant:
        ckpt_path = ckpt_path.parent / f"{ckpt_path.name}-quant"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    tp = 2
    mesh = jax.make_mesh(
        (1, tp, jax.device_count() // tp),
        ("x", "y", "z"),
        devices=jax.devices(),
        axis_types=(AxisType.Explicit,) * 3,
    )
    cfg = gpt_jax.hf_to_jax_config(json.loads((ckpt_path / "config.json").read_text()))
    cfg = dataclasses.replace(
        cfg, mesh=mesh, quant_moe=quant, quant_cache=quant, max_seq_len=2048
    )
    cfg = dataclasses.replace(cfg, use_prefill_attn_kernel=True)

    input = encode_input(
        tokenizer,
        [
            "Tell me your name",
            "What is the weather like expressed in long prose in Old English",
            "Do you like ice cream, be extremely precise",
        ]
        + ["Do you like ice cream, be extremely precise"] * (4 - 3),
    )
    weights_formats, cache_formats = gpt_jax.optimal_formats(cfg)
    weights = gpt_jax.load_pytree(ckpt_path, weights_formats)

    profile = True
    with set_mesh(cfg.mesh):
        zero_cache = gpt_jax.KVCache.init(
            random.key(1), cfg, input.shape[0], cfg.max_seq_len
        )
        zero_cache = jax.tree.map(
            lambda x, sds: jax.device_put(x, sds.sharding), zero_cache, cache_formats
        )
        next_tokens, logits, cache = gpt_jax.prefill(input, weights, zero_cache, cfg)
        curr_tokens = next_tokens.at[:, cache.iter - 1 : cache.iter].get(
            out_sharding=P(None, None)
        )
        tokens_list = []
        for i in range(1024):
            if profile and i == 2:
                jax.profiler.start_trace("/tmp/gpt_profile")
            tokens_list.append(curr_tokens)
            curr_tokens, cache = gpt_jax.decode_step(curr_tokens, weights, cache, cfg)
            if profile and i == 6:
                jax.block_until_ready(tokens_list)
                jax.profiler.stop_trace()
        tokens = np.array(jnp.concatenate(tokens_list, axis=-1))
    responses = [tokenizer.decode(row) for row in tokens]
    print("Responses:")
    for response in responses:
        print(response)
        print("\n".join(3 * ["-" * 80]))
