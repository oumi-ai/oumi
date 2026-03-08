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
from pprint import pformat

import jax
import numpy as np
from etils import epath
from jax import numpy as jnp
from jax import random
from kimi_k2_jax import chkpt_utils as utils
from kimi_k2_jax import model as k2jax


def encode_input(tokenizer, texts, pad_id: int = k2jax.PAD_ID):
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
    jax.distributed.initialize()
    ckpt_path = epath.Path("~/bucket/kimi_k2_jax_chkpt").expanduser()
    tokenizer = utils.load_tokenizer()

    mesh = jax.make_mesh(
        (1, 8, jax.device_count() // 8), ("x", "y", "z"), devices=jax.devices()
    )
    cfg = dataclasses.replace(k2jax.Config(), mesh=mesh)

    weights = utils.load_model(epath.Path(ckpt_path).expanduser(), cfg)

    input = encode_input(
        tokenizer,
        [
            "Tell me your name",
            "What is the weather like expressed in long prose in Old English",
            "Do you like ice cream, be extremely precise",
        ],
    )

    zero_cache = k2jax.KVCache.init(random.key(1), cfg, input.shape[0], cfg.max_seq_len)
    curr_tokens, logits, cache = k2jax.prefill(input, weights, zero_cache, cfg)
    curr_tokens, tokens_list = curr_tokens[:, cache.length - 1 : cache.length], []
    tokens_list = []
    for _ in range(32):
        tokens_list.append(curr_tokens)
        curr_tokens, cache = k2jax.decode_step(curr_tokens, weights, cache, cfg)
    tokens = np.array(jnp.concatenate(tokens_list, axis=-1))
    responses = [tokenizer.decode(row) for row in tokens]
    print("Responses:\n" + pformat(responses))
