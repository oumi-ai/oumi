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

from typing import Optional

import torch
import torch.utils.checkpoint
import transformers.cache_utils as transformers_cache_utils
import transformers.modeling_flash_attention_utils as transformers_flash_attention_utils
import transformers.models as transformers_models

from oumi.models.layers.zigzag import (
    is_zigzag_ring_flash_attn_available,
    zigzag_ring_flash_attn_func,
)

# Derived from:
# jzhang38/EasyContext/easy_context/zigzag_ring_attn/prepare_inputs.py


def extract_local(value, rank, world_size, device, dim=1):
    """Extract the local value from the global value."""
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.to(device)


def prepare_zigzag_ring_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, device
):
    """Prepare the inputs for zigzag ring attention."""
    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local(
        position_ids,
        rank,
        world_size,
        device,
    )
    if target_ids is not None:
        local_target_ids = extract_local(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
    }


def new_flash_attn_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    use_sliding_windows=False,
):
    """New flash attention forward."""
    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    assert attention_mask is None
    assert causal is True
    assert use_sliding_windows is False
    attn_output = zigzag_ring_flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout,
        softmax_scale,
        causal=causal,
    )

    return attn_output


def new_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[transformers_cache_utils.Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
    **kwargs,
) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """New decoder forward."""
    # Was originally LlamaFlashAttention2, but this was deleted in transformers 4.48.0.
    assert isinstance(
        self.self_attn,
        transformers_models.llama.modeling_llama.LlamaAttention,
        # Ditto but with MistralFlashAttention2.
    ) or isinstance(
        self.self_attn,
        transformers_models.mistral.modeling_mistral.MistralAttention,
    ), (
        "Please toggle on the Flash Attention 2 implementation "
        "when using zigzag ring attention monkey patch."
    )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    assert isinstance(hidden_states, torch.Tensor)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs  # type: ignore


def apply_zigzag_ring_attn_monkey_patch_llama():
    """Apply the zigzag ring attention monkey patch to llama."""
    if not is_zigzag_ring_flash_attn_available():
        raise RuntimeError(
            "Ring attention is not available. "
            "Install Flash Attention: `pip install flash-attn --no-build-isolation`."
        )

    # See also https://github.com/zhuzilin/ring-flash-attention/blob/6f1c6f7a6b40ec87cf1a80c3858c3c74b6c415c0/ring_flash_attn/adapters/hf_adapter.py#L167
    transformers_flash_attention_utils._flash_attention_forward = new_flash_attn_forward
    # (transformers_models.llama.modeling_llama.LlamaFlashAttention2.
    #   _flash_attention_forward = new_flash_attn_forward)
    transformers_models.llama.modeling_llama.LlamaDecoderLayer.forward = (
        new_decoder_forward
    )
