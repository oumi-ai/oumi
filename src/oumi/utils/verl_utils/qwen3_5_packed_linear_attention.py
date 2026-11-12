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

"""Keep packed sequences apart in Qwen3.5's Gated DeltaNet layers.

With ``use_remove_padding``, VERL packs a micro-batch into a single row and marks
where each sequence starts by restarting its position ids at 0. Qwen3.5's
full-attention layers use those positions to keep sequences apart, but its Gated
DeltaNet (linear attention) layers run their convolution and recurrent state
straight along the row, so every sequence sees the tail of the one before it.

The patch runs each packed sequence through the DeltaNet on its own. It applies
only to a packed row: batch size 1, no attention mask, no cache, and position ids
that restart at 0 more than once. Padded batches and generation are unchanged.
The decoder layer finds the sequence starts from its own ``position_ids``, so a
gradient-checkpointing recompute splits the row exactly as the forward did.
"""

from collections.abc import Callable
from contextvars import ContextVar
from importlib import import_module
from typing import Any, cast

import torch

_MODELING_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
_DECODER_LAYER_CLASS = "Qwen3_5DecoderLayer"
_LINEAR_ATTENTION_CLASS = "Qwen3_5GatedDeltaNet"
# Every linear-attention decoder layer has this submodule, whereas its layer-type
# attribute is `layer_type` or `block_type` depending on the transformers version.
_LINEAR_ATTENTION_MODULE = "linear_attn"

_SEQUENCE_STARTS: ContextVar[list[int] | None] = ContextVar(
    "qwen3_5_packed_sequence_starts", default=None
)


def _packed_sequence_starts(
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    past_key_values: object | None,
) -> list[int] | None:
    """Returns where each packed sequence starts, or None for an unpacked row."""
    if (
        position_ids is None
        or attention_mask is not None
        or past_key_values is not None
    ):
        return None
    if position_ids.dim() != 2 or position_ids.shape[0] != 1:
        return None
    starts = torch.nonzero(position_ids[0] == 0).flatten().tolist()
    if len(starts) < 2 or starts[0] != 0:
        return None
    return starts


def _with_sequence_starts(decoder_forward: Callable[..., Any]) -> Callable[..., Any]:
    def forward(self, hidden_states, *args, **kwargs):
        if not hasattr(self, _LINEAR_ATTENTION_MODULE):
            return decoder_forward(self, hidden_states, *args, **kwargs)
        starts = _packed_sequence_starts(
            kwargs.get("position_ids"),
            kwargs.get("attention_mask"),
            kwargs.get("past_key_values"),
        )
        token = _SEQUENCE_STARTS.set(starts)
        try:
            return decoder_forward(self, hidden_states, *args, **kwargs)
        finally:
            _SEQUENCE_STARTS.reset(token)

    return forward


def _per_sequence(linear_attention_forward: Callable[..., Any]) -> Callable[..., Any]:
    def forward(self, hidden_states, *args, **kwargs):
        starts = _SEQUENCE_STARTS.get()
        if starts is None:
            return linear_attention_forward(self, hidden_states, *args, **kwargs)
        ends = starts[1:] + [hidden_states.shape[1]]
        return torch.cat(
            [
                linear_attention_forward(
                    self, hidden_states[:, start:end], *args, **kwargs
                )
                for start, end in zip(starts, ends)
            ],
            dim=1,
        )

    return forward


def _install_patch() -> None:
    try:
        modeling = cast(Any, import_module(_MODELING_MODULE))
    except ImportError:
        print(
            "[qwen3_5_packed_linear_attention] skipped: transformers has no Qwen3.5",
            flush=True,
        )
        return
    decoder_layer = getattr(modeling, _DECODER_LAYER_CLASS)
    linear_attention = getattr(modeling, _LINEAR_ATTENTION_CLASS)
    if getattr(decoder_layer, "_qwen3_5_packed_linear_attention_patched", False):
        return
    decoder_layer.forward = _with_sequence_starts(decoder_layer.forward)
    linear_attention.forward = _per_sequence(linear_attention.forward)
    decoder_layer._qwen3_5_packed_linear_attention_patched = True
    print("[qwen3_5_packed_linear_attention] installed", flush=True)


_install_patch()
