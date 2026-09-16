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

"""Gemma 4 KV-sharing compatibility for vLLM 0.19.1."""

from __future__ import annotations

import functools
import importlib.metadata
import logging
import multiprocessing
import os
import re
from collections.abc import Iterable, Iterator
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch
    from vllm.config import CacheConfig  # pyright: ignore[reportMissingImports]
    from vllm.model_executor.layers.quantization import (  # pyright: ignore[reportMissingImports]
        QuantizationConfig,
    )
    from vllm.model_executor.models.gemma4 import (  # pyright: ignore[reportMissingImports]
        Gemma4Attention,
        Gemma4ForCausalLM,
    )

logger = logging.getLogger(__name__)

ACTIVATION_ENV_VAR = "OUMI_VLLM_GEMMA4_COMPAT"
SUPPORTED_VLLM_VERSION = "0.19.1"

_INSTALL_MARKER = "_oumi_gemma4_vllm_0_19_1_compat_installed"
_SHARED_WEIGHT_PATTERN = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<component>k_proj|v_proj|k_norm)(?:\.|$)"
)


class _Gemma4Config(Protocol):
    attention_bias: bool
    num_hidden_layers: int
    num_kv_shared_layers: int


def _is_redundant_shared_weight(
    name: str, first_shared_layer: int, num_hidden_layers: int
) -> bool:
    match = _SHARED_WEIGHT_PATTERN.search(name)
    if match is None:
        return False
    layer_index = int(match.group("layer"))
    return first_shared_layer <= layer_index < num_hidden_layers


def _patch_attention(gemma4_module: ModuleType) -> None:
    attention_class = gemma4_module.Gemma4Attention
    original_init = attention_class.__init__
    original_forward = attention_class.forward

    @functools.wraps(original_init)
    def patched_init(
        self: Gemma4Attention,
        config: _Gemma4Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        use_k_eq_v: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_logits_soft_cap: float | None = None,
        prefix: str = "",
    ) -> None:
        original_init(
            self,
            config,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_position_embeddings,
            use_k_eq_v,
            cache_config,
            quant_config,
            attn_logits_soft_cap,
            prefix,
        )
        if not self.is_kv_shared_layer:
            return

        self.q_proj = gemma4_module.ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        del self.qkv_proj
        del self.k_norm
        del self.v_norm

    @functools.wraps(original_forward)
    def patched_forward(
        self: Gemma4Attention,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if not self.is_kv_shared_layer:
            return original_forward(self, positions, hidden_states, **kwargs)

        query, _ = self.q_proj(hidden_states)
        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        query = self.q_norm(query)
        query = query.flatten(-2, -1)
        query, _ = self.rotary_emb(positions, query, None)
        attention_output = self.attn(query, None, None)
        output, _ = self.o_proj(attention_output)
        return output

    attention_class.__init__ = patched_init
    attention_class.forward = patched_forward


def _patch_weight_loader(gemma4_module: ModuleType) -> None:
    causal_lm_class = gemma4_module.Gemma4ForCausalLM
    original_load_weights = causal_lm_class.load_weights

    @functools.wraps(original_load_weights)
    def patched_load_weights(
        self: Gemma4ForCausalLM,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        num_hidden_layers = self.config.num_hidden_layers
        num_shared_layers = getattr(self.config, "num_kv_shared_layers", 0)
        first_shared_layer = num_hidden_layers - num_shared_layers

        def filtered_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if _is_redundant_shared_weight(
                    name, first_shared_layer, num_hidden_layers
                ):
                    continue
                yield name, weight

        return original_load_weights(self, filtered_weights())

    causal_lm_class.load_weights = patched_load_weights


def register_gemma4_compatibility() -> None:
    """Installs the vLLM 0.19.1 Gemma 4 KV-sharing compatibility patch."""
    if os.environ.get(ACTIVATION_ENV_VAR) != "1":
        return

    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return
    if vllm_version != SUPPORTED_VLLM_VERSION:
        return

    from vllm.model_executor.models import (  # pyright: ignore[reportMissingImports]
        gemma4,
    )

    if getattr(gemma4, _INSTALL_MARKER, False):
        return

    _patch_attention(gemma4)
    _patch_weight_loader(gemma4)
    setattr(gemma4, _INSTALL_MARKER, True)
    logger.info(
        "[oumi-gemma4-compat] installed for vLLM %s in pid=%d process=%s",
        vllm_version,
        os.getpid(),
        multiprocessing.current_process().name,
    )
