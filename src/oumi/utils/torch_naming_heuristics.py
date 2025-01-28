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

"""Utility functions which use detect-by-name heuristics.

# TODO(OPE-303): These should be replaced with something more robust.
"""

import importlib
from typing import Any

import torch
import torch.nn as nn
import transformers

from oumi.utils.logging import logger
from oumi.utils.torch_utils import _get_parameter_names

_PARAMS_KEY = "params"
_WEIGHT_DECAY_KEY = "weight_decay"


def disable_dropout(hf_config: transformers.PretrainedConfig) -> None:
    """Detects dropout probabilities in config and sets them to 0.0.

    This essentially removes the dropout layer, which can aid the compiled model's
    speed. Dropout is normally not used for LLM training, and also hinders the
    effectiveness of model compilation. We assume any attribute with "drop" in the name
    and a float value is a dropout param. For example, this includes `attn_pdrop` and
    `summary_first_dropout` for GPT2.

    Args:
        hf_config: The HuggingFace model config.
    """
    drop_attrs = []
    for k, v in vars(hf_config).items():
        if "drop" in k and isinstance(v, float):
            setattr(hf_config, k, 0.0)
            drop_attrs.append(k)

    logger.info(
        f"Found these dropout attributes and set their values to 0.0: {drop_attrs}"
    )


def group_trainable_params(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Groups trainable params by weight decay for optimization.

    As a rule of thumb, we generally want to weight decay all 2d matrices, i.e.
    weight tensors for matmuls/embeddings, and not biases/layernorms.

    Args:
        model: The model whose parameters will be optimized.
        weight_decay: The weight decay to apply to the appropriate parameters.

    Returns:
        List[Dict[str, Any]]: A list containing two dictionaries: the first with
            parameters that should be weight decayed and the second with parameters
            that shouldn't.
    """
    # Exclude layernorm and bias tensors.
    decay_parameters = _get_parameter_names(
        model, forbidden_layer_types=[torch.nn.LayerNorm]
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # Only include trainable params.
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # Group by weight decay.
    return [
        {
            _PARAMS_KEY: [p for n, p in trainable_params if n in decay_parameters],
            _WEIGHT_DECAY_KEY: weight_decay,
        },
        {
            _PARAMS_KEY: [p for n, p in trainable_params if n not in decay_parameters],
            _WEIGHT_DECAY_KEY: 0.0,
        },
    ]


def guess_transformer_layer_cls(model: nn.Module) -> type[nn.Module]:
    """Guess the transformer layer class based on the model architecture."""
    for module in model.modules():
        for layer_pattern in ["layer", "block", "transformerlayer"]:
            layer_name = str(type(module)).lower()

            if layer_pattern in layer_name and "layernorm" not in layer_name:
                return type(module)

    raise ValueError(
        "Unable to guess transformer layer class. Please specify it explicitly."
    )


def _parse_transformer_layer_cls_string(class_names: str) -> list[str]:
    result: list[str] = []
    for class_name in class_names.split(","):
        class_name = class_name.strip()
        if class_name:
            result.append(class_name)
    return result


def resolve_transformer_layer_cls_string_as_module_set(
    class_names: str,
) -> set[type[nn.Module]]:
    """Get a module class from its string name."""
    result: set[type[nn.Module]] = set()
    for class_name in _parse_transformer_layer_cls_string(class_names):
        parts = class_name.rsplit(".", maxsplit=1)
        if len(parts) == 1:
            # Assume `transformers` by default.
            module_name = "transformers"
        else:
            module_name, class_name = parts
        module = importlib.import_module(module_name)
        transformer_cls = getattr(module, class_name)
        result.add(transformer_cls)

    return result


def simplify_transformer_layer_cls_string(class_names: str) -> str:
    """Replaces fully-qualified class names with pure class names.

    For example, converts 'foo.Block,foo.util.Decoder' to 'Block,Decoder'.

    The `accelerate` library expects the simplified format, while OUMI trainer requires
    fully-qualified class names.
    """
    result = []
    for class_name in _parse_transformer_layer_cls_string(class_names):
        parts = class_name.rsplit(".")
        result.append(parts[-1])
    return ",".join(result)
